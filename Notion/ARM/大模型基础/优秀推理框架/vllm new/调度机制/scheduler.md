---

---
在显存容量（Block）和计算能力（Budget）有限的情况下，尽可能塞入更多的请求，同时还要保证每个请求都能顺畅地生成 Token。

在源码中，调度策略主要分为两类：**默认调度（Default）** 和 **分块预填充调度（Chunked Prefill）**。

---

### 一、 调度的核心约束：`SchedulingBudget`

在开始调度前，必须先看“钱包”里有多少钱。vLLM 用 `SchedulingBudget` 类来管理本次 Step 的预算：

1. `**token_budget**`** (Token 预算)**:
    - 本次推理最多处理多少个 Token？（通常受限于 GPU 计算能力，如 `max_num_batched_tokens`）。
    - *例如：预算 4096 个 Token。如果有一个 Prompt 长 4000，那就没剩多少给别人了。*
2. `**max_num_seqs**`** (并发预算)**:
    - 本次推理最多并行处理多少个请求？（受限于系统配置）。
3. `**block_manager**`** (显存预算)**:
    - 虽然不在 Budget 类里，但这是最硬的物理约束。调度每一步都要问 BlockManager：“还有空闲显存块吗？”

---

### 二、 策略 A：默认调度 (`_schedule_default`)

这是 vLLM 早期版本的逻辑。它的特点是**“新来的优先（Prefill First）”**（在没有 Swapped 请求的情况下）。

**调度顺序流：**

4. **优先处理 Prefill (**`**_schedule_prefills**`**)**:
    - 如果 `swapped` 队列为空，调度器会优先看 `waiting` 队列。
    - 它会尽可能多地把新请求（Prompt）塞入 Budget。
    - **痛点**：如果来了一个 8000 Token 的长 Prompt，它会瞬间耗尽本次 Step 的 `token_budget`。
5. **处理 Decode (**`**_schedule_running**`**)**:
    - 接着处理正在生成中的请求。
    - **问题**：如果步骤 1 把预算花光了，或者显存占满了，这里的 Decode 请求就会被**阻塞**（本次 Step 无法生成 Token），或者触发**抢占**。
    - *这就导致了用户感觉“卡顿”：正在生成的字突然停了一下，因为 GPU 忙着处理别人的长 Prompt 去了。*
6. **处理 Swap (**`**_schedule_swapped**`**)**:
    - 最后尝试把被踢出去的请求接回来。

---

### 三、 策略 B：分块预填充调度 (`_schedule_chunked_prefill`)

这是目前更先进、更推荐的策略（需配置开启）。它的核心思想是**“老客户优先（Decode First）”** 和 **“大块切小（Chunking）”**。

这解决了默认策略中“长 Prompt 阻塞生成”的问题。

**调度顺序流（代码逻辑）：**

7. **雷打不动：优先处理 Decode (**`**_schedule_running**`**)**:
    - **代码位置**：`Scheduler._schedule_chunked_prefill`
    - **逻辑**：先保证正在 GPU 上跑的请求能生成下一个 Token。这保证了 **Inter-token Latency（字与字之间的延迟）** 是平滑的，用户体验最好。
    - 如果显存不足，依然会触发抢占。
8. **尝试捞人：处理 Swap (**`**_schedule_swapped**`**)**:
    - 如果有预算和显存，把之前被踢到 CPU 的请求搬回来。
9. **最后处理新请求：Prefill (**`**_schedule_prefills**`**)**:
    - 利用**剩下**的预算来处理新请求。
    - **关键技术：Chunking（分块）**。
    - 如果剩下了 500 Token 的预算，但新请求的 Prompt 有 2000 Token，调度器**不会**拒绝或阻塞，而是**只处理前 500 个 Token**。
    - 剩下的 1500 个 Token 等下一个 Step 再处理。

> 代码中的分块逻辑：
在 _get_num_new_uncached_and_cached_tokens 中，会调用 _chunk_new_tokens_to_schedule，根据剩余的 budget 强行截断 Prompt 的处理长度。

---

### 四、 详细调度决策流程 (代码级)

不管是哪种策略，处理单个队列（如 `running` 或 `waiting`）时的内部逻辑都是类似的。我们以 `**_schedule_running**` 为例，看它是如何精打细算的：

Python

`# 伪代码逻辑解析
def _schedule_running(self, budget):
    # 1. 准备输出容器
    outputs = SchedulerRunningOutputs()
    
    # 2. 遍历运行队列 (按 FCFS 顺序)
    while self.running:
        seq_group = self.running[0] # 取出队头
        
        # 3. 计算开销
        # Decode 阶段通常需要 1 个新 Token 的计算预算
        num_new_tokens = 1 
        
        # 4. 检查预算 (Budget Check)
        if budget.token_budget < num_new_tokens:
            break # 没钱了，本次 Step 结束，剩下的人下个 Step 再跑
            
        # 5. 检查显存 (Memory Check - 最关键一步)
        # 问 BlockManager：能不能给这个 Token 分配物理块？
        if not self.block_manager.can_append_slots(seq_group):
            # 显存不够！触发之前讲过的“抢占机制”
            self._preempt(...) 
            continue
            
        # 6. 成功调度
        self.block_manager.append_slots(seq_group) # 真正占位
        budget.subtract(...) # 扣钱
        outputs.decode_seq_groups.append(seq_group) # 加入成功列表
        
    return outputs`

### 五、 总结：vLLM 调度的哲学

10. **显存是第一生产力**：
调度器的所有复杂逻辑（抢占、Swap、分块），本质上都是为了解决 **KV Cache 显存不足** 的问题。BlockManager 是调度器背后的“账房先生”。
11. **吞吐量 vs 延迟**：
    - **默认策略**倾向于**吞吐量**（尽快把 Prompt 也就是计算密集型任务跑完）。
    - **Chunked Prefill** 倾向于**延迟稳定性**（保证正在生成的任务不卡顿，把长 Prompt 这种“重活”分摊到每一帧去做）。
12. **连续批处理 (Continuous Batching)**：
vLLM 的调度是 **Step 级别** 的。它不像传统 Serving 那样等一批请求全跑完再接下一批。在一个 Step 中，可能包含：
    - 请求 A 的第 100 个 Token（Decode）。
    - 请求 B 的第 5 个 Token（Decode）。
    - 请求 C 的 Prompt 的前 500 个 Token（Chunked Prefill）。
这就是 vLLM 效率高的根本原因。