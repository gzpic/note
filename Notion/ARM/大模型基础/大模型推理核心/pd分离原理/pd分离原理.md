---

---


| 等级 | 分离粒度 | 调度方式 | 特征 | 代表系统 |
| --- | --- | --- | --- | --- |
| **L1：阶段级分离（低密度）** | prefill / decode 串行但逻辑分离 | 同步执行 | 模型执行上区分两个阶段，但仍同批次 | 早期 MNN-LLM / ONNX RT |
| **L2：批次级分离（中密度）** | prefill 和 decode 各自组成独立批次 | 异步执行 | Continuous Batching；prefill 不阻塞 decode | TensorRT-LLM, vLLM |
| **L3：请求级 / Token-级分离（高密度）** | 同时调度 prefill 与多个 decode token，甚至跨请求 | Pipeline 并行 + Chunked Prefill | Prefill 分块进入 decode pipeline | SGLang, LMCache, Mooncak |

L1

- 模型 pipeline 有 prefill / decode 两个阶段；
- 但 scheduler 不支持异步重叠；
- 同一请求必须 prefill 完毕后才能进入 decode。

output = model.prefill(prompt)  # 阶段1
for _ in range(max_tokens):
token = model.decode(output)

优点：简单。

缺点：prefill 大请求会阻塞后续 decode → GPU 空转。

L2

**核心思想：参数常驻显存，数据动态分页（Paged Allocation）或统一管理。**

### 关键机制：

| 模块 | 作用 | 实现方式 |
| --- | --- | --- |
| **Static Weight Pool** | 模型权重加载一次后长期驻留 | Engine 常驻 GPU global mem |
| **Dynamic KV Pool** | KV Cache、Input、Output 等动态 buffer | 分块 (block) 或分页 (page) 管理 |
| **Scheduler / Allocator** | 动态调度 page/block 的分配与回收 | 类似虚拟内存管理 |

### 举例：vLLM 的实现

vLLM 采用 **PagedAttention + Continuous Batching**：

- 将每个请求的 KV Cache 分成固定大小的 page；
- 通过 `Physical KV Cache Pool` 管理；
- 请求完成后自动回收；
- 权重只加载一次（Parameter 固定），Data 独立回收。

🧠 内存示意：

```plain text
+------------------------------------------------+
| GPU Memory                                     |
|------------------------------------------------|
| [ Parameter Pool ]  -> 模型权重 (静态区)       |
| [ KV Page #1 ]                                |
| [ KV Page #2 ]                                |
| [ Input/Output Buffers (动态区) ]              |
+------------------------------------------------+


```

**优点：**

- 显著提升多请求并发；
- 支持跨请求 context overlap；
- 动态内存利用率高。

**代表系统：**

- TensorRT-LLM（Executor::Impl 管理权重，RequestImpl 管理数据）
- vLLM（Paged KV Cache）
- SGLang（Session 独立 KV 区）

有两个队列 prefill queue decode

队列间解耦，异步执行

| 队列 | 内容 | 生命周期 |
| --- | --- | --- |
| **Prefill Queue** | 新到达的请求，每个请求需要执行全量 prompt 的 self-attention | 一次性执行 |
| **Decode Queue** | 已经完成 prefill 的请求，每轮生成下一个 token | 多轮循环 |

> Prefill 的输出（KV Cache）是 Decode 的输入之一。
> 但注意：

- 这不是“阻塞依赖”，而是“数据可用事件触发”。
- Prefill 完成时会将生成的 KV Cache 注册到一个全局缓存管理器（KV Manager）。
- Decode Queue 并不会等待 Prefill Queue 清空，只需等待自己所需请求的 KV 就绪

## ⚙️ 六、为什么这样设计是高效的？

| 优化点 | 解释 |
| --- | --- |
| **Prefill 不阻塞 Decode** | Prefill 队列可持续处理新请求，不必等 decode 完成 |
| **Decode 聚合执行** | 多请求共享同一 kernel 执行，减少 kernel launch overhead |
| **KV Cache 独立生命周期** | Prefill 结束后立即可被 decode 访问，无需复制 |
| **GPU 重叠利用率高** | Prefill 流计算 + Decode 流采样可以 overlap |

### **3️⃣ L3：多级异构 PD 分离（高密度）**

这是**未来趋势**，实现**跨设备 + 跨请求 + 跨实例共享**。

核心思想是：

> Parameter 和 Data 不仅逻辑上分离，还物理上可分布于不同设备和存储层（GPU/CPU/NVMe）。

### 实现机制：

| 层级 | 内容 | 举例 |
| --- | --- | --- |
| **GPU 层** | 热数据（当前活跃请求的 KV Cache） | KV Cache Hot Page |
| **CPU 层** | 冷数据（近期请求历史、LRU Page） | LMCache / HiCacheFile |
| **NVMe 层** | 超大规模模型权重或跨请求 KV 共享 | Mooncake、DeepSpeed-Inference Offload |

🧠 内存拓扑：

```plain text
[ GPU ]: Hot KV Pages + Compute Kernel
[ CPU ]: Warm KV Cache Pool + Prefetch Thread
[ NVMe ]: Cold Storage + Shared KV Store


```

**高密度分离带来的好处：**

- 允许在小 GPU 显存下运行大模型；
- 多请求共享权重与部分 KV；
- 可跨节点复用 prefill；
- 显存利用率高达 95% 以上。

**代表系统：**

- LMCache（跨 request KV Cache 共享）
- HiCacheFile（文件级 KV Cache 存储）
- Mooncake（异构 KV 管理 + multi-tier memory）
- DeepSpeed-Inference（ZeRO-Inference + offload）


一句话：把 **Prefill** 切成可微调度的 **chunks**，让它与多个请求的 **Decode(逐 token)** 在**同一时间轴**上交错执行；同时，Decode 本身也按 **token-step** 作为最小调度单元在多个请求间穿插。这样能把 GPU 填满、TTFT 极低、吞吐最高。



# 2. 调度“最小粒度”（为什么叫高密度）

- **Prefill 粒度**：`chunk_len`（通常 128~1024 token，可自适应）
- **Decode 粒度**：**单 token step**（每轮对活跃请求统一推进 1 个 token）
- **交错原则**：同一个 **scheduler tick** 可同时派发：
    - 一个（或多个）Prefill-Chunk 批次
    - 一个 Decode 批次（混合多个请求的一步 token）
→ 通过不同 CUDA stream 或 CUDA Graph Node Overlap。

| **Prefill 不阻塞 Decode** | Prefill 队列可持续处理新请求，不必等 decode 完成 |
| --- | --- |
|   |   |

| **Decode 聚合执行** | 多请求共享同一 kernel 执行，减少 kernel launch overhead |
| --- | --- |

| **KV Cache 独立生命周期** | Prefill 结束后立即可被 decode 访问，无需复制 |
| --- | --- |

| **GPU 重叠利用率高** | Prefill 流计算 + Decode 流采样可以 overlap |
| --- | --- |


[[vllm pd]]