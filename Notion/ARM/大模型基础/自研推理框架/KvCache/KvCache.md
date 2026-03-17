---

---
1. vllm的分块
2. mnn的落盘
3. tensor复用
4. lmcache管理
5. 参考sglang的三级缓存架构

| **特性** | **SGLang** | **vLLM** | **TensorRT-LLM** |
| --- | --- | --- | --- |
| **方案名称** | **Radix Attention (树)** | **Automatic Prefix Caching (APC)** | **KV Cache Reuse** |
| **数据结构** | **前缀树 (Radix Tree)** | **哈希块表 (Hash Block Table)** | **哈希块表 / 页表** |
| **复用粒度** | **Token 级** (极致灵活) | **Block 级** (需对齐) | **Block 级** (需对齐) |
| **显存溢出时** | LRU 淘汰 (丢弃) | LRU 淘汰 (丢弃) | **支持 Offload 到 CPU (保存)** |
| **适用场景** | 复杂 Agent、多轮对话分叉、非对齐前缀 | 高并发、公共 API、短 System Prompt | **企业级部署、超长文档(RAG)、显存受限场景** |

1.整体设计思路

这是基于我们所有讨论（包括你的 Review 修正和锦上添花的三点建议）汇总的 **最终版架构设计规范（System Architecture Specification）**。

这份表格可以直接作为 **Jetson Orin Nano Super - StreamLLM over TRT-LLM** 项目的工程落地红线标准。

---

### 1. 核心架构设计规范 (Core Architecture Specs)

| **核心模块 (Module)** | **原始 TRT-LLM 行为** | **StreamLLM 改造方案 (Final)** | **工程实施约束 / 关键细节** |
| --- | --- | --- | --- |
| **内存布局**<br>(Memory Layout) | **线性 Append**<br>持续申请新 Block，直到 `max_num_blocks` 耗尽报错。 | **静态 Ring Buffer**<br>预分配固定容量，逻辑上循环，物理上常驻。 | **容量硬绑定公式**：<br>$Size = (N_{sink} + N_{rolling}) \times Layers \times Heads \times Dim \times 2B$<br>⚠️ **UMA 约束**：必须确保 Buffer 内存 Pin 在 Device 侧，严禁 CPU 缺页迁移。 |
| **分配策略**<br>(Allocation) | **Append Only**<br>从 Free List 取空闲块。 | **Recycle on Boundary**<br>复用 Rolling 窗口中最旧的物理 Block ID 进行覆写。 | **时序红线 (Step Boundary)**：<br>Block 的回收/覆写必须发生在 Decode Step 之间。<br>❌ 严禁在 Kernel 计算 Attention 时修改 Block 状态。 |
| **寻址映射**<br>(Addressing) | **全量物理映射**<br>`block_ids` 包含所有历史 Block。 | **逻辑窗口映射**<br>`block_ids` 仅包含 `[Sink] + [Current Rolling]`。 | **映射逻辑**：<br>`Runtime_Table` = `[Sink_Phy_IDs]` + `[Rolling_Phy_IDs]`<br>Host 端维护逻辑指针，Device 端只看整理好的 Table。 |
| **位置编码**<br>(RoPE / PosID) | **隐式耦合**<br>常用 `step` 或 `cache_len` 推导。 | **语义/物理彻底解耦**<br>物理存放在 Slot $i$ (可能被复用)，语义位置是 $T$ (全局递增)。 | **Plugin 避坑**：<br>必须检查 Fused Attention Plugin，**强制**使用外部传入的 `position_ids` Tensor。<br>❌ 禁止使用 Kernel 内部自动推导的 PosID。 |
| **数据搬运**<br>(Data Mov.) | **Swap / Migration**<br>显存不足时可能触发 Swap。 | **零数据移动 (Zero Data Movement)**<br>仅修改 Host 端的 `int32` 索引表，显存内无任何 `Memcpy`。 | **带宽保护**：<br>Orin 带宽宝贵，此设计确保 Attention 计算的数据吞吐量为常数 $O(Window)$，而非 $O(Length)$。 |
| **调度策略**<br>(Scheduling) | **Request-Level Eviction**<br>显存不足杀掉整个 Request。 | **Token-Level Rolling Eviction**<br>上层 Manager 主动丢弃中间 Token。 | **层级明确**：<br>底层 Engine 认为在跑“短 Context”任务；上层 Manager 负责“欺骗”Engine 并维护长文本逻辑。 |

### 2. 系统边界与假设 (Constraints & Assumptions)

这部分定义了系统的“非功能性约束”，防止误用导致系统崩溃。

| **约束维度** | **决策结论** | **技术原因 (Rationale)** |
| --- | --- | --- |
| **分支策略**<br>(Branching) | **不支持 Beam Search / Tree of Thoughts**<br>仅支持单链解码 (Greedy / Sampling)。 | Ring Buffer 依赖单一的 Head/Tail 指针。多分支会导致对“旧 Block”的回收产生竞态条件 (Race Condition)，且无法实现 Zero-Copy。 |
| **适用阶段**<br>(Scope) | **仅优化 Decode Phase**<br>Prefill 阶段仍采用标准处理或一次性截断。 | Prefill 是计算密集型且一次性的；StreamLLM 解决的是 Decode 阶段随时间积累的显存爆炸问题。 |
| **硬件对齐**<br>(Alignment) | **Block Size 必须 = 16 或 32**<br>禁止奇数或非 2 的幂次。 | 必须与 NVIDIA Warp (32 threads) 和 Memory Transaction (128 bytes) 对齐，避免 Kernel 内部分支判定导致的性能损耗。 |
| **精度模式**<br>(Precision) | **推荐 Int8 KV Cache** | 在 Orin Nano (8GB) 上，StreamLLM (控制数量) + Int8 (压缩体积) 是实现可用长文本的唯一组合。 |

### 3. Jetson Orin Nano 推荐参数 (Reference Configs)

针对 Orin Nano Super (8GB Unified Memory) 的建议“出厂设置”。

| **参数项** | **推荐值** | **备注** |
| --- | --- | --- |
| **Block Size** | **16** | 粒度够细，减少 Rolling 滚动时的显存浪费。 |
| **Sink Tokens** | **64** (4 blocks) | 保证 Attention 初始锚点稳定性 (Safety Buffer)。 |
| **Rolling Window** | **2048 ~ 3072** | 在 8GB 显存下，配合 Int8 量化，除去模型权重(7B Int4)后剩下的安全空间。 |
| **Batch Size** | **1** | 端侧交互通常是单用户的，Batch=1 能最大化单次响应的窗口长度。 |

### ✅ Sign-off Conclusion

此设计方案在 **理论层** (StreamLLM)、**系统层** (TRT-LLM Runtime) 和 **硬件层** (Jetson Orin SoC) 之间取得了最佳平衡。

- **Done**: 无限长度生成、恒定显存占用、恒定推理延迟。
- **Prevented**: OOM 崩溃、带宽耗尽卡顿、Race Condition。
- **Trade-off**: 放弃了 Beam Search 和部分长距离依赖召回能力（StreamLLM 原生特性）。

我们现在的设计涵盖了两种互斥的编译模式：**Stream Mode (流式/监控)** 和 **Agent Mode (智能体/多轮交互)**。

| **核心维度** | **Stream Mode (原有方案)** | **Agent Mode (新增 Radix Tree)** | **Tiered Storage (分级缓存策略)** |
| --- | --- | --- | --- |
| **内存拓扑**<br>(Topology) | **静态环形缓冲区 (Ring Buffer)**<br>O(1) 复杂度，物理地址固定。 | **动态基数树 (Radix Tree)**<br>树状结构，支持前缀共享 (Prefix Sharing)。 | **Hot**: GPU Pinned Memory (Pinned)<br>**Warm**: CPU Paged Memory (Unpinned)<br>**Cold**: NVMe SSD (Serialized) |
| **分配策略**<br>(Allocation) | **Recycle (覆写)**<br>覆盖最旧的块。 | **Best-Fit + Reference Count**<br>查找最长公共前缀，无匹配则申请新块。 | **Eviction Policy**: <br>L1 (Hot) 满 -> 降级到 L2 (Warm)<br>L2 (Warm) 满 -> 序列化到 L3 (Cold) |
| **适用场景**<br>(Use Case) | 无限长度对话、小说续写、长时间伴随。 | Agent 思考、Tool Use 回溯、多轮 Few-Shot。 | **冷启动加速**：从 NVMe 瞬间恢复 Agent 记忆。<br>**显存扩展**：利用 Orin 64GB SSD 扩展上下文。 |
| **寻址映射**<br>(Addressing) | Window Mapping (`block_ids` 是滑窗)。 | Tree Traversal (根据 Token Hash 走树)。 | **虚拟地址表 (VAT)**：<br>Block ID 指向的可能是 GPU 指针，也可能是 NVMe 文件偏移量。 |
| **编译选项**<br>(CMake) | `-DENABLE_STREAM_RING=ON` | `-DENABLE_RADIX_TREE=ON` | `-DENABLE_TIERED_CACHE=ON` |
| **Orin 优化** | Zero-Copy (只改指针)。 | Zero-Redundant (去重)。 | **Async Prefetch**: 利用 DMA 引擎在计算时后台从 NVMe 预取数据。 |

设计的文档：

# System Design Document: StreamLLM Integration for TRT-LLM (Jetson Orin Nano)

Version: 2.0 (Final Engineering Spec)

Target Hardware: NVIDIA Jetson Orin Nano Super (8GB Unified Memory)

Target Framework: TensorRT-LLM (v0.8.0 or later)

Objective: Implement infinite streaming generation with constant memory usage and constant latency via "Sink + Rolling" KV Cache strategy.

---

## 1. System Architecture Overview

This module implements a **Zero-Copy, Ring-Buffer managed KV Cache** on top of the existing TRT-LLM PagedAttention kernel.

### 1.1 Core Principles

- **Static Allocation:** Physical KV blocks are allocated once at startup. No runtime `malloc/free`.
- **Logical-Physical Decoupling:** The model sees a continuous context window; the hardware sees a static set of blocks reused in a ring buffer pattern.
- **Explicit Positioning:** RoPE position IDs are decoupled from physical memory slots to support StreamLLM logic.

### 1.2 Resource Constraints (Orin Nano)

- **Memory Paging:** All KV blocks must be pinned to GPU memory (Unified Memory Architecture) to prevent page faults.
- **Branching:** **NO** support for Beam Search. Only Greedy/Sampling (Single-branch decoding).
- **Latency:** Must guarantee $O(1)$ memory operations per step.

---

## 2. Data Structures & State Management

**Context:** These structures typically reside in the C++ Runtime (e.g., `BlockManager` or `GptScheduler`).

### 2.1 Configuration (`StreamLLMConfig`)

C++

`struct StreamLLMConfig {
    bool enabled = true;
    int sink_blocks = 4;        // Keep first N blocks (e.g., 64 tokens)
    int max_window_blocks = 128;// Total capacity (Sink + Rolling)
    int block_size = 16;        // HARD CONSTRAINT: Must be 16 or 32
};`

### 2.2 Runtime State (`RingBufferState`)

Attached to each active `Sequence` / `Request`.

C++

`struct RingBufferState {
    // Indices of physical blocks allocated to this request
    std::vector<int> physical_block_ids; 
    
    // Logical pointer for the rolling window
    // Points to the index in 'physical_block_ids' where the NEXT rolling token goes
    int rolling_write_pointer = 0; 
    
    // Current logical step (total tokens generated)
    int64_t current_step = 0;
};`

---

## 3. Implementation Logic (The "How-To")

### 3.1 Memory Allocation Strategy (The Ring)

**Module:** `BlockManager::allocate_blocks` / `GptScheduler`

**Logic Flow:**

6. **Pre-fill Phase:** Behave normally. Append blocks linearly until `max_window_blocks` is reached.
7. **Decode Phase (Step Boundary):**
    - **IF** `used_blocks < max_window_blocks`:
        - Allocate new block from global free pool.
        - Append to `physical_block_ids`.
    - **ELSE (StreamLLM Active):**
        - **DO NOT** allocate memory.
        - Identify the **Victim Block** logic:
            - Skip `sink_blocks` (indices `0` to `sink-1` are immutable).
            - Calculate rolling index: `idx = sink_blocks + (rolling_write_pointer % rolling_capacity)`.
        - **Action:** Reuse `physical_block_ids[idx]` as the target for the current step.
        - Increment `rolling_write_pointer`.

### 3.2 Block Table Construction (The "Illusion")

**Module:** `RuntimeBuffers` (Preparation of tensors sent to GPU)

**Requirement:** The `block_table` sent to the kernel must appear sequential but map to the ring buffer.

**Algorithm:**

C++

`// Construct the block_table input tensor for the kernel
std::vector<int> kernel_block_table;

// 1. Add Sink Blocks (Always fixed)
for (int i = 0; i < config.sink_blocks; i++) {
    kernel_block_table.push_back(physical_block_ids[i]);
}

// 2. Add Rolling Blocks (Re-ordered to look sequential to the kernel IF needed, 
//    but for Attention, order usually doesn't matter as long as PosID is correct. 
//    Simpler approach: Just pass the current physical list.)
for (int i = config.sink_blocks; i < physical_block_ids.size(); i++) {
    kernel_block_table.push_back(physical_block_ids[i]);
}

// CRITICAL: The kernel sees a list of size 'max_window_blocks', 
// regardless of how many tokens have truly been generated (10k, 100k...).`

### 3.3 Position Embedding (The "Correction")

**Module:** `GptContext` / Attention Plugin inputs

**Constraint:** Do NOT let the kernel derive `position_ids` from `sequence_length` or `block_table` size.

**Tensor Construction:**

- **Shape:** `[batch_size, 1]` (for decode step)
- **Value:** `[current_global_step]` (e.g., 1005, even if window size is 128)
- **Integration Point:** Pass this tensor explicitly to `context_attention` or `masked_mha` plugin inputs.

---

## 4. Hardware Specific Optimizations (Orin Nano)

### 4.1 Memory Pinning (CUDA)

To avoid Unified Memory Page Faults/Migration on the Tegra SoC:

C++

`void pin_memory_for_orin(void* ptr, size_t size) {
    int device = 0;
    // 1. Allocate as Managed
    cudaMallocManaged(&ptr, size);
    // 2. Advise: Preferred location is GPU
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
    // 3. Advise: Accessed by GPU (prevents thrashing)
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
}`

### 4.2 Alignment

- **Block Size:** Must be set to `16` or `32` in the Model Config and Builder.
- **Data Type:** KV Cache data type MUST be `int8` (if supported by model) or `fp16`.

---

## 5. Constraints & Edge Cases (Strict Enforcements)

8. **NO Beam Search:**
    - The `GptScheduler` must define `beam_width = 1`. If `beam_width > 1` is detected with StreamLLM enabled, throw an Exception.
    - *Reason:* Ring buffer state management for branching paths is $O(N)$ complexity and violates Zero-Copy principles.
9. **Step Boundary Updates Only:**
    - Updates to `rolling_write_pointer` and `physical_block_ids` MUST happen strictly **between** kernel launches.
    - Never modify the mapping while `context_attention` is in flight.
10. **Prefill Handling:**
    - If `prompt_length > max_window_blocks`: Implementation must truncate the prompt OR fall back to standard sliding window (non-StreamLLM) for the prefill phase, then switch to StreamLLM for generation. *Recommendation: Truncate prompt to window size.*

---

## 6. Action Items for Code Generation

**Instructions for the AI Developer:**

11. **Modify **`**cpp/src/runtime/blockManager.cpp**`:
    - Override the standard allocation logic to implement the Ring Buffer recycling described in Section 3.1.
    - Implement the `StreamLLMConfig` struct.
12. **Modify **`**cpp/src/runtime/gptSession.cpp**`:
    - Ensure `position_ids` are generated using the global step counter, independent of the KV cache layout.
    - Ensure these `position_ids` are passed to the enqueue function of the attention layer.
13. **Modify **`**examples/run.py**`** (or builder script)**:
    - Add command line arguments: `-enable_streamllm`, `-sink_token_num`, `-max_window_size`.
    - Ensure Builder configuration forces `use_custom_all_reduce=False` (simplification for single-GPU Orin) and proper RoPE scaling mode.

# 端侧多轮对话：

## buffer设计：

设计的目的：满足100k上下文的要求

设计核心：低比特量化 + 稀疏注意力

### kvcache的量化：(Int4 / Int8)

- **K Cache：** 对 RoPE 后的 Key 进行非对称 Int4/Int8 量化。
- **V Cache：** 直接进行 Int4/Int8 量化。

### 动态稀疏策略 (StreamingLLM / H2O)

长文档中只有少部分 Token 对生成下一个词真正重要（"Heavy Hitters"）。

- **设计：** 采用类似 **H2O (Heavy Hitter Oracle)** 或 **StreamingLLM** 的驱逐策略。
    1. **Attention Sinks：** 永久保留最开始的几个 Token（如 System Prompt），保证注意力计算的稳定性。
    2. **Sliding Window：** 保留最近的 N 个 Token（局部上下文）。
    3. **Eviction（驱逐）：** 中间部分的 Token，根据 Attention Score 动态丢弃分数低的 KV Block。
- **收益：** 可以用固定的显存预算（例如只存 4k Token 的大小）去推理无限长的上下文。

### C. 分层存储 (Multi-Tier Storage)

如果端侧设备拥有较快的 NVMe 存储（如 iPhone 或高端 PC）。

- **设计：三级缓存设计，热，温，冷，热、温储存在ram，热缓存正常储存，温缓存压缩，冷储存在nvme，** 将不常用的 KV Cache 页面 **Swap Out** 到磁盘，推理需要时预取（Prefetch）。
- 
- **注意：** 这需要极精细的 Pipeline 设计，掩盖 I/O 延迟，否则推理速度会骤降。



初始化预分配，环形的buffer，

端侧agent：

[[tensorrt llm]]

[[mnnllm]]