---

---
> TensorRT-LLM 的 KV/LV cache 优化重点不是“最大化复用”，而是：
👉 极致确定性 + Kernel 友好 + 显存访问效率最大化

所以它是一个**“为 GPU kernel 服务的 cache 设计”**，而不是“为多会话/Agent 逻辑服务”的 cache 系统。

---

## 一、TensorRT-LLM 对 KV / LV cache 的三个核心关注点

### 1️⃣ **面向 Kernel 的内存布局（这是最核心的）**

TRT-LLM 的 KV cache 设计第一原则是：

> 让 attention kernel 读 cache 的时候，做到：

- 连续
- 对齐
- 可预测
- 少间接寻址

### 具体体现：

- KV cache 采用 **block / page 化的线性显存布局**
- 每个 block 对应：
```plain text
[num_layers][2(K,V)][block_tokens][num_heads][head_dim]


```
- **block size 是固定的**（比如 16 / 32 / 64 tokens）
- kernel 里：
    - blockIdx → block
    - threadIdx → head / dim
    - 不需要 radix 查找、hash、指针跳转

📌 **这是 TRT-LLM 和 vLLM 的本质分叉点**

---

### 2️⃣ **Paged KV Cache（但目标不是“复用”，而是“稳定分配”）**

TRT-LLM 也有 paging，但目的不同：

| 框架 | paging 的核心目的 |
| --- | --- |
| vLLM | 前缀复用 + 高吞吐多请求 |
| TRT-LLM | 避免碎片 + kernel 可预测性 |

### TRT-LLM 的 paging 特点：

- cache manager 在 **生成前** 就决定：
    - 一共用多少 block
    - block index 映射是静态的
- decode 阶段：
    - **不会频繁 realloc / move block**
    - 不允许“token 级别乱跳”

👉 这是为了保证：

- CUDA Graph 可复用
- kernel launch shape 不变
- latency 抖动最小

---

### 3️⃣ **强烈偏向“静态 / 半静态 cache 生命周期”**

你之前也问过这个点，我们当时的结论是：

> TRT-LLM 更擅长：

- 单会话
- 短多轮
- Prompt 已知 / 长度上界明确
- batch 内形态相似

### 所以它的 cache 假设是：

- sequence 是**单调增长的**
- 不支持：
    - 回滚
    - 分叉
    - token 删除
- cache 生命周期 ≈ request 生命周期

📌 这和 Agent / tool use 场景天然冲突