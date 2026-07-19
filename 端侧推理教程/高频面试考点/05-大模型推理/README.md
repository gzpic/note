# 大模型推理

## 高频题

### 1. Transformer 一次前向的主流程

1. token -> embedding
2. 每层执行：
   - norm
   - QKV projection
   - attention
   - output projection
   - residual
   - FFN
   - residual
3. final norm
4. lm head
5. logits -> sampling

### 2. Prefill 和 Decode 区别

- `Prefill`
  - 一次处理整段 prompt
  - 更像大矩阵批处理
  - 更偏计算密集
- `Decode`
  - 一次推进少量 token
  - 需要反复读历史 KV Cache
  - 更偏访存和调度

### 3. 为什么要有 KV Cache

- 因为历史 `K/V` 会在后续 token 生成中重复使用
- 缓存它们可以避免每步重复算历史部分
- 没有 `Q Cache` 的原因：
  - `Q` 只对当前步有意义

### 4. GQA / MQA 为什么重要

- 核心目的都是减少 `KV Cache` 体积和带宽压力
- 面试稳妥说法：
  - 这是为了让长上下文和 decode 更可部署

### 5. RoPE / RMSNorm 高频答法

- `RoPE`
  - 一种相对位置编码方式
  - 更适合长上下文扩展
- `RMSNorm`
  - 不做减均值，只做基于 RMS 的缩放
  - 比 LayerNorm 更轻

### 6. FlashAttention 的核心价值

- 核心不是公式变化
- 核心是重排计算顺序，减少中间 score 矩阵写回显存
- 依赖：
  - tile 化
  - online softmax

### 7. Prefix Cache / Continuous Batching / Paged KV

- `Prefix Cache`
  - 复用共享前缀的 KV
- `Continuous Batching`
  - 不等整批结束，每个 step 动态混入请求
- `Paged KV`
  - 把 KV Cache 分页或分块管理，降低碎片，便于回收

### 8. PTQ、QAT、AWQ、GPTQ

- `PTQ`
  - 训练后量化
- `QAT`
  - 量化感知训练
- `AWQ/GPTQ`
  - 更偏大模型部署里的代表性量化路线
- 面试里最稳的回答：
  - 量化不是只降 bit，核心是控制误差、适配 kernel、兼顾部署收益

### 9. Decoder-only、Encoder-only、Encoder-Decoder

- `Decoder-only`
  - 适合自回归生成
- `Encoder-only`
  - 适合理解类任务
- `Encoder-Decoder`
  - 适合 seq2seq 任务，如翻译和摘要

## 回答骨架

### 大模型推理题的通用答法

1. 先说这个机制解决什么问题
2. 再说它换来了什么收益
3. 再说代价或限制
4. 最后补工程落地场景

例如“为什么 decode 更难优化”：

- 因为它每步计算量小，但要频繁访问历史 KV Cache。
- 这导致瓶颈更容易落在带宽、缓存命中和小 kernel 调度上。
- 所以 decode 往往不是单纯算力问题，而是系统和内存路径问题。

## 高频口头结论

### 为什么 Prefill 更像 compute-heavy

- prompt 整段处理时，矩阵更大，更容易把 Tensor Core 和大 GEMM 路径吃满

### 为什么 Decode 更像 memory-heavy

- 每步只生成少量 token，但会持续读历史 KV
- 数据复用方式和 launch 粒度都更不友好

### 为什么 Prefix Cache 值钱

- 直接省重复 prefill
- 在共享系统 prompt、共享前缀问答场景里收益很直接

## 易错点

- 不要把 KV Cache 说成缓存所有中间激活
- 不要把 FlashAttention 说成“换了注意力公式”
- 不要把量化理解成“bit 越低一定越快”
- 不要把 continuous batching 说成普通 static batching

## 原笔记入口

- `[[Notion/面试整理/02-八股文/05-常见大模型]]`
- `[[Notion/面试整理/02-八股文/06-推理步骤]]`
- `[[Notion/面试整理/03-项目问题准备/02-推理框架问题]]`
- `[[Notion/面试整理/03-项目问题准备/03-推理算子问题]]`
- `[[Notion/ARM/大模型基础/大模型基础]]`
- `[[Notion/ARM/大模型基础/大模型量化]]`
