# KV Cache 计算

## 1. 你现有笔记里已经明确的公式线索

从 `[[ARM/大模型基础/自研推理框架/KvCache/KvCache]]` 提炼：

- 环形缓存容量公式：
  - `Size = (N_sink + N_rolling) × Layers × Heads × Dim × 2B`
- 这里的 `2B` 对应 K 和 V 两份存储

## 2. 面试常用通用公式

对单 batch、单 token 的 KV cache：

`KV_per_token = Layers × KV_Heads × Head_Dim × 2 × bytes_per_element`

对总上下文长度 `T`：

`KV_total = Batch × T × Layers × KV_Heads × Head_Dim × 2 × bytes_per_element`

## 3. 回答时要主动说明的变量

- 是否是 MHA 还是 GQA/MQA
- `KV_Heads` 是否等于 `Attention Heads`
- 精度是 FP16、BF16、INT8 还是更低
- 算的是单请求还是多请求
- 算的是理论容量还是加上 block/page 对齐浪费后的真实占用

## 4. 和项目结合的点

- Decode 阶段显存增长主要就是 KV cache
- 长上下文时，KV cache 往往比激活更关键
- block/page 管理会引入额外碎片和对齐成本
