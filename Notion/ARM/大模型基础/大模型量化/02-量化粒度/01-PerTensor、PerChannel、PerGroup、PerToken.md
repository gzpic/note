# PerTensor、PerChannel、PerGroup、PerToken

## 1. 量化粒度本质上在决定什么

量化公式里最关键的是 `scale`：

```text
x_q = clamp(round(x / s) + z)
```

不同粒度的区别，本质上就是：

> 哪些元素共享同一个 `scale`。

## 2. Per-Tensor

整个张量共用一个 `scale`。

特点：

- 实现最简单
- 性能最好
- 精度最差
- 对 outlier 很敏感

适合：

- baseline
- 一些对精度不特别敏感的激活或缓存

## 3. Per-Channel

每个通道使用一个独立 `scale`。

特点：

- 精度显著好于 per-tensor
- 工程复杂度仍可控
- 是工业界最常用的方案

常见用法：

- 权重量化默认首选
- 高质量 INT8 推理常用

一句话：

> per-channel 是精度和工程成本之间的甜点区。

## 4. Per-Group

把通道按组划分，每组共享一个 `scale`。

特点：

- scale 数量比 per-channel 少
- 精度接近 per-channel
- 更利于压缩 scale 和提升访存友好度

常见于：

- GPTQ
- AWQ

## 5. Per-Token

每个 token 独立使用一个 `scale`。

特点：

- 精度很高
- 更适合处理激活动态变化
- 但推理实现代价很大

主要问题：

- scale 随 token 变化
- decode 更难静态化
- kernel 更复杂

所以它更常出现在研究或分析里，而不是高性能部署主线里。

## 6. 四种粒度横向对比

| 粒度 | scale 数量 | 精度 | 性能 | 实现复杂度 | 推理友好度 |
| --- | --- | --- | --- | --- | --- |
| Per-Tensor | 1 | 低 | 高 | 低 | 高 |
| Per-Channel | C | 高 | 较高 | 中 | 高 |
| Per-Group | C/G | 较高 | 较高 | 中 | 中 |
| Per-Token | B×S | 很高 | 低 | 高 | 低 |

## 7. 工程建议

面向 LLM 推理时，常见组合是：

- Weight：per-channel 或 per-group
- Activation：per-tensor
- 需要保精度时用 SmoothQuant 或更强校准策略

## 8. 面试速记版

- per-tensor 最粗，性能最好但精度最低
- per-channel 是工业默认方案
- per-group 是 GPTQ/AWQ 常见折中
- per-token 精度高，但不适合高性能 decode 推理
