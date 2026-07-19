# Jetson Orin Nano 量化路线

## 1. 先看硬约束

Jetson Orin Nano 的量化路线不能只看论文效果，要先看硬件现实：

- GPU 是 Ampere 架构
- `FP16` 和 `INT8` 支持成熟
- 显存和带宽都紧
- 功耗受限

一句话：

> 在 Orin Nano 上，`INT8` 是甜点区，`INT4` 是挑战区。

## 2. 推荐的主线量化路线

更合理的工程路线通常是：

1. `FP16 baseline`
2. `INT8 + SmoothQuant`
3. 视需要尝试部分 activation quant
4. `INT4 weight-only + AWQ`
5. PTQ 不够时再考虑局部 QAT

不要一上来就做最激进方案。

## 3. 阶段一：FP16 baseline

这一步的意义不是最终部署，而是：

- 验证数值正确性
- 查清性能瓶颈
- 先做好 kernel fusion 和内存复用

如果 FP16 路径本身没有整理好，后面量化收益通常也会很有限。

## 4. 阶段二：INT8 + SmoothQuant

这是 Orin Nano 上很稳妥的一条主线。

理由：

- INT8 支持成熟
- SmoothQuant 精度和复杂度平衡好
- 相比直接压低 bit，更容易落地

工程上常见结构是：

- Activation：FP16 或更温和地处理
- Weight：INT8 per-channel
- 关键 Linear 做高效 INT8 GEMM

## 5. 阶段三：Activation INT8

这一步不是主线，而是可选优化。

原因：

- activation 动态范围更麻烦
- requant / dequant 开销可能抵掉收益
- Attention 和 Norm 附近更难做干净

所以在端侧，activation INT8 往往没有想象中那么划算。

## 6. 阶段四：INT4 weight-only + AWQ

这一阶段更偏显存优化。

优点：

- 显存下降明显
- 端侧 weight-only 的工程可行性较高
- AWQ 对端侧适配更友好

注意点：

- `INT4` 不一定比 `FP16` 或 `INT8` 更快
- 但通常会更省显存
- unpack 和 dequant 的设计会决定最终速度

## 7. 阶段五：QAT 只作为兜底

如果：

- SmoothQuant 精度不够
- AWQ 精度不够

再考虑：

- LoRA-QAT
- block-level QAT

不要把 QAT 作为第一步。

## 8. 一句话拍板建议

如果只选一条最现实主线：

> `FP16 → INT8(SmoothQuant) → 精度不够再看 AWQ / 局部 QAT`

## 9. 面试速记版

- Orin Nano 上最重要的是访存和 kernel 组织，不是盲目追更低 bit
- `INT8 + SmoothQuant` 是主线
- `INT4 + AWQ` 更像显存优化路线
- `QAT` 是精度兜底，不是起手式
