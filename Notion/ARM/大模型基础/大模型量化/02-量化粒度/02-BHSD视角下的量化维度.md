# BHSD视角下的量化维度

## 1. 先统一张量视角

大模型推理里常见张量布局可以写成：

```text
[B, H, S, D]
```

其中：

- `B`：batch
- `H`：head
- `S`：sequence / token
- `D`：head_dim 或 feature 维度

真正要问的是：

> scale 在这些维度上是怎么共享的。

## 2. Per-Tensor 在 BHSD 下是什么意思

整个 `[B, H, S, D]` 共用一个 `scale`：

```text
scale shape = [1]
```

也就是：

- 不区分 batch
- 不区分 head
- 不区分 token
- 不区分 feature

## 3. Per-Channel 在 BHSD 下对应哪个维度

这个问题不能死记成某一个维度，关键要看当前张量在算子语义里谁是“channel”。

### 3.1 常见情况：对应 D

在很多 Linear / FFN 场景里，`channel` 更接近 feature 维，也就是 `D`：

```text
scale shape = [D]
x[b, h, s, d] 使用 scale[d]
```

### 3.2 另一种情况：对应 H

在某些 attention / KV cache 相关场景里，也可能按 `head` 做量化：

```text
scale shape = [H]
x[b, h, s, d] 使用 scale[h]
```

所以更准确的说法不是“per-channel 一定等于 D”，而是：

> 在 LLM 中，per-channel 通常沿着语义上的 channel 维做，最常见是 D，也可能是 H。

## 4. Per-Group 在 BHSD 下怎么理解

per-group 本质上是在 `H` 或 `D` 上继续分组：

- 按 `D` 分组：`scale shape = [D / G]`
- 按 `H` 分组：`scale shape = [H / G]`

例如：

```text
x[b, h, s, d] 使用 scale[floor(d / G)]
```

## 5. Per-Token 在 BHSD 下怎么理解

per-token 的含义是：

- 每个 token 一个独立 `scale`
- 通常对同一个 `(b, s)` 共享

也就是：

```text
scale shape = [B, S]
x[b, h, s, d] 使用 scale[b, s]
```

这也是为什么它对 decode 更不友好，因为 `scale` 会随着 token 动态变化。

## 6. 常见误区

### 6.1 误区一：per-channel 一定等于 D

不一定。

更准确地说：

- Linear / FFN 中常常是 D
- KV cache 或某些 attention 场景里也可能按 H

### 6.2 误区二：per-token 就是对 D 量化

不是。

per-token 是沿着 `S` 在变化，通常同一个 token 内的 `H` 和 `D` 共享一个 scale。

## 7. 工程判断句

最好记成这句：

> 谁出现在 scale 的下标里，谁就是“被单独区分”的维度；没出现在 scale 下标里的维度，就共享同一个 scale。

## 8. 面试速记版

- BHSD 下看量化粒度，本质上就是看 scale shape
- per-tensor：`[1]`
- per-channel：通常是 `[D]`，有时也可能是 `[H]`
- per-group：`[D/G]` 或 `[H/G]`
- per-token：`[B, S]`
