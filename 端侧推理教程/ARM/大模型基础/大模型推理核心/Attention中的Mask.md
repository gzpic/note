# Attention 中的 Mask

## 1. Mask 是干什么的

在大模型里，mask 的本质作用就是：

> 规定 attention 里“哪些位置能看，哪些位置不能看”。

也可以把它理解成 attention 的可见性规则表。

它不改变模型结构，但会直接决定：

- 哪些 token 参与注意力计算
- 哪些位置被强制屏蔽
- 最终 softmax 概率怎么分配

## 2. 为什么需要 mask

主要有 3 类原因。

### 2.1 防止偷看未来

在自回归生成里，当前位置只能看自己和过去，不能看未来。

这就是 `causal mask` 的作用。

例如长度为 4 的可见性可以写成：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

也就是：

- 第 1 个 token 只能看第 1 个
- 第 2 个 token 能看前 2 个
- 第 3 个 token 能看前 3 个

### 2.2 忽略 padding

一个 batch 里的样本长度往往不同，所以会补 PAD。

PAD 不是真实内容，不能让模型把注意力分给它，所以需要 `padding mask`。

### 2.3 控制特殊可见范围

有些模型或算子会做更特殊的可见性限制，例如：

- 滑动窗口 attention
- block sparse attention
- prefix 可见
- 多模态里文本只能看部分图像 token

这些本质上也都是 mask。

## 3. Mask 在 attention 的哪个阶段起作用

mask 主要是在 attention score 计算之后、softmax 之前起作用。

流程大致是：

```text
Q, K, V
  ↓
scores = QK^T / sqrt(d)
  ↓
scores = scores + mask
  ↓
prob = softmax(scores)
  ↓
output = prob · V
```

也就是说：

- mask 不在 embedding 阶段
- 不在 MLP 阶段
- 不在最后采样阶段
- 它主要作用在 `score -> softmax` 这段

## 4. 为什么必须在 softmax 之前处理

因为 mask 的目标，是让某些位置的概率直接变成 0。

常见做法是：

- 允许的位置加 `0`
- 不允许的位置加 `-inf` 或很小的负数，比如 `-1e9`

这样 softmax 后：

- 合法位置还能分到概率
- 非法位置概率约等于 0

如果等 softmax 之后再处理，概率已经分完了，就不自然了。

## 5. 最常见的几类 mask

### 5.1 Causal Mask

最常见于自回归 LLM。

作用是：

- 当前 token 不能看未来 token

### 5.2 Padding Mask

作用是：

- 屏蔽补齐出来的 PAD

### 5.3 Combined Mask

实际系统中经常会把：

- causal mask
- padding mask

合并成一个总的 attention mask。

## 6. 训练和推理里的 mask 有什么区别

### 6.1 训练时

训练通常一次输入整段序列，所以往往会构造完整的 mask 矩阵。

例如：

- `[B, 1, S, S]`
- 或可广播到 score 张量的其他形状

### 6.2 推理时

推理尤其是 decode 阶段，通常一次只生成 1 个 token。

这时：

- query 往往只有当前 1 个位置
- key/value 是全部历史 cache

所以很多时候“不能看未来”已经天然满足了一部分。

也正因为如此，decode 阶段经常不需要真的构造完整 `[S, S]` mask。

## 7. 一般怎么实现

常见有两种实现方式。

### 7.1 显式构造 mask 矩阵

最直观的实现是：

```text
scores = QK^T / sqrt(d)
scores += mask
probs = softmax(scores)
```

这里的 `mask` 往往是：

- 可见位置为 `0`
- 不可见位置为 `-inf`

优点：

- 直观
- 容易调试
- 训练阶段常见

缺点：

- 需要额外存一个大矩阵
- 长序列时占内存和带宽

### 7.2 在 kernel 内部隐式处理

高性能推理里更常见的是：

- 不显式构造完整 mask tensor
- 只把规则传给 kernel
- kernel 在计算 score 时自己判断当前位置是否合法

例如伪代码：

```text
if (k > q) score = -inf;          // causal
if (k >= valid_len) score = -inf; // padding
```

优点：

- 更省显存
- 更省带宽
- 更适合 fused attention / FlashAttention / FMHA

缺点：

- 实现更复杂

## 8. 工程里常见的几种输入方式

### 8.1 直接传 attention_mask tensor

上层框架里很常见，例如：

- PyTorch
- Hugging Face

这类更偏框架友好型实现。

### 8.2 只传长度参数

很多推理框架不会传完整 mask，而是只传：

- 有效序列长度
- 当前 query 长度
- 历史 KV 长度
- 是否 causal
- 窗口大小

然后在 kernel 内部判断哪些位置有效。

这在高性能推理框架里很常见。

### 8.3 融进 FlashAttention / FMHA kernel

再进一步的实现里，mask 逻辑会直接融合进 attention kernel。

也就是说：

- 逻辑上仍然是在 softmax 前屏蔽非法位置
- 工程上已经不再显式生成一个完整 mask 矩阵

## 9. Prefill 和 Decode 里的 mask

### 9.1 Prefill

prefill 阶段一次处理整段 prompt，通常仍需要完整地表达：

- causal 关系
- padding 关系

所以更容易看到显式 mask 或完整的规则控制。

### 9.2 Decode

decode 一次通常只有一个 query token。

这时更常见的是：

- 只让它访问历史 KV
- 未来位置根本不存在或根本不读取

所以 decode 的 causal mask 往往更像一种隐式约束。

## 10. 为什么 mask 在大模型推理里很重要

它直接影响：

- 结果是否正确
- PAD 会不会污染 hidden states
- prefill / decode 能不能正确衔接
- KV cache 的读取边界是否正确

如果 causal mask 错了，模型就可能“看到未来”；如果 padding mask 错了，模型就会把注意力浪费在无效 token 上。

## 11. 一句话总结

mask 的本质就是：

> 在 attention 里限制“谁能看谁”。

它通常在 `QK^T` 之后、softmax 之前起作用；训练里更常见显式 mask，端侧和高性能推理里更常见“传规则 + kernel 内部隐式处理”。
