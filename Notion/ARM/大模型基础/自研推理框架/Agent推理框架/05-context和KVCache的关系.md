# context 和 KV Cache 的关系

## 1. 先说结论

在推理时，通常既需要 context，也需要 KV cache。

但它们不是两份完全重复的数据。

可以直接记成：

- context：逻辑上的输入内容
- KV cache：这些输入经过模型计算后的加速缓存

## 2. context 负责什么

context 负责表示：

- 当前模型应该看到什么
- 哪些历史保留
- 哪些内容属于 system / user / assistant
- 哪些内容要截断、压缩、替换

它更偏语义和内容管理层。

## 3. KV Cache 负责什么

KV cache 负责表示：

- 这些 context token 在每一层里对应的 `K / V`
- 后续 decode 时旧 token 不用重新算

它更偏计算加速层。

## 4. 为什么不能只有 KV Cache

因为 KV cache 只告诉你：

- 哪些 token 已经算过了

但它不适合直接回答：

- 原始文本是什么
- 哪几轮该删
- 哪段是 tool result
- 哪部分要做 summary

所以 context 仍然必须存在。

## 5. 为什么不能只有 context

如果只有 context，没有 KV cache，那么每次 decode 新 token 时都要：

- 把全部历史重新过一遍模型

这样会非常慢。

所以现代 LLM 推理几乎都会保留 KV cache。

## 6. 它们的关系是什么

可以理解成：

```text
context tokens
  ↓ prefill
生成每层 K / V
  ↓
KV cache
```

所以：

> KV cache 是 context 经过模型计算后的缓存结果。

## 7. 当 context 变化时会怎样

一旦 context 发生变化，例如：

- 截断旧对话
- 插入 tool result
- summary 替换旧历史

那么对应的 KV cache 也要跟着变化：

- 一起删除
- 一起失效
- 或尽量局部复用

因为 KV cache 必须和当前 context 严格对齐。

## 8. 为什么 Agent 场景更难

普通聊天中，context 往往只是不断向后追加。

Agent 场景里，context 更可能：

- 追加
- 截断
- 插入
- 重写

所以真正困难的地方是：

> context 变化时，KV cache 怎么尽量复用，而不是每次都从头重算。

这也是很多端侧 Agent 推理框架最难做的点之一。

## 9. 一句话总结

context 是“模型应该看到什么”，KV cache 是“这些内容我已经算好了什么”。

前者负责内容管理，后者负责推理加速。
