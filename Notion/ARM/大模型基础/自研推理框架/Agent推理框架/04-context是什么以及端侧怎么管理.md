# context 是什么以及端侧怎么管理

## 1. context 是什么

在 LLM / Agent 推理里，context 可以理解为：

> 这一轮推理时，模型能看到的全部输入内容。

在推理引擎内部，它最终会表现成：

- token ids
- 对应的 message 结构 / metadata

## 2. 普通聊天里的 context

普通聊天里，context 通常包括：

- system prompt
- 历史对话
- 当前用户输入

例如：

```text
System: ...
User: ...
Assistant: ...
User: ...
Assistant:
```

## 3. Agent 场景里的 context 更复杂

Agent 里，context 常常还会包含：

- tool description
- tool result / observation
- memory
- reasoning 痕迹

所以它会比普通聊天更快膨胀。

## 4. 为什么端侧特别怕 context 变长

因为 context 长度直接决定 prefill 开销。

context 越长：

- prefill 越慢
- KV cache 越大
- 显存占用越高

所以端侧系统通常会非常严格地管理 context。

## 5. 端侧最常见的管理策略

### 5.1 Sliding Window

最常见的是：

- 只保留最近几轮对话

优点：

- 实现简单
- 时延稳定
- context 长度可控

### 5.2 Token 截断

不是按轮数，而是按总 token 数限制。

超过最大长度就从最旧部分开始截断。

### 5.3 Summary

稍复杂一点的系统会把旧对话压缩成 summary，再保留最近几轮。

这种方式更智能，但实现和延迟成本都更高。

## 6. Agent 场景下的额外处理

Agent 会生成很多：

- Thought
- Action
- Observation

端侧系统通常不会把这些都完整永久保留，而是会做取舍。

一个常见原则是：

- 尽量删掉冗长 reasoning
- 只保留必要 action 和 observation

## 7. 一句话总结

context 是模型当前能看到的全部输入，而端侧系统的关键不是“把所有历史都塞进去”，而是：

> 尽量用最短的 context 保留最有价值的信息。
