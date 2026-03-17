# Agent 场景下推理框架层的变化

## 1. 只看推理框架层，变化大不大

如果只看 inference engine 这一层，Agent 场景并不是完全颠覆式变化，但有几个关键点会变得更重要：

- KV cache 管理
- 多次 prefill / decode 的调度
- context editing
- 低延迟优化
- structured output / constrained decoding

## 2. KV Cache 管理策略会不同

普通推理通常是：

- 一次 prefill
- 多次 decode
- 一次会话结束

Agent 场景更像：

- prefill
- decode
- 插入 tool result
- 再 prefill
- 再 decode

所以 KV cache 生命周期更长，也更容易变化。

这就要求推理框架更重视：

- paged KV cache
- ring KV cache
- eviction
- prefix reuse

## 3. Prefill / Decode 调度模式会变化

普通聊天更像：

- 1 次 prefill
- 长一段 decode

Agent 则常常变成：

- 多次短 decode
- 多次重新 prefill

所以框架层的重点会从：

- 极限 decode 吞吐

部分转向：

- 反复 prefill 的代价控制
- prefix reuse
- context append / truncate

## 4. Context Editing 能力更重要

Agent 场景下 context 不是只会单调增长，还会发生：

- 插入 tool result
- 删除旧历史
- 压缩旧对话
- 替换 summary

因此推理框架层更需要支持：

- KV cache truncate
- append
- 对 prefix 的复用

## 5. 低延迟比高吞吐更重要

普通推理框架常常优化：

- continuous batching
- batch decode
- 最大 token/s

而 Agent 场景里，更常见的是：

- 很多短推理
- 很多中断点
- 很多 tool call 边界

所以更重要的是：

- 单次任务响应快
- 能及时停
- 能快速恢复继续推理

## 6. Structured Output 更重要

普通聊天偏自然语言。

Agent 则更常见：

- function call
- JSON
- tool call schema

所以推理框架层更需要支持：

- constrained decoding
- grammar-based sampling
- stop token / early stop

## 7. 投机解码是不是硬要求

不是。

投机解码主要优化的是 decode，但很多端侧 Agent 的主要瓶颈其实是：

- 多次 prefill
- context 膨胀

所以比起 speculative decoding，端侧 Agent 更值得优先优化的是：

- prefix cache
- context reuse
- KV cache 管理
- early stop

## 8. 一句话总结

如果只看推理框架层，Agent 场景最本质的变化不是“模型换了”，而是：

> context 生命周期变复杂了，prefill / KV / 中断恢复变得更重要了。
