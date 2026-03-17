# Agent 记忆机制与 Retrieval Memory

## 1. 先说结论

Agent 需要 memory，是因为 LLM 本身没有会话外的持久状态。

但 memory 不等于“把所有历史都塞进 prompt”。

更准确的理解是：

- 短期记忆放在当前 context 里
- 长期记忆放在外部存储里
- 需要时再检索少量相关内容放回 prompt

所以现代 Agent 的核心不是“记得更多”，而是：

> 以尽量小的 prompt，拿到尽量有用的历史信息。

## 2. 为什么 Agent 需要记忆

LLM 本质上是无状态函数：

```text
output = model(prompt)
```

如果不额外保存信息，模型不会自动记住：

- 之前和用户聊过什么
- 刚才调用过哪些工具
- 用户有哪些长期偏好
- 当前任务已经做到哪一步

所以 Agent 必须自己维护 memory。

## 3. 短期记忆是什么

短期记忆通常就是“当前任务还在用的信息”，它一般直接进入当前 context。

典型包括：

- 最近几轮对话
- 当前任务的 plan
- 最近一次或几次 tool result
- 当前 observation
- 本轮推理需要继续依赖的中间状态

它的作用是：

- 让任务不断线
- 让模型知道刚刚发生了什么

可以把它理解成 working memory。

## 4. 长期记忆是什么

长期记忆是“这次不一定立刻要用，但以后可能有用”的信息。

例如：

- 用户住在上海
- 用户偏好早班航班
- 用户常订某家酒店
- 某类任务的历史摘要

长期记忆通常不会直接常驻在 prompt 里，而是单独存到外部存储。

它更偏向：

- 个性化
- 跨任务保留
- 长期知识积累

## 5. 为什么不能把所有记忆都放进 prompt

因为这样做代价太高。

一旦把全部历史都注入 prompt，就会出现：

- context 急速膨胀
- prefill 越来越慢
- KV cache 越来越大
- 显存和延迟越来越难控制

所以真正可用的 Agent 系统，几乎都不会做“全历史注入”。

## 6. Retrieval Memory 是什么

Retrieval Memory 的核心思想是：

> 记忆先存在外部数据库里，运行时只取最相关的少量几条放进当前 prompt。

也就是说：

- memory 可以很多
- prompt 仍然可以很短

这就是它和“直接堆历史”最大的区别。

## 7. 一条 memory 里通常存什么

一条 memory 通常不只是文本，还会带一些结构化信息。

常见字段有：

- `text`
- `embedding`
- `metadata`

例如：

```text
text: 用户喜欢早班航班
embedding: [...]
metadata: {user_id, time, source, tag}
```

这样做的好处是：

- 可以做语义检索
- 可以按用户 / 时间 / 类型过滤

## 8. Retrieval Memory 的工作流程

运行时常见流程是：

```text
当前用户问题
  ↓
计算 query embedding
  ↓
在 memory store 中做相似度检索
  ↓
取 top-k memories
  ↓
把这几条相关记忆插入 prompt
```

例如用户说：

```text
帮我订机票
```

runtime 可能检索到：

- 用户喜欢早班航班
- 用户经常从上海出发

然后把这两条加入 prompt，而不是把全部历史都加进去。

## 9. 现实系统里常见的几种 memory 策略

### 9.1 Sliding Window

只保留最近几轮对话。

优点：

- 最简单
- 延迟稳定

缺点：

- 容易丢长期信息

### 9.2 Summary Memory

把旧历史压缩成摘要。

优点：

- 比直接保留全文省 token
- 能保留一部分长期语义

缺点：

- 摘要本身也可能失真
- 需要额外生成成本

### 9.3 Retrieval Memory

把历史存到外部数据库，按需检索。

优点：

- 历史规模可以很大
- prompt 仍可控

缺点：

- 系统复杂度更高
- 检索质量会影响效果

### 9.4 Tool-based Memory

把 memory 读写也做成工具，例如：

- `search_memory()`
- `store_memory()`

这样 memory 本身就成为 Agent 工具链的一部分。

## 10. 每次对话都要带全部历史吗

不需要。

现实系统通常只会带：

- recent conversation
- summary
- retrieved memory
- 当前输入

而不会带：

- 全部聊天历史
- 全部 tool result
- 全部 reasoning 痕迹

否则 prompt 很快就不可控。

## 11. 端侧系统一般怎么取舍

端侧设备更敏感，因为：

- prefill 成本高
- 显存小
- context window 更宝贵

所以端侧一般更偏向简单策略：

- 最近几轮对话
- token 长度截断
- 少量 summary

真正复杂的 retrieval memory 在端侧不是不能做，但通常要更克制。

一个更现实的端侧路线往往是：

1. 先用 sliding window
2. 再加 token 截断
3. 必要时加轻量 summary
4. 最后才考虑 retrieval memory

## 12. 端侧为什么尤其需要 Retrieval 思维

虽然端侧不一定马上上完整向量数据库，但“检索式思维”仍然很重要。

因为端侧最怕的是：

- context 线性增长
- prefill 反复变重

而 Retrieval 的本质正是在解决：

> memory 很多，但 prompt 不能跟着一起无限增长。

## 13. 一句话总结

Agent 的 memory 不是把所有历史都喂给模型，而是把信息分层管理：

- 当前任务相关的，放进 context
- 长期保留但不总是需要的，放到外部 memory
- 运行时只检索最相关的一小部分再注入 prompt
