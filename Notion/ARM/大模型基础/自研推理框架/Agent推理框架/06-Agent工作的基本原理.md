# Agent 工作的基本原理

## 1. 一句话理解

Agent 的本质不是“模型一次性把答案说完”，而是：

> LLM 负责推理和决策，runtime 负责解析、调工具、更新上下文，再继续推理，直到任务完成。

所以它是一个循环系统，而不是一次性生成系统。

## 2. 从用户指令开始

用户输入一个任务，例如：

```text
帮我查今天上海天气，并提醒我带伞
```

系统首先会构造 prompt，一般包括：

- system prompt
- 历史对话
- 当前用户指令
- 可用工具列表

这些内容合在一起，形成当前的 context。

## 3. 第一次 LLM 推理

构造好 context 后，runtime 会调用底层推理框架：

- prefill
- decode

推理框架本身只负责：

- 计算 token
- 维护 KV cache
- 输出下一个 token

它不理解什么是 tool、plan、agent。

## 4. Agent Runtime 解析模型输出

模型生成的内容不会立刻直接返回给用户，而是先被 runtime 解析。

例如模型可能输出：

```text
Thought: 我需要先查天气
Action: weather_api
Action Input: {"city":"上海"}
```

runtime 看到这里会判断：

- 这不是最终回答
- 这是一个 tool call

于是会停止当前生成，转入工具执行阶段。

## 5. 工具执行阶段

runtime 调用对应工具，例如：

```text
weather_api("上海")
```

工具返回：

```text
今天上海有雨
```

然后 runtime 会把这个结果写回上下文，通常以 observation 的形式加入。

## 6. 再次推理

更新 context 后，runtime 会再次调用模型。

现在模型看到的是：

- 原始任务
- 自己前面的 action
- 工具返回的 observation

于是它可以继续推理，例如生成：

```text
Final Answer: 今天上海有雨，记得带伞
```

这时 runtime 识别到最终回答，任务结束。

## 7. 整体流程图

可以把完整链路理解成：

```text
用户输入
  ↓
Agent Runtime 构造 prompt
  ↓
Inference Engine 推理
  ↓
生成 token
  ↓
Runtime 解析输出
  ├─ 如果是最终答案 → 返回用户
  └─ 如果是工具调用 → 执行工具 → 更新 context → 再推理
```

## 8. 和普通聊天的本质区别

普通聊天更像：

```text
用户 → 模型 → 输出
```

Agent 更像：

```text
用户
 ↓
模型推理
 ↓
工具调用
 ↓
模型推理
 ↓
工具调用
 ↓
模型推理
 ↓
最终回答
```

所以 Agent 的关键不是“生成更多 token”，而是：

- 多次推理
- 多次上下文编辑
- 多次状态切换

## 9. 为什么这会影响端侧推理框架

因为 Agent 会带来：

- 频繁 prefill
- 短 decode
- 上下文不断变化
- KV cache 复用更复杂

所以端侧框架更需要：

- 快速 prefill
- context editing
- prefix reuse
- 可回退的 KV cache

## 10. 一句话总结

Agent 的基本原理就是：

> 用户给任务，LLM 负责“想下一步”，runtime 负责“把这一步真正执行掉”，再把结果喂回模型，直到任务完成。
