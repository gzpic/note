# Agent 输出解析、修复与格式约束

## 1. 推理框架真正输出的是什么

推理框架本身并不知道什么是：

- Thought
- Action
- JSON
- tool call

它真正输出的只是：

- `token id`
- 或 logits

例如：

```text
29871, 13, 920, 345, ...
```

再经过 tokenizer decode 后，才会变成文本。

## 2. 为什么 Agent 要求“固定格式输出”

Agent runtime 需要解析模型输出，才能知道：

- 是继续生成
- 还是调用工具
- 还是已经得到最终答案

所以 runtime 通常会通过 prompt 约束模型按结构化格式输出，例如：

### 2.1 ReAct 风格

```text
Thought: ...
Action: ...
Action Input: ...
```

### 2.2 JSON 风格

```json
{
  "tool": "weather_api",
  "arguments": {
    "city": "上海"
  }
}
```

注意：

> 这是 LLM 生成文本的模板，不是推理框架底层 I/O 的模板。

## 3. 谁真正关心这个模板

分三层看：

- 推理框架：只关心 token
- LLM：按 prompt 生成文本
- Agent runtime：解析这些文本结构

所以更准确的说法是：

> 不是推理框架的输入输出固定模板，而是 Agent 希望模型生成符合模板的文本。

## 4. 如果输出不符合预期怎么办

这在 Agent 系统里非常常见，因为 LLM 本质上是概率生成模型。

常见问题包括：

- 普通文本，不是结构化输出
- JSON 不完整
- 工具名错误
- 参数字段错误
- schema 不符合要求

## 5. Runtime 怎么知道哪里错了

通常是因为 runtime 内置了一套：

- parser
- validator
- tool registry

典型流程是：

```text
LLM output
  ↓
Parser
  ↓
Schema Validator
  ↓
Tool Validator
  ↓
决定执行 / 修复 / 重试
```

### 5.1 Parser

负责解析：

- JSON
- ReAct 格式
- function calling 文本

### 5.2 Validator

负责检查：

- JSON 合法性
- schema 是否正确
- 参数字段是否齐全
- 类型是否匹配

### 5.3 Tool Registry

负责检查：

- tool 是否存在
- tool 是否允许被调用

## 6. Runtime 常见怎么处理错误

### 6.1 Retry with feedback

最常见。

runtime 会根据错误类型给模型一个针对性的修复提示，例如：

```text
Your previous output was not valid JSON.
Please output valid JSON only.
```

这不是简单重复原 prompt，而是：

> parse -> diagnose -> repair prompt -> re-run

### 6.2 自动修复

如果只是轻微格式问题，例如少一个大括号，runtime 可能会直接修复，而不重新调用模型。

### 6.3 Fallback

如果重试多次仍失败，runtime 会：

- 退化成普通回答
- 返回错误
- 请求用户澄清

## 7. 为什么 constrained decoding 很重要

因为端侧小模型和量化模型更容易生成格式错误。

为了减少解析失败，很多系统会做：

- grammar constrained decoding
- JSON schema constrained decoding

这样可以在生成阶段就限制：

- 只允许合法 token
- 只允许合法 JSON 结构

优点是：

- parse error 更少
- runtime 更稳定
- 端侧重试次数更少

## 8. 一句话总结

推理框架只负责生成 token；
Agent runtime 负责把这些 token decode 成结构化文本，并通过解析、校验、修复、重试来保证它能被真正执行。
