# Agent 推理框架与普通推理框架的差异

## 1. 先看一句话结论

普通 LLM 推理框架的核心任务是：

- token -> token

而 Agent 推理框架面对的是：

- 推理 -> 决策 -> 调工具 -> 再推理

所以它不是单纯把生成做快，而是要支撑一个循环式任务执行过程。

## 2. 推理循环不同

### 2.1 普通推理框架

普通流程通常是：

```text
prompt
  ↓
prefill
  ↓
decode loop
  ↓
token output
```

它主要关心：

- KV cache
- sampling
- memory management

### 2.2 Agent 推理框架

Agent 的循环更像：

```text
用户输入
  ↓
LLM 推理
  ↓
判断是否调用工具
  ↓
执行 tool
  ↓
把 observation / result 回写上下文
  ↓
再次推理
```

所以框架必须支持：

- 中断生成
- 插入外部结果
- 再次继续推理

## 3. Context 管理复杂度更高

普通聊天的 context 通常只是：

- system prompt
- 历史对话
- 当前输入

而 Agent 的 context 往往还要加入：

- tool description
- tool result
- reasoning 痕迹
- memory

所以 context 增长会更快。

在端侧，这会直接带来：

- prefill 更慢
- KV cache 更大
- 显存压力更高

## 4. Tool Runtime 是新增能力

普通推理框架一般只关心模型执行。

而 Agent 推理框架需要把工具也纳入运行时：

- search
- vision
- control
- audio
- 本地数据库 / RAG

所以它需要额外有：

- tool registry
- tool interface
- tool scheduler

## 5. 调度目标不同

普通 LLM 推理框架更偏向：

- 最大化 token/s
- 批处理吞吐

而 Agent 场景更偏向：

- 最小任务完成延迟
- 推理、IO、tool 的协同
- 低时延抖动

也就是说，它更像一个任务执行引擎，而不只是 token engine。

## 6. Memory 系统也不同

普通 LLM 主要记的是：

- 当前 context
- KV cache

Agent 还常常需要：

- short-term memory
- long-term memory
- tool memory
- retrieval memory

所以它的 memory 结构会天然更复杂。

## 7. 一句话总结

普通推理框架优化的是 token 生成过程；
Agent 推理框架优化的是“任务执行循环”。

所以它从 token engine 变成了 decision engine。
