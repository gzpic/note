# Agent Runtime 与 Planner

## 1. Agent Runtime 是什么

Agent Runtime 可以理解成：

> LLM 控制器 + 状态机 + 工具执行器

它负责把一个“只会生成 token 的模型”变成“能完成任务的系统”。

## 2. Runtime 的核心职责

一个典型 Agent Runtime 通常负责：

- Context Manager
- Prompt Builder
- LLM Interface
- Output Parser
- Tool Executor
- Memory Manager
- Error Handler

也就是说，Runtime 不只是“调一下模型”，而是在管理整个任务循环。

## 3. ReAct Loop

几乎所有 Agent Runtime 都有一个类似的循环：

```text
Thought -> Action -> Observation
```

流程是：

```text
用户任务
  ↓
LLM 推理
  ↓
是否调用工具
  ├─ 否：继续生成或结束
  └─ 是：执行工具，写入 observation，再继续推理
```

## 4. Runtime 其实是一个状态机

可以把 Runtime 抽象成几个状态：

- `THINK`
- `ACT`
- `OBSERVE`
- `FINISH`

状态会不断切换，直到任务结束。

## 5. Planner 是什么

Planner 的作用是：

> 把复杂任务拆成多个可执行步骤。

例如：

```text
帮我订机票并安排酒店
```

Planner 可能先生成：

1. 搜索航班
2. 预订机票
3. 搜索酒店
4. 预订酒店

## 6. Planner 一般是谁生成的

Plan 不是推理框架生成的，而是：

- 由 LLM 生成
- 由 Runtime 解析和执行

推理框架只是在背后负责：

- 逐 token 计算

它并不知道自己正在生成“plan”。

## 7. Planner 的常见模式

### 7.1 ReAct

没有显式 plan，边想边做。

### 7.2 Plan-then-Execute

先完整生成计划，再逐步执行。

### 7.3 RePlan

执行过程中如果环境变化，再重新规划。

## 8. 为什么 Planner 很重要

因为复杂任务如果没有规划，很容易：

- 顺序错乱
- 漏步骤
- 重复调用工具

Planner 提供的是：

- 任务拆解
- 执行顺序
- 步骤间依赖关系

## 9. Runtime 和推理框架的关系

Runtime 负责：

- 构造 prompt
- 控制循环
- 解析输出
- 调工具

推理框架负责：

- prefill
- decode
- token 输出

所以关系是：

```text
Runtime 控制流程
Inference Engine 负责算 token
```

## 10. 一句话总结

Agent Runtime 是控制器，Planner 是任务拆解器，LLM 是计划和决策的生成者，推理框架只是底层 token 计算引擎。
