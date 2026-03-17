# OpenClaw 整体架构

## 1. 一句话理解

OpenClaw 可以理解成：

> Gateway + Agent Runtime + LLM + Tools + Memory

它不是单纯的聊天机器人，而是一个带通信入口、本地执行能力和工具生态的 Agent 平台。

## 2. 整体分层

可以粗略画成：

```text
Communication Layer
        ↓
Gateway
        ↓
Agent Runtime
        ↓
LLM
        ↓
Tools / Skills
        ↓
Memory
```

其中：

- LLM 负责决策
- Runtime 负责控制循环
- Tools 负责执行
- Gateway 负责接外部世界

## 3. Communication Layer

这一层负责用户交互入口，例如：

- Slack
- Telegram
- WhatsApp
- Discord

用户并不是在一个专门网页里和它聊天，而是在现有通信工具里发消息。

## 4. Gateway

Gateway 是中心控制层，主要负责：

- 消息接入
- 用户身份
- 会话路由
- 不同 Agent / 会话之间的分发
- LLM provider 管理

它更像一个控制中枢。

## 5. Agent Runtime

这一层是真正的 Agent 核心，负责：

- prompt 构造
- tool calling
- context 管理
- observation 回写
- 多轮推理循环

也就是：

- Thought
- Action
- Observation

这套循环的调度器。

## 6. LLM 层

LLM 层负责：

- 接收 prompt
- 输出 token
- 做推理决策

它可以是：

- Claude
- GPT
- Gemini
- 本地模型

所以 OpenClaw 的模型层通常是可替换的。

## 7. Skills / Tools

OpenClaw 把执行能力封装成 skills，例如：

- filesystem
- shell
- browser
- email
- api

本质上就是一组可被 Agent 调用的函数能力。

## 8. Memory

Memory 负责保存：

- conversation history
- persistent state
- 文件和日志
- 某些长期状态信息

它让 Agent 不是一次性会话，而是一个可以持续工作的系统。

## 9. 为什么要把 Gateway 和 Runtime 分开

这个设计很关键，因为：

- Gateway 负责接入和路由
- Runtime 负责推理与执行循环

分开后更容易做：

- 多用户
- 多 Agent
- 并发调度
- 模型和工具解耦

## 10. 一句话总结

OpenClaw 的本质不是“一个模型应用”，而是：

> 一个把通信入口、Agent runtime、LLM、工具系统和状态管理连接起来的本地自动化平台。
