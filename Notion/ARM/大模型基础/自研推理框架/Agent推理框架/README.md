# Agent 推理框架

这组笔记用于整理端侧 Agent 推理框架相关内容，重点关注：

- Agent 场景与普通 LLM 推理框架的差异
- 端侧 Agent 为什么更慢、但为什么仍然有价值
- 推理框架层在 Agent 场景下需要发生哪些变化
- context、KV cache、context reuse 的关系

## 目录

- `01-Agent推理框架与普通推理框架的差异.md`
- `02-端侧Agent为什么慢以及什么时候值得做.md`
- `03-Agent场景下推理框架层的变化.md`
- `04-context是什么以及端侧怎么管理.md`
- `05-context和KVCache的关系.md`
- `06-Agent工作的基本原理.md`
- `07-Agent输出解析、修复与格式约束.md`
- `08-OpenClaw整体架构.md`
- `09-Agent Runtime与Planner.md`
- `10-Agent记忆机制与Retrieval Memory.md`

## 使用方式

- 想看系统设计差异，先看 `01` 和 `03`
- 想看端侧价值判断，重点看 `02`
- 想看上下文和缓存管理，重点看 `04` 和 `05`
