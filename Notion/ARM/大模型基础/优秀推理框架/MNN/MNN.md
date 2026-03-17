---

---
1. 整体推理流程

demo_llm

1.

整体推理流程

[[LLM class]]

**✔ needCopy：数据必须复制吗？**

因为 dtype、layout、backend、pack 或 stride 不兼容。

**✔ needMalloc：需要新申请设备内存吗？**

因为 shape/dtype/backend 指针不兼容。

**✔ needCopy=true 通常伴随 needMalloc=true**

但 decode 阶段 often needCopy=false、needMalloc=false（极快）。


[[后端框架]]
