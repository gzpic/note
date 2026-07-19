---

---
- **Prefill Queue 与 Decode Queue 分离调度**
    - 新请求进入 prefill
    - 完成 prefill 后进入 decode
- **Chunked Prefill（分段预填充）**
    - 长 prompt 被切成多个 chunk
    - 避免长 prefill 阻塞 decode
    - 支持 continuous batching
- **Decode 优先级最高，可随时插队**
    - decode > prefill_chunk > prefill
    - 保障生成延迟稳定
    - 是 continuous batching 成功的前提
- **Prefill / Decode 异步调度、统一时间片执行**
    - 队列分离，但 GPU 执行串行
    - 调度器根据优先级动态选下一步执行哪种阶段
- **独立优化路径**
    - **Prefill：** 大算子、吞吐优先、chunking
    - **Decode：** 小算子、CUDA Graph、continuous batching
→ 两者优化方向完全不同，因此必须分离

prifill可在decode的overlap执行的
