---

---
**大模型核心算子与硬件架构详尽实现表**

| 算子分类 | 核心算子名称 | 对应硬件实现 (SM版本) | 数据格式与关键技术特性 | 备注 |
| --- | --- | --- | --- | --- |
| **注意力机制 (Attention)** | **Flash Attention v2 (FMHA)** | SM80, SM89, SM90, SM100f | BF16, FP16, E4M3 (FP8); 利用 **TMA** (Tensor Memory Accelerator) 和 **WS** (Workspace) 优化 | 推理 Prefill 阶段核心实现 |
|   | **Flash MLA (多头潜在注意力)** | SM90 | BF16, FP16, FP8; 针对 **DeepSeek** 等模型优化 | 显存压缩型注意力实现 |
|   | **XQA (高效查询注意力)** | SM80, SM86, SM89, SM90, SM120 | BF16, FP16, INT8, E4M3; 支持 **Paged KV Cache** | 解码 (Generation) 阶段核心算子 |
|   | **Sage / Sparse Attention** | SM89, SM90 | 包含稀疏注意力逻辑与 SageAttention 优化实现 | 针对长文本和稀疏场景 |
| **矩阵乘法 (GEMM/BMM)** | **Batched GEMM (BMM)** | SM100a, SM100f, SM103a | BF16, FP32, E2M1, E4M3; 支持 **Warp-specialized** 调度和 **TMA** | Blackwell 架构下的高性能批处理矩阵乘法 |
|   | **FP4 / FP8 GEMM** | SM89, SM90, SM100, SM120 | **FP4 (NVFP4/MXFP4)**, FP8 (Block-scale/Row-wise); 利用 CUTLASS 模板实现 | 极低精度量化推理，显著提升吞吐量 |
|   | **Weight-Only GEMV** | SM90 (Hopper专用优化) | BF16/FP16 激活 + **INT4/INT8** 权重; 支持 Group-wise 和 Per-channel 量化 | 针对内存带宽受限场景的权重量化 |
|   | **All-Reduce 融合 GEMM** | SM90, SM100 | BF16; 利用 **NVLS** (NVIDIA Link System) 进行计算与通信融合 | 解决多卡并行下的通信延迟问题 |
| **混合专家模型 (MoE)** | **MoE Routing (路由调度)** | 通用 / 深度适配实现 | 针对 **DeepSeek** 和 **Llama4** 的专用路由逻辑 | 专家动态选择核心算子 |
|   | **Fused MoE GEMM** | SM80, SM90 | 支持 BF16, FP16, FP4, FP8 等多种混合精度输入 | 将计算、激活与通信 (All-to-All) 深度融合 |
| **序列建模 (SSM/Mamba)** | **Selective Scan (选择性扫描)** | 通用实例化实现 | 支持 BF16, FP16, FP32 的实例化 | **Mamba** 架构的核心循环算子 |
|   | **Causal / Mamba Conv1d** | 通用实现 | 针对序列建模的一维因果卷积优化 | 配合 SSM 的预处理算子 |
| **规范化与激活 (Norm & Act)** | **Norm 算子** | 通用 / 低延迟实现 | LayerNorm, RMSNorm, **Group RMSNorm**; 支持 FP4 转换 | 维持模型数值稳定性 |
|   | **Fused Gated GEMM** | SM90, SM100f | 矩阵乘法融合 **SwiGLU** 等门控激活函数; 支持 E4M3 格式 | 减少中间显存读写 |
| **解码增强 (Decoding)** | **投机采样 (Speculative)** | 通用逻辑优化 | **Medusa** (美杜莎), **Eagle**, **MTP** (多标记预测) | 通过预测后续 Token 加速解码 |
|   | **Beam Search (束搜索)** | 模板化实现 | 支持 16, 32, 64, 128, 256, 512, 1024 等不同束宽 (Beam Width) | 传统的高质量解码搜索实现 |