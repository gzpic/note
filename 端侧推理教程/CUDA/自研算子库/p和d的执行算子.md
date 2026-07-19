---

---
d:

| 序号 | 阶段 | 输入 → 输出 shape | 核心算子 | 推荐优化手段 |
| --- | --- | --- | --- | --- |
| D0 | input token | `[B, 1]` | index | CPU/GPU |
| D1 | Embedding | `[B, 1] → [B, 1, H]` | Gather | fuse |
| D2 | RMSNorm (attn) | `[B, 1, H]` | elementwise | fuse |
| D3 | QKV 投影 | `[B, 1, H] → [B, 1, 3H]` | GEMM | small GEMM |
| D4 | reshape | `→ [B, Nh, 1, Dh]` | view | packed |
| D5 | RoPE | `[B, Nh, 1, Dh]` | elementwise | fuse |
| D6 | append KV | `→ cache[t]` | memcpy | aligned / paged |
| D7 | Q×Kcacheᵀ | `[B, Nh, 1, t]` | GEMM | fused kernel |
| D8 | Softmax | `[B, Nh, 1, t]` | softmax | online softmax |
| D9 | Attn×Vcache | `[B, Nh, 1, Dh]` | GEMM | fused |
| D10 | attn out proj | `[B, 1, H]` | GEMM | small GEMM |
| D11 | Residual | `[B, 1, H]` | elementwise | fuse |
| D12 | RMSNorm (ffn) | `[B, 1, H]` | elementwise | fuse |
| D13 | FFN up | `[B, 1, H] → [B, 1, 4H]` | GEMM | fp16 |
| D14 | FFN gate | `[B, 1, H] → [B, 1, 4H]` | GEMM | fp16 |
| D15 | SwiGLU | `[B, 1, 4H]` | elementwise | fuse |
| D16 | FFN down | `[B, 1, 4H] → [B, 1, H]` | GEMM | fp16 |
| D17 | Residual | `[B, 1, H]` | elementwise | fuse |
| D18 | Final RMSNorm | `[B, 1, H]` | elementwise | fuse |
| D19 | LM Head | `[B, H] → [B, V]` | GEMM | top-k fuse |
| D20 | Sampling | `[B, V] → [B, 1]` | reduce | GPU sampling |

p:

| 序号 | 阶段 | 输入 → 输出 shape | 核心算子 | 推荐优化手段 |
| --- | --- | --- | --- | --- |
| P0 | input / pos 准备 | `[B, T]` | index / arange | CPU 预处理，GPU 合批 |
| P1 | Embedding | `[B, T] → [B, T, H]` | Gather | embedding + scale fuse |
| P2 | RMSNorm (attn) | `[B, T, H] → [B, T, H]` | elementwise | kernel fusion |
| P3 | QKV 投影 | `[B, T, H] → [B, T, 3H]` | GEMM | tensor core / fp16 / bf16 |
| P4 | reshape + split | `[B, T, 3H] → 3×[B, Nh, T, Dh]` | view | layout 设计 |
| P5 | RoPE(Q,K) | `[B, Nh, T, Dh]` | elementwise | fuse into QK kernel |
| P6 | 写 KV cache | `→ cache[T]` | memcpy | chunked / aligned write |
| P7 | Attention score | Q×Kᵀ → `[B, Nh, T, T]` | GEMM | FlashAttention |
| P8 | Mask + Softmax | `[B, Nh, T, T]` | softmax | fused softmax |
| P9 | Attn×V | `[B, Nh, T, Dh]` | GEMM | FlashAttention |
| P10 | attn reshape | `[B, T, H]` | view | NHWC / packed |
| P11 | attn out proj | `[B, T, H]` | GEMM | tensor core |
| P12 | Residual add | `[B, T, H]` | elementwise | fuse |
| P13 | RMSNorm (ffn) | `[B, T, H]` | elementwise | fuse |
| P14 | FFN up | `[B, T, H] → [B, T, 4H]` | GEMM | tensor core |
| P15 | FFN gate | `[B, T, H] → [B, T, 4H]` | GEMM | tensor core |
| P16 | SwiGLU | `[B, T, 4H]` | elementwise | fuse |
| P17 | FFN down | `[B, T, 4H] → [B, T, H]` | GEMM | tensor core |
| P18 | Residual add | `[B, T, H]` | elementwise | fuse |
| P19 | Final RMSNorm | `[B, T, H]` | elementwise | fuse |
| P20 | LM Head | `[B, H] → [B, V]` | GEMM | 只算最后 token |