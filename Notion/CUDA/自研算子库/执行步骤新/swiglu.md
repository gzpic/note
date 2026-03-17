---

---
1.公式介绍（重新弄）

![[image 14.png]]

```javascript
2. 简单代码
// x: [B, d_model]

// fused GEMM
[g, u] = matmul(x, W_concat)   // [B, 2*d_ff]

// SwiGLU
for i in elements:
    s = sigmoid(g[i])
    y[i] = u[i] * g[i] * s

// down projection
out = matmul(y, W_down)
```

1. 常用的优化：

权重水平融合（Wgate和Wup两矩阵融合）

| **优化方向** | **借鉴原代码的思想** | **Orin Nano 具体落地手段** |
| --- | --- | --- |
| **计算流** | 权重合并 (FC1) | 将 Gate/Up 权重合并，单次 GEMM 调用。 |
| **带宽** | Epilogue Fusion | 使用 CUTLASS/TensorRT 将 SiLU/Swish 融合进 GEMM。 |
| **带宽** | 向量化访存 | 自定义 Kernel 必须使用 `float4`/`int4` 加载 128-bit 数据。 |
| **显存** | W4A16 量化 | 实现权重的 INT4 存储、FP16 计算（Groupwise Scale）。 |
| **显存** | Buffer 复用 | 精细规划显存池，复用中间变量 Buffer。 |

4.本质上：swish+elementwise+sgemm