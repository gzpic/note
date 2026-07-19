---

---
MLIR 的核心是：把一次性、单链式的编译过程，拆分为多个具有不同抽象层级的中间表示（IR），并允许在每一层上进行最适合该层语义的优化和转换。

### LLVM IR 太低层

LLVM IR 接近机器指令，不适合进行：

- 张量级别的优化（如 matmul → block → warp）
- 图层级的算子融合
- 并行调度/Tile/Vectorize 等结构化优化

换句话说：

```plain text
LLVM 是“太晚了”再做优化。


```

---

## 🪜 MLIR 做什么？

MLIR 提供多个 **可定制的中间层**，称为 **Dialect**，每一个 Dialect 抽象不同“语义层次”：

| 层级 | Dialect 示例 | 关心什么 | 可做的优化 |
| --- | --- | --- | --- |
| 高层算子语义 | MHLO / TOSA / Torch | 算子级别、计算图结构 | 算子融合、常量折叠 |
| 张量与循环表示 | Linalg / Tensor / Affine | Tile、循环并行、向量化 | Tile、Fuse、Vectorize、Unroll |
| 低层控制流 | SCF / CFG | 基本块级控制流 | Pass 驱动微优化 |
| 硬件目标层 | LLVM / NVVM / ROCDL | 对应具体指令集 | register 分配、调度、代码生成 |

**每一层 Lower 到下一层 → 都可以插入对应领域最有效的优化。**

---

## 🔥 为什么这很强？

因为不同的硬件需要不同的优化策略：

| 硬件 | 优化重点 |
| --- | --- |
| CPU | Cache 局部性、向量化（AVX / AMX） |
| NVIDIA GPU | Block/warp tiling、共享内存调度、warp mapping |
| TPU / NPU | 张量块映射、矩阵加速单元调度 |
| FPGA | 流水线 / Parallel schedule / HLS 结构化展开 |

**传统编译器没法统一处理这一切。
MLIR 用 Dialect 和 Lowering Pipeline 把这个问题模块化了。**

---

## 🧠 所以 MLIR 的“核心”不是 Lowering 本身，而是：

### ✅ **用多层 IR 拆解编译问题**

让每个优化“在对的层次发生”。

### ✅ **用 Dialect 描述领域语义**

保留高层信息，不会过早丢失结构化信息。

### ✅ **用 Pass Pipeline 逐层变换**

形成可控、可组合、可复用的优化链。

### ✅ **用 Pattern Rewrite 实现跨层映射**

让 lowering 不是“硬编码”，而是可扩展且自动化的。
