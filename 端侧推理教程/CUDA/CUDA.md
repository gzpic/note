---

---
# CUDA全景图

## 软件 

### 计算&内存

       一句话总结，高性能计算的重点无非存和算亮相，算都不是制约计算性能的关键因素，存才更关键

<u>**cuda计算模型**</u>

grid-block-thread三级分层

![[image 7.png]]

- 线程-顾名思义
- block-一组线程的集合，一个 block 最多只有 1024 个线程（硬件上限）。
block和tread是实现SIMT的关键，cuda的线程都是SMIT(Single Instruction, Multiple Threads)，同一个block里的线程执行相同的指令。？如果执行到了分支指令例如if，会怎么办呢？部分线程会等待，这个叫分歧是，这个是cuda编程需要优化的重点！ 线程分歧
- grid-一个应用程序需要的线程数有时会大于单block支持的最大数量，这个时候就需要调度多个block给一个应用程序，为了方便组织大规模bock进行并行计算，（）
大模型训练/推理 上百万个 block，就有了grid的概念。最简单的，百万个block如何索引？有了grid更方便。
- [ ] grid、block最大支持三维。

> myKernel<<<gridDim, blockDim>>>(...);

## 硬件

![[image 8.png]]

![[image 9.png]]

gpu的实际

一个grid里的多个block会被分到多个sm当中执行，

core - 线程执行位置

warp- 一个warp包含多个core，一个bock的32个线程会放入又给warp执行的

sm- 一block会被调到一个sm执行且一个sm可以有多个block。

内存

![[image 10.png]]

寄存器： kernal的一些变量会放在者，线程独有，不能共享 SRAM

本地内存：寄存器溢出的变量会放在这里，线程独有，不能共享 DRAM

共享内存：同一个block内的线程共享资源用的内存 SRAM

全局内存：储存主力，数据从host来-到host去通过全内，所有线程可访问，跨block共享数据的可DRAM

常量内存：高速缓存常量。所有线程只读，且共享。 DRAM +CACHE

纹理内存：为图像处理设计。 DRAM+自己专用的cache

cuda优化的在内存方面的核心是尽量在共享内存上计算避免全局内存。

| CUDA 内存类型 | 硬件类型 | 所在位置 | 访问延迟 | 共享范围 | 缓存策略 / 特点 |
| --- | --- | --- | --- | --- | --- |
| 🟢 **寄存器** | SRAM | SM 内部（on-chip） | 1 cycle | 线程私有 | 不可缓存（本身就是最顶层） |
| 🟢 **共享内存** | SRAM | SM 内部（on-chip） | 1–2 cycles | block 内共享 | 可手动读写，需使用 `__syncthreads()` 同步 |
| 🔵 **本地内存** | DRAM | GPU 显存（off-chip） | 400–600 cycles | 线程私有 | 自动缓存到 L1/L2（与 global 类似） |
| 🔵 **全局内存** | DRAM | GPU 显存（off-chip） | 400–800 cycles | 所有线程共享 | 自动缓存到 L1 和 L2，可设置缓存策略 |
| 🟡 **常量内存** | DRAM + 常量缓存 | GPU 显存 + L1 constant cache | 几百 cycles（若命中 cache 非常快） | 所有线程共享（只读） | 有独立常量缓存，warp 访问相同地址会广播 |
| 🟡 **纹理内存** | DRAM + 纹理缓存 | GPU 显存 + texture cache | 快（命中） / 慢（未命中） | 所有线程共享（只读） | 有局部性优化的专用 cache，支持插值与边界处理 |
| 🟡 **表面内存（surface）** | DRAM + surface cache | GPU 显存 + cache | 类似纹理 | 所有线程共享（可读写） | 用于图像/3D 图的 IO，支持缓存 |
| 🔶 **L1 Cache** | SRAM | 每个 SM on-chip | ~20–30 cycles | block 内线程共享（隐式） | 缓存 global/local memory 读写，可配置开启/关闭 |
| 🔶 **L2 Cache** | SRAM | GPU 统一片上缓存 | ~100 cycles | 所有 SM 共享 | 所有 memory 的最后一级缓存，全局共享 |

## 五、“黄金经验口诀”（工程实践总结）

- ✅ **“计算在 shared，数据来自 global”**：把计算搬进共享内存。
- ✅ **“warp 不分家”**：线程要齐步走，避免 warp 分歧。
- ✅ **“访存要对齐”**：coalesced access 提速显著。
- ✅ **“kernel 要精简”**：kernel 太复杂，性能反而差。
- ✅ **“occupancy 别浪费”**：register 和 shared memory 用太多会压低并发数。
- ✅ **“能 broadcast 就用常量”**：常量内存让线程一起读快得多。
- ✅ **“stream 用得好，CPU 空不了”**：stream 异步重叠计算和 IO，最大化利用。

[[note/Notion/CUDA/常用算子整理/常用算子整理]]

[[自研算子库]]

[[cuda优化重点]]

[[知识点整理]]

接下来将会用矩阵加法、矩阵乘法、转置三个例子说明在计算和内存上的优化点。

[[模型咋部署]]