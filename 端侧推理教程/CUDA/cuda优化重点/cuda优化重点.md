---

---
### 一、线程调度与并行结构优化

| 优化点 | 说明 |
| --- | --- |
| ✅ **减少线程分歧（thread divergence）** | 同一个 warp 中线程尽量走相同控制路径（避免 `if`/`switch` 分支不一致） |
| ✅ **保证 warp 大小对齐（32 的倍数）** | blockDim.x 最好是 32、64、128、256、512 等，提高 warp 利用率 |
| ✅ **合理设计 grid/block** | 让线程数覆盖整个数据空间，避免浪费计算资源 |
| ✅ **使用多个 block 而非超大 block** | 有利于 SM 间负载均衡、提高 occupancy |

---

### 💾 二、内存访问与带宽优化

| 优化点 | 说明 |
| --- | --- |
| ✅ **合并内存访问（coalesced access）** | warp 中线程顺序访问连续地址，提升 global memory 访问效率 |
| ✅ **多使用共享内存（shared memory）做计算** | 利用共享内存的高速特性减少 global memory 的访问次数 |
| ✅ **避免 shared memory bank conflict** | 不同线程访问不同 bank，提高并行性 |
| ✅ **避免使用本地内存（local memory）** | 超出寄存器的变量会落入 local memory（其实是 slow DRAM） |
| ✅ **用常量内存（**`**__constant__**`**）缓存小只读表** | warp 广播机制读同一地址超快 |
| ✅ **使用纹理内存访问图像** | 有 cache 和插值加速，适合图像处理 |

---

### ⚙️ 三、计算效率与资源利用优化

| 优化点 | 说明 |
| --- | --- |
| ✅ **最大化 occupancy（并发块数）** | 不要让某个资源（register/shared memory）限制 block 并发数 |
| ✅ **减少寄存器压力** | 每线程使用寄存器过多会降低并发度（warp 数减少） |
| ✅ **最小化分支/循环** | 在 GPU 上每个判断都是负担，尽量“扁平化”逻辑或用 lookup table 代替 |
| ✅ **避免在 device 上做 malloc/free** | 使用预分配内存，避免频繁分配释放导致性能下降 |
| ✅ **融合计算和访存（loop tiling/fusion）** | 将多个 kernel 合并，减少访存轮次和 kernel launch overhead |

---

### ⛓️ 四、调度与执行优化

| 优化点 | 说明 |
| --- | --- |
| ✅ **使用多个 stream 实现异步并发（overlap）** | 异步执行 kernel + memcpy，提升 GPU 吞吐 |
| ✅ **合理 pipeline kernel 调度顺序** | 避免 kernel 阻塞，提升流水线效率 |
| ✅ **使用 CUDA Graph 或 cooperative kernel** | 多 kernel 间联合调度、减少 launch overhead（适合小 kernel 多次执行） |

gpu分支预测

[[1.精度介绍]]