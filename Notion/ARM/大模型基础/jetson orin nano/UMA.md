# Jetson Orin Nano 的 UMA、Pinned、Zero-Copy、DMA 与 GPU 页表

## 1. UMA 是什么

`UMA` = `Unified Memory Architecture`，统一内存架构。

一句话版：

> CPU 和 GPU 共享同一块物理内存，而不是 CPU 一套内存、GPU 一套独立显存。

和传统离散显卡架构相比：

- 非 UMA：CPU 内存和 GPU 显存物理隔离
- UMA：CPU / GPU 指向同一套物理 DRAM

在 Jetson Orin Nano 上，这意味着：

- 没有独立显存这回事
- 系统内存要同时分给 OS、应用、模型权重、KV Cache、中间 buffer
- 真正的核心约束往往不是纯算力，而是统一内存容量和带宽

一句话：

> 在 Orin Nano 上，最怕的不是“算一下”，而是“多搬一下”。

## 2. UMA 和传统 dGPU 架构的区别

### 2.1 非 UMA / 离散显卡

可以粗理解成：

```text
CPU ── DDR
       │
       │ Host ↔ Device 拷贝
       │
GPU ── GDDR / HBM
```

特点：

- CPU 内存、GPU 显存物理隔离
- Host 和 Device 之间常常要显式 memcpy / DMA
- 显存带宽高
- 但拷贝成本也高

典型平台：

- RTX 3090
- A100
- H100

### 2.2 UMA / 统一内存架构

可以粗理解成：

```text
CPU ─┐
     ├── LPDDR / DDR（同一块物理内存）
GPU ─┘
```

特点：

- CPU / GPU 共享同一套物理内存
- 物理上没有独立显存
- Host ↔ Device 的物理割裂更小
- 但带宽和容量是共享资源

典型平台：

- Jetson Orin / Xavier
- Apple M1 / M2 / M3
- Snapdragon
- 部分 AMD APU

## 3. UMA 对大模型推理意味着什么

在端侧 LLM 推理里，UMA 的影响非常直接。

### 好处

- KV Cache 不需要跨独立显存域搬运
- CPU 管理、GPU 消费同一套物理内存更自然
- context cache / radix cache / 流式推理更容易做
- zero-copy 和 pinned buffer 更有工程价值

### 坏处

- CPU、GPU、DMA、外设都在抢同一套带宽
- decode 阶段本来就 memory-bound，更容易被打爆
- 多并发不一定带来更高吞吐
- batching 在 prefill 和 decode 上表现差异很大

一句话：

> 在 UMA 上，prefill 还能比较像算力问题，decode 往往更像带宽问题。

## 4. UMA、DMA、零拷贝三者的关系

可以直接背这三句：

- `UMA`：CPU 和 GPU 共用同一套物理内存
- `DMA`：专门硬件负责搬数据
- `零拷贝`：尽量减少无意义的数据副本和重复 memcpy

它们的关系是：

- `UMA` 决定了共享物理内存是底座
- `DMA` 决定了数据尽量不要让 CPU 亲自搬
- `zero-copy` 决定了工程上要尽量避免冗余 copy

但注意：

- `UMA` 不等于“完全没有搬运成本”
- `DMA` 不等于“不要 CPU / GPU 参与同步”
- `zero-copy` 不等于“数据完全不动”

## 5. Pinned Buffer 是什么

`Pinned memory` 也叫页锁定内存。

它的核心含义是：

- 这块 host 内存不会被 swap
- 物理页固定
- 设备可以直接通过 DMA 访问

常见 CUDA 接口：

- `cudaMallocHost`
- `cudaHostAlloc`

Pinned buffer 的价值：

- 为 DMA 提供稳定物理页
- 减少 staging copy
- 让 Host 到 GPU 的搬运更快、更稳定

但缺点也很明确：

- 占用物理内存
- 用多了会拖慢整个系统

## 6. Zero-Copy 是什么

`Zero-copy` 的意思不是“没有内存”，而是：

> GPU 直接访问 CPU 侧那块可映射的内存，不再额外做一次 Host → Device memcpy。

在 UMA 上它更自然，因为：

- 物理上本来就是同一块 DRAM
- 不像离散显卡那样要跨 PCIe 到独立显存

但 zero-copy 的代价是：

- GPU 每次访问更像直接打系统内存
- 延迟高
- 带宽低
- cache 命中率通常更差

一句话：

- zero-copy 省的是拷贝
- 付出的代价是访问路径变慢

## 7. Pinned 和 Zero-Copy 的区别

最重要的一句：

> `Pinned + DMA` 是先搬一次，后面高速访问；`zero-copy` 是不搬，但以后每次访问都走慢路径。

可以用这张表记：

| 项目 | Pinned + DMA | Zero-Copy |
| --- | --- | --- |
| 是否拷贝 | 有，一次 DMA | 无额外 Host→Device 拷贝 |
| GPU 访问 | 后续访问 device 路径 | 直接访问 host/共享内存 |
| 访问延迟 | 低 | 高 |
| 带宽 | 更高 | 更低 |
| 适合场景 | 大张量、反复访问 | 小数据、控制流、偶发访问 |

### 工程记忆法

- `Pinned + DMA`：先搬家，再高速住
- `Zero-copy`：不搬家，但天天通勤

## 8. 两者各自是怎么访问的

### 8.1 Pinned + DMA

流程是：

1. CPU 在 pinned host buffer 写数据
2. DMA / Copy Engine 把数据搬到 GPU 更适合访问的那块工作区
3. GPU kernel 后续反复访问这份数据

可以粗理解成：

```text
CPU
  ↓ write
Pinned Host Memory
  ↓ DMA（一次）
GPU 工作区 / Device 访问路径
  ↓
GPU SM 反复访问
```

它的本质是：

- 付一次搬运成本
- 换后面大量访问都走快路径

### 8.2 Zero-Copy

流程是：

1. CPU 在可映射 buffer 写数据
2. GPU kernel 直接访问这块共享物理页
3. 不做额外 memcpy

可以粗理解成：

```text
CPU 写共享内存
GPU 直接 load/store 这块共享页
```

它的本质是：

- 省掉搬运
- 但每次访问都比较慢

## 9. 在端侧 LLM 推理里怎么选

一句工程结论：

> 控制流 zero-copy，数据流 pinned / device。

### 更适合 zero-copy 的东西

- token id buffer
- decode 输出 token
- metadata
- ring buffer 指针
- scheduler 和 runtime 之间的小控制块

### 不适合 zero-copy 的东西

- KV Cache
- attention 中的大 tensor
- 权重
- activation

原因很直接：

- 这些数据会被 GPU 高频反复访问
- 如果它们走 zero-copy，就会持续打共享 DRAM
- 在 UMA 上很容易把带宽打爆

### 一句话记忆

- 小而偶尔用：zero-copy
- 大而反复用：不要 zero-copy

## 10. 为什么要区分 Host Memory、Pinned Host Memory、GPU Device Memory

这个点很容易误解。

在 UMA 上，它们通常不是“物理上硬切成三块内存”，而是：

> 同一套物理 DRAM 上，不同的逻辑分区 / 访问语义。

### 普通 Host Memory

更偏 CPU 世界：

- 低延迟
- 控制流友好
- 强虚拟内存语义
- 可以 swap

### Pinned Host Memory

本质上还是 host page，只是打了额外标签：

- 不可换出
- 物理页固定
- 可供 DMA / 设备访问

### GPU Device Memory

更偏 GPU 世界：

- 顺序、大并发访问
- 更强调高吞吐
- 更适合作为 GPU 工作集

一句话：

- pinned host memory 是“CPU 世界为设备让步”
- device memory 是“GPU 世界为吞吐而生”

## 11. 这是物理分区还是逻辑分区

结论先写死：

> 在 UMA 架构下，这种区分通常主要是逻辑分区，不是物理硬切分。

但这个“逻辑”不是虚的，它会真实改变：

- 页表映射
- cache 策略
- 访问路径
- 预取行为
- 内存控制器仲裁

所以：

- 同一块 DRAM
- 在 CPU 和 GPU 视角下
- 可以表现出非常不同的性能

一句话：

> UMA 是“物理不分家，逻辑要分家”。

## 12. 这种逻辑分区到底怎么实现

它不是一个单点机制，而是一组机制叠加：

- CPU 页表和 GPU 页表分开
- 页属性不同
- IOMMU / SMMU 映射不同
- CPU / GPU cache 策略不同
- GPU prefetch 行为不同
- Memory Controller 对不同流量的仲裁不同

可以把因果链记成：

```text
API 选择
→ Driver 分配策略
→ 页表与页属性
→ Cache / Prefetch 行为
→ Memory Controller 仲裁
→ 真实性能
```

所以“逻辑分区”不是口头约定，而是硬件行为集合。

## 13. GPU 有自己的 MMU 和页表

这个点要重点记：

> GPU 有自己的 MMU 和自己的页表，它不是借用 CPU 页表。

也就是说：

```text
CPU 访问：
CPU VA → CPU Page Table → PA

GPU 访问：
GPU VA → GPU Page Table → PA
```

它们共享的通常是：

- 物理内存

它们不共享的是：

- 虚拟地址空间
- 页表
- 访问语义

这点特别关键，因为它直接决定：

- CPU 指针不能直接给 GPU 用
- zero-copy 共享的是物理页，不是指针值
- 同一块 DRAM 可以同时有 CPU VA 和 GPU VA 两种视角

一句话：

> zero-copy ≠ 共享指针，zero-copy = 共享物理页 + 双页表映射。

## 14. 为什么 zero-copy 下 GPU 不能直接用 CPU 指针

根因是：

> 指针不是物理地址，而是“某个设备视角下的虚拟地址”。

CPU 指针只在 CPU 虚拟地址空间里有意义。

GPU 如果要访问同一块物理页，需要：

- 把这块物理页映射到 GPU 页表
- 给 GPU 一个属于自己地址空间的 pointer

所以 zero-copy 的典型流程才会是：

```cpp
cudaHostAlloc(&h_ptr, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
```

这里：

- `h_ptr` 是 CPU 视角虚拟地址
- `d_ptr` 是 GPU 视角虚拟地址
- 它们指向同一块物理页

也就是说：

- 共享的是 physical page
- 不是 pointer value

## 15. UMA、UVA、CUDA Unified Memory 不是一回事

这几个概念经常混：

| 概念 | 层面 |
| --- | --- |
| `UMA` | 硬件层统一物理内存 |
| `UVA` | 统一虚拟地址视图 / runtime 抽象 |
| `CUDA Unified Memory` | 软件分配模型 |
| zero-copy | 一种访问方式 |

所以：

- `UMA` 是“物理内存统一”
- `UVA` 是“看起来更像统一”
- `cudaMallocManaged` 在非 UMA 平台也能用，但底层可能发生 page migration

## 16. DMA 搬数据时，CPU 和 GPU 在干嘛

结论：

> DMA 搬数据时，真正干体力活的是 DMA / Copy Engine，不是 CPU ALU，也不是 GPU SM。

更准确地说：

- CPU 负责下命令、配置描述符、同步
- DMA / Copy Engine 真正搬数据
- GPU 的 SM 不负责逐字节搬运

所以在 DMA 期间：

- CPU 通常可以继续做别的任务
- GPU 的 SM 也可能继续跑别的 kernel

### 但为什么说“可能”

因为是否真能并行，还取决于：

- 是否用了异步 memcpy
- 是否在不同 stream 上
- 是否有独立 copy engine
- 是否被共享带宽拖慢

### 在 UMA 上的关键限制

虽然算力单元能并行，但：

- DMA
- GPU kernel
- CPU

仍然共享同一套 DRAM 带宽。

所以：

- Prefill 阶段，copy 和 compute overlap 往往还有收益
- Decode 阶段，如果本来就是 memory-bound，再叠 DMA 可能反而更慢

一句话：

> DMA 解放的是计算单元，不是内存带宽。

## 17. 对 Runner / Scheduler / LLM 推理框架设计的直接影响

在 Orin Nano 上做推理框架，通常会有这些设计偏好：

- 尽量预分配静态 buffer
- 控制类数据尽量 zero-copy
- 大块数据尽量 pinned + DMA 或 GPU 工作集路径
- 尽量减少大块 memcpy
- 尽量让 copy 和 compute overlap
- 尽量控制 KV Cache 增长
- 尽量避免 CPU 和 GPU 同时打满统一内存带宽

所以很多工程红线都来自 UMA：

- 禁止大规模无意义 memcpy
- decode 路径不要滥用 zero-copy
- prefill 和 decode 要分开思考
- KV Cache 必须按 GPU 高频工作集来设计

## 18. 一个直观例子

比如摄像头输入一帧，最后喂给 GPU 做推理：

低效路径：

- 摄像头 DMA 到一块内存
- CPU 再 copy 一份
- 再 copy 到 GPU 更适合访问的区域
- 再开始算

更合理的 UMA 路径：

- 摄像头先通过 DMA 写入共享 buffer
- 控制和小 metadata 直接 zero-copy 复用
- 大块会被反复用的数据再走 pinned + DMA 或 GPU 工作集路径
- CPU 只做调度和同步

改进点就是：

- 少 copy
- 少带宽浪费
- 少 CPU 参与

## 19. 面试里怎么答

可以直接这样说：

> Jetson Orin Nano 是 UMA 架构，CPU 和 GPU 共享同一套物理内存，所以端侧大模型推理的核心瓶颈往往不是单纯算力，而是统一内存的容量和带宽。DMA 仍然重要，因为它负责把外设或存储的数据异步搬到内存，减轻 CPU 负担；但在 UMA 上 DMA、CPU、GPU 仍然共享同一套带宽，所以搬运不是免费的。Pinned buffer 适合大块数据的一次搬运后反复使用，zero-copy 更适合小而频繁的控制类数据。GPU 还有自己独立的 MMU 和页表，所以 zero-copy 共享的是物理页，不是 CPU 指针本身。工程上因此要把控制流、数据流和 GPU 工作集分开设计，否则 decode 很容易被共享带宽拖垮。 

## 20. 速记版

- Orin Nano = `UMA`
- UMA 的核心问题 = 统一内存容量和带宽紧
- `Pinned + DMA` = 先搬一次，后面高速访问
- `Zero-copy` = 不搬，但以后每次访问都更慢
- GPU 有自己的 MMU 和页表
- zero-copy 共享的是物理页，不是指针
- 在 UMA 上，“少搬数据”往往比“多做一点算”更重要
