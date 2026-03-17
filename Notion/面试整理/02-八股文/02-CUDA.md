# CUDA

## 1. 一句话先讲清楚 CUDA

- CUDA 是 NVIDIA 的并行计算平台和编程模型，本质是把大量相同或相似的小任务分配给 GPU 上很多线程并行执行。
- CUDA 代码通常围绕两件事优化：
  - 算子怎么并行拆分
  - 数据怎么高效搬运
- 真正决定性能的往往不是算得够不够快，而是数据能不能高效送到计算单元。

## 2. CUDA 执行模型

- 组织层级是 `grid -> block -> thread`。
- block 内线程会按 32 个线程组成一个 warp。
- warp 是调度和执行的基本单位，所以很多优化都围绕 warp 展开。
- block 会被调度到某个 SM 上执行，一个 block 不会跨 SM。
- 一个 SM 可以同时驻留多个 block 和多个 warp，靠多 warp 切换来隐藏访存延迟。

面试回答时可以这么说：

- thread 是最小逻辑计算单元。
- warp 是最小执行单元，32 个线程同步前进。
- block 是资源分配单元，共享 shared memory 和同步原语。
- grid 是为了描述更大规模并行任务。

## 3. CUDA 硬件和内存层次

### 3.1 计算单元

- GPU 由多个 SM 组成。
- SM 内有 CUDA Core、Tensor Core、寄存器文件、shared memory、warp scheduler。
- 对大模型推理来说，GEMM 更依赖 Tensor Core，逐元素和小归约更依赖普通 CUDA Core 与访存效率。

### 3.2 内存层次

- Register
  - 最快，线程私有
  - 数量有限，寄存器太多会压低 occupancy
- Shared Memory
  - 位于 SM 内，block 内共享
  - 延迟低，适合 tile 缓存和中间结果复用
- L1 Cache
  - SM 私有
  - 对空间局部性敏感
- L2 Cache
  - 全 GPU 共享
  - 对时间局部性敏感，decode 场景很关键
- Global Memory
  - 容量最大，延迟最高
  - 访存模式写得差时，性能会非常差
- Local Memory
  - 名字叫 local，本质还是显存
  - 常见于寄存器 spill，开销很大

### 3.3 面试常问的重点

- shared memory 快，是因为在芯片上，延迟远低于 global memory。
- 但 shared memory 也有限，而且可能出现 bank conflict。
- L2 是整个 GPU 共享缓存，对 KV Cache 读取命中率影响很大。
- 寄存器不够时会 spill 到 local memory，性能会明显下滑。

## 4. 为什么 CUDA 优化主要看访存

- 很多推理算子不是纯 compute-bound，而是 memory-bound。
- 典型 memory-bound 算子：
  - RMSNorm
  - layernorm
  - softmax
  - elementwise add / mul
  - KV Cache 读写
- 这类算子 FLOPs 不高，但读写量大，所以重点不是“多做算”，而是“少搬数据、搬得更整齐”。

面试里可以直接给出这句：

- CUDA 优化的主线通常是先保证访存模式正确，再考虑 Tensor Core、融合和调度。

## 5. 访存优化的几个核心原则

### 5.1 Coalesced Access

- 一个 warp 内线程尽量访问连续且对齐的地址。
- 这样硬件能合并成更少的 memory transaction。
- 如果访问不连续、跨 stride 或对齐差，带宽利用率会很低。

### 5.2 向量化加载

- 常见做法是 `half2`、`float4` 这类向量化 load/store。
- 作用：
  - 降低指令条数
  - 增大单次访存有效载荷
  - 提高吞吐
- 代价：
  - 需要地址对齐
  - 可能增加寄存器压力

### 5.3 Shared Memory 复用

- 把会重复访问的数据先搬到 shared memory。
- 常见场景：
  - GEMM tiling
  - block 内归约
  - 转置
- 需要注意：
  - bank conflict
  - `__syncthreads()` 的同步成本

### 5.4 减少中间结果落回显存

- 多个小算子串联时，如果每一步都写回 global memory，成本很高。
- 所以经常会做 kernel fusion，把多个 elementwise / reduce 步骤合并。

## 6. Warp、分歧与线程间通信

### 6.1 Warp Divergence

- 一个 warp 内线程走不同分支时，会串行执行不同路径。
- 这就是线程分歧。
- 分歧会让有效并行度下降。

常见优化：

- 尽量让同一 warp 处理相似数据
- 减少复杂 if/else
- 用查表或掩码代替重分支

### 6.2 Warp 级通信

- 常用原语是 `__shfl_sync` 系列。
- 作用是在 warp 内直接交换寄存器数据，不必写 shared memory。
- 常用场景：
  - warp reduce
  - prefix sum
  - 广播某个 lane 的值

面试里一句话总结：

- warp shuffle 适合小范围归约，开销低于 shared memory + block 同步。

## 7. Occupancy 和资源约束

- occupancy 指一个 SM 上同时驻留的活跃 warp 数量占理论上限的比例。
- occupancy 高不代表一定最快，但太低通常很难隐藏访存延迟。

影响 occupancy 的主要因素：

- 每线程寄存器数
- 每 block 使用的 shared memory
- block 大小
- 硬件最大活跃 warp / block 限制

面试中不要说“occupancy 越高越好”，更稳妥的说法是：

- occupancy 要足够高以隐藏延迟，但最终还是要看实际瓶颈是算力、带宽还是依赖链。

## 8. 常见 CUDA 优化手段

### 8.1 并行结构优化

- block size 选 32 的倍数
- 让 grid 覆盖完整数据空间
- 避免 block 过大导致资源浪费
- 保证任务划分均匀，减少尾部线程空转

### 8.2 访存带宽优化

- coalesced access
- 向量化 load/store
- 使用 shared memory 缓存热点数据
- 减少不必要的 global memory 往返
- 避免寄存器 spill

### 8.3 计算效率优化

- 尽量走 Tensor Core 路径
- 降低控制流复杂度
- 融合多个小 kernel
- 减少 kernel launch overhead

### 8.4 调度执行优化

- 用 stream 做异步 overlap
- 合理安排 memcpy 和 kernel 顺序
- 固定 shape 场景下可用 CUDA Graph

## 9. CUDA Graph

- CUDA Graph 是把一串 kernel launch、memcpy、memset 和依赖关系录成一张图，后续可重复提交。
- 主要收益是减少 CPU launch 开销。
- 适合：
  - 固定 shape
  - 固定地址
  - 重复执行很多次的工作流
- 不适合：
  - 动态 shape 很多
  - 参数地址频繁变化
  - 图结构经常变

面试常用回答：

- 小 kernel 很多时，CPU launch 可能成为瓶颈，这时候 CUDA Graph 会比较有效。

## 10. 精度与量化

### 10.1 常见精度

- FP32
  - 精度高，开销大
- FP16
  - 吞吐高，显存占用低
  - 指数位少，容易溢出
- BF16
  - 指数范围接近 FP32
  - 更稳，训练和推理都常见
- INT8 / FP8 / INT4
  - 主要为了进一步提升吞吐和压缩显存

### 10.2 面试里怎么答精度选择

- 如果更关注稳定性，通常 BF16 比 FP16 更稳。
- 如果更关注吞吐和显存，低精度更有优势。
- 量化不是简单降 bit，核心是误差控制、outlier 处理和 kernel 适配。

## 11. 常见性能指标

### 11.1 带宽和访存

- `dram__throughput`
- `l2` 相关吞吐和命中情况
- `gld_efficiency`
- `gst_efficiency`

### 11.2 计算利用率

- `sm__throughput`
- Tensor Core 指令占比
- `eligible_warps_per_cycle`

### 11.3 Stall 原因

- memory dependency
- long scoreboard
- exec dependency
- not selected

### 11.4 指令结构

- ld/st 占比
- fma / mma 占比
- control 指令占比

## 12. 工具怎么分工

- `nsys`
  - 看宏观时间线
  - 看 CPU/GPU 协同
  - 看 launch gap、stream overlap、memcpy 与 kernel 排布
- `ncu`
  - 看单个 kernel 为什么慢
  - 看 occupancy、寄存器、shared memory、stall、Tensor Core 利用率

## 13. 面试高频判断题

- `gld_efficiency` 很低
  - 优先怀疑地址不连续、没对齐、stride 访问
- DRAM 吞吐不高，SM 利用率也不高
  - 往往不是纯带宽瓶颈，而是并行度不足或依赖链过长
- eligible warps 很低
  - 说明可运行 warp 不够，隐藏不了延迟
- control 指令占比高
  - 分支和循环结构可能不够友好
- local memory 很高
  - 常见是寄存器 spill

## 14. 面试收尾版答案

- CUDA 优化我通常会分四层看：
- 第一层看并行划分对不对，block 和 warp 利用率够不够。
- 第二层看访存模式，是否连续、对齐、可向量化、是否能放 shared memory。
- 第三层看资源利用率，寄存器、shared memory、occupancy 有没有互相卡住。
- 第四层看工程手段，比如 kernel fusion、CUDA Graph、stream overlap 和 profiling 工具定位。

## 15. GPU 内存结构和共享内存优化

- L1 Cache
  - 每个 SM 私有
- Shared Memory
  - 每个 SM 内、block 共享
- L2 Cache
  - 全 GPU 共享

### 15.1 什么是 bank conflict

- shared memory 被划成多个 bank。
- 如果同一时刻多个线程访问同一个 bank 的不同地址，就会串行化。
- 这就是 bank conflict。

常见优化：

- 调整数据布局
- padding
- 让访问模式更规整

## 16. SIMT、SIMD、Stream 和 Occupancy

### 16.1 SIMT 和 SIMD

- SIMD
  - 一条指令处理多个数据
  - 更强调硬件向量寄存器
- SIMT
  - 单指令多线程
  - CUDA warp 内 32 个线程执行同一条指令

### 16.2 Stream

- stream 是一条 CUDA 操作队列。
- 同一 stream 内默认有顺序。
- 不同 stream 之间可以并发，常用来让计算和数据传输重叠。

### 16.3 Occupancy 再补充

- occupancy 和寄存器数、shared memory、block size 都有关。
- 高 occupancy 不一定最快，但过低通常不好。

### 16.4 block 能不能跨 SM

- 不能。
- 一个 block 只能在一个 SM 上执行。

## 17. block / grid 为什么影响性能

- block 太少，SM 吃不满。
- block 太大，可能压爆寄存器或 shared memory。
- block 线程数不是 32 的倍数时，最后一个 warp 可能不满。
- grid 太小会导致整体并行度不够。

经验上：

- block size 常选 128、256、512 这类 32 的倍数。
- grid 数量通常至少让每个 SM 都有活干。

## 18. CUDA 常见工程问题

### 18.1 锁页内存

- 页锁定内存也叫 pinned memory。
- GPU 可以更高效地通过 DMA 与这块内存传输数据。
- 优点是 H2D / D2H 更快。
- 缺点是占用物理内存、申请释放成本更高。

### 18.2 CUDA 计时

- 测 kernel 时间常用 CUDA Event。
- 测整段端到端时间也可以用 `std::chrono`。

### 18.3 规约

- 常见做法：
  - 树形规约
  - 相邻规约
  - 多级规约
- 优化重点：
  - shared memory
  - 减少同步
  - warp 级 shuffle

## 19. 面试补充题

- CPU 和 GPU 的区别：
  - CPU 擅长复杂控制和低延迟单线程
  - GPU 擅长大规模并行数据处理
- A100 常见记忆点：
  - L2 大约 40MB
  - 每个 SM 的 L1 / shared memory 组合空间较大

## 20. 手撕 CUDA 题怎么答

### 20.1 写 CUDA 的标准流程

1. 准备主机端数据
2. 申请 device 内存
3. 把数据从 host 拷到 device
4. 配置 `grid` 和 `block`
5. 启动 kernel
6. 同步并检查错误
7. 把结果拷回 host
8. 释放资源

面试里如果只要说思路，这个顺序基本够用。

### 20.2 100 万个浮点数求和怎么做

- 最直接的思路是并行规约。
- 每个 block 先把一段数据读到 shared memory。
- 块内先做局部规约，得到 block 级部分和。
- 再把多个 block 的部分和继续规约，直到得到最终结果。

优化点：

- 合并访存
- shared memory
- warp 级 shuffle
- 分层规约
- 避免大量原子操作

### 20.3 CUDA Softmax 怎么做

- 先做一遍 reduce 找最大值
- 再减去最大值做 `exp`
- 再做一遍 reduce 求和
- 最后归一化

优化重点：

- 数值稳定性
- warp / block 归约
- 减少中间结果写回

### 20.4 CUDA LayerNorm / RMSNorm 怎么做

- 一般先做均值或平方和归约
- 再做归一化和仿射变换
- 常见是 memory-bound

优化重点：

- 向量化加载
- warp 级归约
- 与后续线性层融合

## 21. CUDA 矩阵乘法怎么讲

### 21.1 最基础思路

- 一个线程负责输出矩阵中的一个元素
- 沿着 K 维做乘加
- 这是最容易写对的 baseline

### 21.2 为什么要分块

- A 和 B 中一块数据会被多个线程重复使用
- 如果直接从 global memory 反复读，带宽开销太大
- 所以通常把 tile 先搬到 shared memory，再复用计算

### 21.3 经典优化点

- tile 化
- shared memory 缓存
- coalesced access
- 避免 bank conflict
- 向量化 load/store
- Tensor Core
- double buffering

### 21.4 为什么常见 `16x16` 或 `32x32`

- 和 warp 组织、shared memory 大小、数据复用方式比较匹配
- 对支持 Tensor Core 的硬件，也更容易贴合底层 mma 单元
- 但最终还是要看具体硬件和矩阵规模，不是固定答案

### 21.5 如果被问为什么 block/grid 会影响 GEMM 速度

- block 太小，数据复用不够
- block 太大，资源吃太多
- grid 太小，SM 吃不满
- block 形状还会直接影响访存是否连续

## 22. CUDA 规约怎么讲

### 22.1 树形规约

- 每轮把参与线程数减半
- 直到一个 block 内只剩一个结果

### 22.2 相邻规约

- 让固定间隔的线程两两相加
- 实现简单，但前几轮线程利用率不高

### 22.3 多级规约

- 先块内规约
- 再块间规约
- 大数据量场景非常常见

### 22.4 常见追问

- 为什么 shared memory 有用
  - 因为块内重复访问很快
- 为什么后半段可以用 warp shuffle
  - 因为 warp 内线程可以直接交换寄存器数据，减少同步

## 23. CUDA 面试高频口头题

- Nsight Systems 和 Nsight Compute 区别：
  - `nsys` 看整条时间线
  - `ncu` 看单个 kernel 微观指标
- 同步流和异步流：
  - 默认流上操作更偏串行
  - 多 stream 可让拷贝和计算重叠
- Tensor Core 为什么快：
  - 针对矩阵乘加专门设计，吞吐远高于普通 CUDA Core
- FP16 和 BF16：
  - FP16 尾数更多，精度更细
  - BF16 指数更多，范围更大，训练和推理更稳
