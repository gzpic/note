# CUDA 与 GPU

## 高频题

### 1. CUDA 执行模型

- 组织层级是 `grid -> block -> thread`
- `warp` 是最小执行单位，通常是 32 个线程
- `block` 是资源分配单位，共享 shared memory 和同步原语
- 一个 `block` 不会跨 `SM`

### 2. CUDA 内存层次

- `register`
  - 最快，线程私有
- `shared memory`
  - 在 `SM` 内，block 共享
- `L1/L2 cache`
  - 缓存层
- `global memory`
  - 容量大，延迟高
- `local memory`
  - 名字叫 local，本质常还是显存

面试最稳的一句话：

- CUDA 优化主线通常先看访存，再看算力。

### 3. 什么是 coalesced access

- 一个 warp 内线程尽量访问连续且对齐的地址
- 这样硬件能把访存合并成更少 transaction
- 如果地址零散，带宽利用率会很差

### 4. warp divergence 是什么

- 同一 warp 内线程走不同分支时，会串行执行不同路径
- 本质是控制流分歧导致并行度下降
- 常见优化：
  - 让同一 warp 处理相似数据
  - 减少重分支

### 5. occupancy 是什么

- 一个 `SM` 上活跃 warp 数量占理论上限的比例
- 高 occupancy 不代表一定最快
- 但太低通常很难隐藏访存延迟
- 主要受下面几项影响：
  - 寄存器数
  - shared memory
  - block size

### 6. memory-bound 和 compute-bound 怎么区分

- `memory-bound`
  - DRAM/L2 压力大
  - 算力单元吃不满
- `compute-bound`
  - Tensor Core 或 FMA 利用率高
  - 计算单元更接近瓶颈
- 真正判断通常要：
  - 先用 `nsys` 找热点
  - 再用 `ncu` 看指标

### 7. shared memory 为什么快，为什么会有 bank conflict

- shared memory 在片上，延迟低
- 但被分成多个 bank
- 多线程同时打到同一个 bank 的不同地址时会串行化
- 常见优化：
  - padding
  - 改布局
  - 调整访问模式

### 8. CUDA Graph 和 Stream

- `stream`
  - 一条 CUDA 操作队列
  - 常用于 overlap copy 和 compute
- `CUDA Graph`
  - 把固定执行流程录成图，重复提交
  - 核心收益是减少 CPU launch overhead

### 9. `nsys` 和 `ncu` 区别

- `nsys`
  - 看整体 timeline
  - 看 CPU/GPU 协同
- `ncu`
  - 看单个 kernel 为什么慢
  - 看 occupancy、stall、L2、DRAM、Tensor Core

### 10. 手撕 CUDA 常见题

- reduction
- softmax
- layernorm / RMSNorm
- GEMM
- transpose

回答顺序一般是：

1. 先说并行划分
2. 再说访存
3. 再说同步和资源约束
4. 最后说 profiling 和迭代优化

## 高频口头结论

### Prefill 和 Decode 在 GPU 上怎么区分

- Prefill 更偏大矩阵计算，通常更像 compute-heavy
- Decode 每步计算量小，但更依赖 KV Cache 读取和带宽
- 所以 decode 更容易出现 GPU 利用率不高但延迟仍高

### 为什么很多小 kernel 会拖慢系统

- launch overhead 被放大
- kernel 间 gap 增多
- CPU 提交和同步成本占比上升

### CUDA 优化四层答法

1. 并行划分是否合理
2. 访存是否连续、对齐、可复用
3. 资源利用是否卡在寄存器或 shared memory
4. 工程上是否能做融合、graph、stream overlap

## 易错点

- 不要说 occupancy 越高越好
- 不要把 local memory 理解成“芯片上本地高速内存”
- 不要只靠理论判断一个 kernel 是 memory-bound 还是 compute-bound
- 不要把 `nsys` 和 `ncu` 的结论混着用

## 原笔记入口

- `[[Notion/面试整理/02-八股文/02-CUDA]]`
- `[[Notion/CUDA/CUDA]]`
- `[[Notion/CUDA/知识点整理/知识点整理]]`
- `[[Notion/CUDA/cuda优化重点/cuda优化重点]]`
- `[[Notion/CUDA/知识点整理/cuda graph 的原理和使用]]`
