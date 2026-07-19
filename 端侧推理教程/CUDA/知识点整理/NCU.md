---

---
| 工具名 | 作用 | 场景 |
| --- | --- | --- |
| **Nsight Systems** (`nsys`) | 宏观分析：时间线、kernel 调度、CPU/GPU 协同 | 看“整体慢在哪里” |
| **Nsight Compute** (`ncu`) | 微观分析：warp、寄存器、共享内存、Tensor Core 利用率 | 看“内核为什么慢” |

ncu -o gemm_profile ./cutlass_gemm
ncu-ui gemm_profile.ncu-rep

bash
复制代码
ncu --kernel-name regex:your_kernel_name ./your_program
你将看到：

Tensor Core 占用率

Warp stall 原因（memory dependency / execution dependency）

SM occupancy

register pressure

shared memory bandwidth

dram throughput

