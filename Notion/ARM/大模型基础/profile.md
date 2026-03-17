---

---
1.指标体系构建

##  kernel 做微观 profiling（Nsight Compute）

### ① Occupancy / 活跃 warp

- **achieved_occupancy   有**
- **active_warps_per_sm **
- **theoretical_warps_per_sm   **

👉 判断：

- < 25%：几乎一定有大问题
- 25–50%：可能是寄存器 / block 太大
- 60%：occupancy 不是主因

---

### ② Memory 吞吐

- **dram__throughput**
- **l2_throughput**
- **gld_efficiency**
- gst_efficiency

👉 判断：

- DRAM 带宽 < 40% 峰值 → 访存型瓶颈
- 合并访问效率 < 80% → layout / 对齐有问题

---

### ③ Compute 利用率

- **sm__throughput **
- inst_fp16 / inst_tensor
- **eligible_warps_per_cycle**

👉 判断：

- 算力 < 40–50% 峰值 → 没喂饱 or 并行度不够
- eligible_warps 很低 → 依赖 / stall 多

---

### ④ Stall 原因

- stall_memory_dependency
- stall_long_scoreboard
- stall_not_selected
- stall_exec_dependency

👉 判断：

- memory_dependency 高 → latency hiding 不够
- not_selected 高 → warp 太多但调度器选不上

---

### ⑤ 指令结构

- ld/st 指令比例
- fma / mma 指令比例
- control 指令比例

👉 判断：

- load 太多 → tile 太小 / reuse 不够
- control 太多 → 分支 or loop 结构差

ncu咋得到


sudo /usr/local/cuda-12.6/bin/ncu \
--target-processes all \
--kernel-name-base function \
--kernel-name "hgemm_t_8x8_sliced_k_f16x4_pack_kernel" \
--section SpeedOfLight \
--section MemoryWorkloadAnalysis \
--section Occupancy \
--section SchedulerStats \
--section InstructionStats \
--export gemm_key.ncu-rep --force-overwrite \
./gemm_basic

/usr/local/cuda-12.6/bin/ncu \
--import gemm_key.ncu-rep \
--print-summary per-kernel

/usr/local/cuda-12.6/bin/ncu \
--import gemm_cublas.ncu-rep \
--print-summary per-kernel

## `gld_efficiency` —— **load 是否合并（coalescing）**

### 是什么

- 实际传输字节 / 理论最少字节
- 本质是：**一个 warp 的 load 有没有合并成少量 memory transaction**

### 经验判断

- **> 90%**
    - ✔️ 访问模式正确
- **70–90%**
    - ⚠️ 可能有边界 / stride 问题
- **< 70%**
    - ❌ 明确 layout / index 问题

### 3️⃣ `eligible_warps_per_cycle`

**这是最重要的一个**

**是什么**

- 每个周期中：
> “当前可被调度、无依赖、无 stall 的 warp 数”

**关键理解**

- GPU 不缺算力
- GPU 缺的是：**现在能跑的 warp**

**经验区间**

- **< 1**
    - ❌ 严重依赖 / stall
    - ❌ latency hiding 失败
- **1–2**
    - ⚠️ 勉强
- **≥ 2–4**
    - ✔️ 调度器有选择空间

> 💡 真正的 latency hiding =
> **eligible warps × 单 warp **

## 你原来那 9 个指标 ↔ 现在 summary 里对应哪一行

1. **sm__throughput**
→ `GPU Speed Of Light Throughput` 里的 `**Compute (SM) Throughput %**`（你这：**86.81%**）
2. **inst_fp16 / inst_tensor**
→ 你这份 summary **没有分 FP16/Tensor 管线**（只给了总 `Executed/Issued Instructions`）。
要看到 FP16/Tensor：**需要换 section（不是 Instruction Statistics 这个总表）**。
3. **eligible_warps_per_cycle**
→ 没有直接给 “per cycle”，但等价看：
- `Scheduler Statistics` 的 `**Issued Warp Per Scheduler**`（≈ 每周期发射强度，0~1）
- `Eligible Warps Per Scheduler`（你这 **3.03 warps**，越多越好）
如果你坚持要“per cycle”那种名字：也需要换 section/metrics。
4. **dram__throughput**
→ summary 里没叫 dram__，对应看：
- `GPU Speed Of Light Throughput` 的 `**Memory Throughput %**`（你这：**69.62%**）
- 以及 `Memory Workload Analysis` 的 `**Max Bandwidth %**`（你这：**43.12%**，更像“占峰值带宽比例”）
5. **l2_throughput**
→ `GPU Speed Of Light Throughput` 的 `**L2 Cache Throughput %**`（你这：**14.52%**）
6. **gld_efficiency**
→ 这份 summary **没有直接给**（Global Load Efficiency 不在这几行里）。
你能间接判断的是：`L1/TEX Hit Rate` 很低、`L2 Hit Rate` 很高，但这不是效率本身。
7. **achieved_occupancy**
→ `Occupancy` 的 `**Achieved Occupancy %**`（你这：**63.70%**）
8. **active_warps_per_sm**
→ `Occupancy` 的 `**Achieved Active Warps Per SM**`（你这：**30.58**）
9. **theoretical_warps_per_sm**
→ `Occupancy` 的 `**Theoretical Active Warps per SM**`（你这：**32**）
