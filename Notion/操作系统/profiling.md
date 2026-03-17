---

---
### ② Memory 吞吐

- dram__throughput
- l2_throughput
- gld_efficiency
- gst_efficiency

👉 判断：

- DRAM 带宽 < 40% 峰值 → 访存型瓶颈
- 合并访问效率 < 80% → layout / 对齐有问题

### ③ Compute 利用率

- sm__throughput
- inst_fp16 / inst_tensor
- eligible_warps_per_cycle

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