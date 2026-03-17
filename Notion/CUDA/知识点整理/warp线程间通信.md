---

---
1. Shuffle

它允许线程从同一个 Warp 中的另一个线程读取寄存器值。

__shfl_sync(mask, val, src_lane);指定一个源线程 ID，其他线程都去读这个线程的数据（广播）。
__shfl_up_sync(mask, val, delta);相对于当前线程 ID 向“上”偏移。常用于前缀和（Prefix Sum）计算。
__shfl_down_sync(mask, val, delta);相对于当前线程 ID 向“下”偏移。常用于归约（Reduction）计算。
__shfl_xor_sync(mask, val, laneMask)
于按位异或（XOR）掩码交换数据。这是实现 **蝶式网络（Butterfly Network）** 归约的核心，非常高效。

shfl.sync.idx
shfl.sync.up
shfl.sync.down
shfl.sync.bfly   // xor



2. **Vote  (**`**__any_sync**`**, **`**__all_sync**`**, **`**__ballot_sync**`**)**
这类指令不是用来交换具体的数值，而是用来交换**逻辑状态**（谓词）。
• `**__any_sync(mask, predicate)**`：只要 Warp 中有一个线程的条件为真，就返回真。
• `**__all_sync(mask, predicate)**`：必须所有线程条件都为真，才返回真。
• `**__ballot_sync(mask, predicate)**`：这是最强大的投票指令。它返回一个 `int32`，每一位（bit）对应一个线程。如果线程 $i$ 的条件为真，则结果的第 $i$ 位就是 1。
    ◦ *用途：* 快速统计 Warp 中满足条件的线程数量或寻找第一个满足条件的线程。

## 3. Match (`__match_any_sync`, `__match_all_sync`)

这是在 Volta (Compute Capability 7.0) 及之后引入的新指令，专门用于解决**分组问题**。

- 它能让 Warp 内具有相同值的线程自动找寻彼此。
- 例如，如果有 32 个线程在处理不同的 Key，`__match_any_sync` 可以返回一个掩码，告诉你哪些线程持有和你一样的 Key。这在实现高效的哈希表或基数排序时非常有用。

## 4. Activemask  (`__activemask()`)

返回当前 Warp 中所有活跃（且未分支退出）线程的掩码。这通常作为其他 `_sync` 指令的第一个参数使用，以确保在分支路径中也能正确同步。