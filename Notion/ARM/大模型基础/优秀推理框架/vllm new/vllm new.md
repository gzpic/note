---

---
1. 调度机制

6.decode自抢占

# 二、最常见的“只剩你一个人但你仍然 OOM”的 4 种真实来源

我给你按**真实推理系统**逐个拆：

---

## ✅ 1️⃣ 你自己历史 token 的 KV 占满了显存

你这个 seq 本身可能已经：

- prompt：4096 token
- 已经 decode：2000 token
- KV cache 总长度 = **6096 token**

如果模型是：

- 32 层
- FP16
- head=32
- hidden=4096

那你一个人占的 KV 可能就是：

```plain text
6096 × 32 × 2 × 4096 × 2B ≈ 12~20GB 级别


```

这时：

> ✅ running 里只有你
> ❌ 但你自己已经把显存吃干了
> 
> ❌ 现在你再想 append 一个 token 的 KV → 没地方放了

于是：

```python
can_append(seq) == False


```

出现。

---

## ✅ 2️⃣ 被你抢占的“别人”的 KV 还没回收完

你之前经历过这样的过程：

```python
self.preempt(self.running.pop())


```

理论上：

- preempt → 标记 WAITING
- 再异步回收 KV
- 再归还 block pool

但现实系统里：

- 回收可能是 **延迟的**
- CUDA kernel 可能还没结束
- block manager 还没真正 free

于是就会出现：

> ✅ running 里已经“没别人了”
> ❌ 但显存池里还有 **别人残留的 KV page**

---

## ✅ 3️⃣ Page 级 KV 管理带来的【碎片化死亡】

你现在的系统明显是：

```plain text
KV 不是一整块 malloc 的
而是按 page / block 分配的


```

典型问题是：

```plain text
你需要：1 个完整 page
但显存里只有：
[半个] [半个] [半个] [半个]


```

结果：

> ✅ 物理空闲显存是够的
> ❌ 但 **没有“连续可用的 page”**
> 
> ❌ `can_append(seq)` 依然返回 False

这就是：

> ✅ “显存还有，但 KV 还是分不到”

---

## ✅ 4️⃣ max_num_seqs + KV block 上限形成“逻辑死锁”

还有一种极容易被忽略：

你现在的逻辑是：

```python
while self.running and num_seqs < self.max_num_seqs:


```

也就是说：

- 本轮 decode 最多调度 `max_num_seqs` 个
- 但 block_manager 里可能：
    - 某些 block 已经 earmark 给下一轮
    - 但逻辑上尚未 free
    - 于是你现在 **逻辑看起来空了，物理还没空**

---

# 三、所以“只剩你一个人却没内存”的本质公式是：

> ✅ running 队列 = 逻辑存活
> ✅ KV Block Pool = 物理真实占用
> 
> ❗**这两者是“弱一致”的，而不是“强一致”的**

所以就会自然出现你觉得违反直觉的状态：

```plain text
running.size == 1
但
block_manager.free_blocks == 0


```

这不是 bug，而是：

> ✅ Paged KV + 抢占式调度 的必然中间态

---

# 四、这时为什么“只能抢自己”？这一步的真实语义是：

```python
self.preempt(seq)


```

等价于：

> ❗“我没资格再继续 decode 了，我必须把自己占用的 KV 也全部释放掉，否则整个系统永远无法前进”

这一步本质不是“自杀”，而是：

| 行为 | 实际含义 |
| --- | --- |
| preempt(self) | 释放你全部 KV |
| status = WAITING | 你退回等待队列 |
| 等待下一次 prefill | 后面重新进 |

这是一个：

> ✅ 以“牺牲当前请求”换取“系统不死锁” 的工程兜底策略

---

# ✅ 最终一句终极总结（你可以直接背下来）

> “只剩我一个人 ≠ 显存只属于我一个人”
“running 是逻辑队列，KV 是物理资源池”
“当两者不同步时，自抢占就是唯一的破局手段”

---

如果你愿意，我可以下一步帮你做其中任一件：

[[调度机制]]

[[分块与填充的]]