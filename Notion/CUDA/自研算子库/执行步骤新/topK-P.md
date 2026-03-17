---

---
logits
→ sort
→ exp / normalize =softmax
→ cumulative sum
→ cutoff =累计概率 ≥ p 为止
→ sampling 在累计概率范围内随机生成一个数看落到哪

logits
→ argpartition / topk
→ mask
→ softmax
→ multinomial-按概率抽一个，好像和上面那种方法的也没啥区别确实是一样

top-k：
prefix sum 只用于 sampling

top-p：
prefix sum 用于
决定候选集合
sampling

为什么topk的softmax在后面，

1️⃣ softmax 不改变排序 → 用 logits 就能找 top-k
2️⃣ vocab 很大 → 全量 softmax 很贵
3️⃣ 只对 top-k 做 softmax 更快