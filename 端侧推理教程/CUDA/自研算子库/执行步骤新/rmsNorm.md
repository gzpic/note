---

---

### 1.计算的方法呢

![[image 11.png]]

---

![[image 12.png]]

![[image 13.png]]

1. memory bound OR compute bound ?
memory bound

2. 优化的重点：Kernel 融合 > 半精度 > Warp 级归约 > 向量化 > CUDA Graph / TRT>register tilling /smem

3. 优化具体：

半精度

warp规约：没啥好说的…

向量化加载：

本质上：规约(block reduceSum + warp reduceSum)+\
