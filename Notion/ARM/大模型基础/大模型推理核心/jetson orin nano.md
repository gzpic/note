---

---
硬件瓶颈点

为啥什么prefill是动态的而decode是静态能用cuda graph

prefill能复用kvcache？

launch 一个kernel做哪些事情

录制一个cuda graph录制一些啥

jetson专有优化的方法

延迟隐藏，异步io，预取，overlap

wmma和mma的区别是什么

比较浮点大小

transformer的前缀特性与mask，如不mask？

如果我要给qwen3适配端侧的算子需要适配哪些算子

行主序列主序

padding or mask     可以占位用区别

<<<>>>的作用

残差连接作用

数据布局比较NW

transformer各层的作用


一个问题是，经过预先的填充，当前graph里只能有唯一一个动态的维度，且它的值必须是batch size，这也意味着，子图里一些诸如Concat，Gather，Split等可能会导致破坏这一条件的操作应当要被谨慎的排除出去。

1. **指数 / 倍数递增** = 减少 capture 图数量，覆盖大部分 batch
2. **padding 到下一个 capture 倍数** = 避免动态 shape，保证 Graph replay 成功
3. **对 GPU launch 友好** = 保持 SM occupancy / warp efficiency

> 简而言之：
> **这是显存、Graph 数量和性能的折中方案**，工业界的“黄金实践”。

### mask，防作弊。

decoder-only模型是自回归模型根据前面的词预测后面的词，如果前面的词直接看到了后面的，不就作弊了。

| 架构 | **前面的词能看到后面的词吗？** | 说明 |
| --- | --- | --- |
| **Encoder-only（BERT 类）** | ✅ **可以** | 全双向 self-attention |
| **Encoder–Decoder（T5 / Transformer）** | Encoder：✅ 可以Decoder：❌ 不可以 | 编码理解 vs 自回归生成 |
| **Decoder-only（Qwen / LLaMA）** | ❌ **不可以** | 严格自回归 |

自回归 = 用“已经生成的自己”，一步一步生成后面的自己。

Encoder 之所以可以、而且应该双向，是因为它的目标不是“生成”，而是“理解一个已经完整存在的序列”。

所以不一样

## 训练时会出现什么“必然现象”？

### 🚨 现象 1：loss 异常低（虚假的好成绩）

举个最直观的例子：

```plain text
我 爱 吃 苹果


```

训练时预测“吃”这个 token：

- 如果能看到后面的 “苹果”
- 模型会学到一个规则：
> “只要后面是苹果，前面大概率是‘吃’”

于是：

- loss 快速下降
- perplexity 看起来非常漂亮
- 但这是 **信息泄漏**

👉 你在看的是 **作弊后的分数**。

---

### 🚨 现象 2：attention 权重高度集中到未来 token

实际观察中会出现：

- attention map 出现大量：
```plain text
t → t+1, t+2


```
- 而不是：
```plain text
t → t-3, t-10


```

模型会本能地选择：

> 最近、最确定、信息量最大的 token

未来 token 就是“标准答案”。

---

## 3️⃣ 那推理时会发生什么？（最致命的部分）

### 推理现实是这样的：

- 你在生成第 `t` 个 token
- **未来 token 根本不存在**

但模型在训练时已经形成了依赖：

```plain text
x_t  ←  x_{t+1}


```

于是推理时：

- 这个依赖突然被砍掉
- attention 分布崩溃
- FFN 激活分布漂移
- Layer

对dynamic recast会转成啥类型

const recast转后的变量能修改吗

tops的定义

![[IMG_0752.png]]

| 修饰符 | 意义 | 调用者 | 返回值/执行 | 并行粒度 |
| --- | --- | --- | --- | --- |
| `__global__` | GPU kernel（全局函数） | **只能从 host（CPU）调用** | 没有返回值（`void`），异步执行 | grid/block/warp 级别并行 |
| `__device__` | GPU 设备函数（只能在 GPU 上执行） | **只能从 device（GPU kernel）或其他 **`**__device__**`** 调用** | 可以返回值，像普通函数 | thread 级别并行（每个 thread 调用自己的实例） |

引用传递的时候一个引用是多大的呢，和指针一样大，和平台有关


arm64和x86 gpu默认的端序 小端序 

网络字节流 tcp/ip大端序的

为什么一单例里单例对象指针必静态

ptx和sass区别是什么呢

sass指令

transformer本身及各模块复杂度


工业级别的算子实现方法就像几核切

dense和moe区别是啥

找到自己特长gpu kernel！

函数指针

表驱动

状态机有什么用

回掉函数作用

写矩阵换维度

qwem6b这样的模型怎么来的

dma会不会经由cache呢

全局内存到共享内存还有寄存器拷贝

多warps竞争一个sm延迟隐藏

xxx利用率是不是越大越好呢为什呢这样

有cublas还需要手写？和cutlass的区别是什么呢？

为什么sampling后面交给cpu

jetson上prefill和decode哪个更瓶颈

深拷贝浅拷贝

定义和声明

多请求为什么能复用kvcache

unroll是不是越多越好

cuda graph的优化细节可以查看一下上次的那篇知乎推文挺好可以参考有非常大借鉴的

cuda memco


```plain text
python llmexport.py \
--path /home/chen/coding/llm/mnn/MNN/transformers/llm/export/models/Qwen2-0.5B-Instruct\
--export mnn --hqq
```

类的析构顺序

| **椤圭洰** | **鑳戒笉鑳芥槸 virtual** | **鑳戒笉鑳借皟鐢ㄨ櫄鍑芥暟** | **鏄惁鍙戠敓澶氭€** | **鍏稿瀷缁撹** |
| --- | --- | --- | --- | --- |
| 鏋勯€犲嚱鏁 | 鉂 涓嶅厑璁 | 鉁 鍙互 | 鉂 涓嶅彂鐢 | 鍙細璋冪敤褰撳墠鏋勯€犲眰绾鐨勭増鏈 |
| 鏋愭瀯鍑芥暟 | 鉁 鍏佽 | 鉁 鍙互 | 鉂 鏋愭瀯杩囩▼涓笉鍙戠敓 | 蹇呴』璁句负 virtual 鎵嶅畨鍏 delete |
| 鏅€氭垚鍛樺嚱鏁 | 鉁 | 鉁 | 鉁 鍙戠敓 | 姝ｅ父澶氭€ |
| 鏋勯€犲嚱鏁颁腑璋冪敤铏氬嚱鏁 | 鈥 | 鉁 | 鉂 | 绛夊悓闈 virtual |
| 鏋愭瀯鍑芥暟涓皟鐢ㄨ櫄鍑芥暟 | 鈥 | 鉁 | 鉂 | 绛夊悓闈 virtual |
