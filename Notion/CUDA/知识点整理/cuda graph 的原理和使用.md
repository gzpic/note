---

---
**一、一句话定义**：

> CUDA Graph 是把一段 GPU 工作流（kernel launch、memcpy、memset、event 等）提前录制成一张有向无环图（DAG），之后反复一次性提交给 GPU 执行，避免每次 launch 的 CPU 开销。

传统方式是这样：

```plain text
CPU:
  launch kernel A
  launch kernel B
  launch kernel C
```

CUDA Graph 方式是：

```plain text
CPU:
  launch graph
GPU:
  A -> B -> C
```

👉 **核心目标**：减少 CPU→GPU 的 launch / driver 调用开销，提高吞吐。

二、

### Kernel launch 本身很“贵”

一次 kernel launch 包含：

- CPU → Driver
- 参数打包
- 同步/依赖处理
- 提交给 GPU

在 **小 kernel / 推理场景** 下：

- 真正算力占比很低
- **launch overhead 占比很高**

CUDA Graph 做的是：

> 把 N 次 launch 的 CPU 开销 → 1 次 graph launch

---

### 2️⃣ 提前固定执行拓扑

Graph capture 之后：

- kernel 执行顺序已知
- 依赖关系已知
- driver 可以提前做调度优化

⚠️ 但代价是：

👉 **拓扑 & shape 必须稳定**

三、简单的介绍

Graph 本质是一个 **DAG（有向无环图）**，节点类型包括：

| 节点类型 | 说明 |
| --- | --- |
| Kernel Node | kernel 执行 |
| Memcpy Node | H2D / D2H / D2D |
| Memset Node | cudaMemset |
| Event Node | event record / wait |
| Host Node | CPU 回调（较少用） |

节点之间通过 **依赖边** 连接。

四、基本使用方法

### ✅ 方式一：Stream Capture（最常用）

这是 **推理框架几乎全部采用的方式**。

### 使用流程

```c++
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 原来你怎么写，现在还怎么写
kernelA<<<...>>>(...);
kernelB<<<...>>>(...);
cudaMemcpyAsync(..., stream);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, ...);


```

之后运行：

```c++
cudaGraphLaunch(graphExec, stream);


```

### 特点

- **侵入性低**
- 代码改动小
- 易用于已有代码

⚠️ 限制：

- capture 期间不能有：
    - malloc/free
    - 某些 cuda API
    - 非确定行为

---

### ✅ 方式二：手动构建 Graph（较少用）

```c++
cudaGraphCreate(&graph,0);
cudaGraphAddKernelNode(...);
cudaGraphAddMemcpyNode(...);


```

特点：

- 完全可控
- 非常繁琐
- 通常只用于底层库

---

## 五、CUDA Graph 的关键限制（很重要）

### 1️⃣ **不支持动态 shape**

Graph 实例化后：

- kernel grid/block
- 指针地址
- 参数 layout
👉 **全部固定**

所以在推理服务中：

- batch size 经常变化
- sequence length 经常变化

📌 工程解法（你之前问过的）：

> 预 capture 多个 graph

```plain text
batch =1,2,4,8,16,32 ...


```

---

### 2️⃣ 参数地址不能变

❌ 错误示例：

```c++
float* p =newfloat;
kernel<<<>>>(p);


```

下次 p 地址变了 → graph 失效

✅ 正确方式：

- 预分配 buffer
- 运行时只改内容，不换地址

---

### 3️⃣ 不能随意插入同步

capture 期间：

- `cudaDeviceSynchronize()` ❌
- 一些 event 用法 ❌

### ❌ 不适合的场景

| 场景 | 原因 |
| --- | --- |
| 大 kernel | launch 开销占比低 |
| shape 高度动态 | capture 成本高 |
| 控制流复杂 | graph 表达能力有限 |

## 一、CUDA Graph 录制了哪些东西（核心）

一句话总结：

> CUDA Graph 录制的是：GPU 执行层面的“操作 + 拓扑 + 资源绑定”

### 1️⃣ Kernel launch 本身（最核心）

**会被录制的内容：**

- kernel 函数指针
- gridDim / blockDim
- shared memory 大小
- kernel 参数 **布局 + 地址**
- 所在 stream

📌 注意关键词：**参数地址**

```c++
kernel<<<grid, block, smem, stream>>>(ptr, n);


```

Graph 里记住的是：

- `ptr` 的 **设备地址**
- `n` 的 **值**
- launch 配置

⚠️ 所以：

- 指针地址不能变
- grid / block 不能变

---

### 2️⃣ 显存操作（Memcpy / Memset）

### cudaMemcpyAsync

录制内容包括：

- 拷贝类型（H2D / D2D / D2H）
- 源地址 / 目的地址
- 拷贝字节数
- stream

📌 结论：

- buffer 地址固定
- 拷贝大小固定

### cudaMemsetAsync

- 地址
- 填充值
- 字节数

---

### 3️⃣ 执行顺序 & 依赖关系（非常重要）

CUDA Graph 会记录：

- kernel A → kernel B 的顺序
- stream 之间的隐式同步
- event wait / record（可支持的一部分）

📌 这就是 Graph 的 **DAG 结构**。

---

### 4️⃣ Event（有限支持）

支持的：

- `cudaEventRecord`
- `cudaStreamWaitEvent`

Graph 会把 event 当成 **依赖节点**。

不支持的：

- capture 中动态创建 event
- 某些 timing event

---

### 5️⃣ Host Node（少用但存在）

- 录制 CPU 回调函数
- 执行在 GPU 调度流程中

⚠️ 实际推理框架里几乎不用（会破坏并行性）。

---

## 二、CUDA Graph 不会录制的东西（也很重要）

### 1️⃣ 不会录制「计算结果」

Graph 不关心：

- buffer 里是什么数据
- 上一轮算出来什么

👉 只要：

- 地址一样
- 大小一样
Graph 就可以复用。

---

### 2️⃣ 不会录制「控制流」

比如：

```c++
if (x >0) {
kernelA();
}else {
kernelB();
}


```

Graph 只能录到：

- **capture 当时走的那条路径**

下一次再想走另一条：

❌ 不可能

---

### 3️⃣ 不会录制动态资源管理

capture 期间 **禁止**：

- `cudaMalloc / cudaFree`
- new / delete（间接导致）
- 动态创建 stream / event

📌 原因：

Graph 要求资源在 **执行前已确定**。

---

### 4️⃣ 不会录制 shape 变化

例如：

- batch size
- sequence length
- hidden dim

这些一旦变化：

- kernel launch 参数变
- memcpy size 变
👉 Graph 必须重新录制。

---

## 三、用一张“锁死程度表”来理解

| 项目 | 是否被录制 | 是否允许变化 |
| --- | --- | --- |
| Kernel 函数 | ✅ | ❌ |
| Grid / Block | ✅ | ❌ |
| 参数地址 | ✅ | ❌ |
| 参数值（标量） | ✅ | ❌ |
| Memcpy size | ✅ | ❌ |
| Buffer 内容 | ❌ | ✅ |
| 控制流 | ❌ | ❌ |
| CPU 逻辑 | ❌ | ✅ |

---

## 四、一个推理场景的直观例子

### RMSNorm（简化）

```c++
cudaStreamBeginCapture(stream);
rmsnorm<<<grid, block>>>(x, w, y, hidden);
cudaStreamEndCapture(stream, &graph);


```

Graph 固定的是：

- `x / w / y` 的设备地址
- `hidden`
- grid / block

但**每一轮推理可以：**

- 往 `x` 里写不同输入
- 得到不同输出

---

## 五、为什么 batch size 一定要预分桶（你之前问过的）

当 batch size 变时：

| 变化 | 后果 |
| --- | --- |
| gridDim | ❌ |
| memcpy size | ❌ |
| kernel 参数 | ❌ |

👉 所以只能：

- batch=1 一个 graph
- batch=2 一个 graph
- batch=4 一个 graph …

---

## 六、一句话记忆法（工程视角）

> CUDA Graph 录制的是：
“执行什么 + 怎么执行 + 用哪些资源”，
不录制“数据本身”

---