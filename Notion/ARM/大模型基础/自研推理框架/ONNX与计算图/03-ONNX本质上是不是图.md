# ONNX本质上是不是图

## 1. 结论

是的，ONNX 的核心就是一张计算图。

但更准确地说：

> ONNX 不是“只有图”，而是“图结构 + 数据 + 元信息”的组合。

## 2. 图的基本形态

ONNX 的算子结构通常可以理解成 DAG：

- Directed
- Acyclic
- Graph

也就是有向无环图。

在这个图里：

- 节点是算子
- 边是张量

数据流形式通常是：

```text
Tensor → Node → Tensor → Node
```

## 3. 举一个简单例子

如果模型是：

```text
y = relu(x @ W + b)
```

图结构可以理解成：

```text
      W
      ↓
X → MatMul → A → Add → B → Relu → Y
                ↑
                b
```

这里：

- `MatMul / Add / Relu` 是节点
- `X / A / B / Y / W / b` 都是张量

## 4. ONNX 为什么不能只说成“一个图”

因为 `.onnx` 文件除了图关系，还会携带：

- 权重数据
- shape 信息
- dtype 信息
- opset 版本

所以更完整的表述应该是：

> ONNX 的核心抽象是一张计算图，但文件本身还包含图运行所需的权重和描述信息。

## 5. 面试速记版

- ONNX 的核心是 DAG 计算图
- 节点是算子，边是 Tensor
- 但 `.onnx` 文件不只有图，还有权重、shape、dtype 等元信息
