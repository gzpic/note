# ONNX的结构

## 1. 先看整体层次

ONNX 文件本质上是一个 protobuf 序列化的模型描述文件。

可以粗略理解成：

```text
Model
 └── Graph
      ├── Node
      ├── Tensor
      ├── Input / Output
      └── ValueInfo
```

## 2. 最外层：ModelProto

ONNX 文件最外层对象通常可以理解成 `ModelProto`。

常见信息包括：

- `ir_version`
- `opset_import`
- `producer_name`
- `graph`

这里面的重点是：

- 版本信息
- 算子集版本
- 真正的图结构入口

## 3. 核心部分：GraphProto

真正描述模型结构的核心对象是 `GraphProto`。

它里面最重要的内容有：

- `node`
- `initializer`
- `input`
- `output`
- `value_info`

## 4. Node：算子节点

每个算子是一个节点，例如：

- `MatMul`
- `Add`
- `Relu`
- `Softmax`

一个节点通常会描述：

- 输入张量名
- 输出张量名
- 算子类型
- 算子属性

## 5. Initializer：权重

权重、bias、embedding table 等常量张量通常放在 `initializer` 中。

本质上它们也是 Tensor，只不过是常量 Tensor。

## 6. Input / Output / ValueInfo

### 6.1 Input

描述模型输入，例如：

- 名字
- dtype
- shape

### 6.2 Output

描述模型输出。

### 6.3 ValueInfo

描述中间张量信息，尤其是 shape 和类型信息。

## 7. 一个简单抽象

可以把 ONNX 看成：

```text
ModelProto
 └── GraphProto
      ├── NodeProto      算子
      ├── TensorProto    权重 / 常量
      ├── Input          输入
      ├── Output         输出
      └── ValueInfo      中间张量信息
```

## 8. 面试速记版

- ONNX 最外层是模型对象
- 核心是 Graph
- Graph 里有节点、权重、输入输出和中间张量信息
- 权重在结构上通常表现为 constant tensor
