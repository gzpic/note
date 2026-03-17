# ONNX是什么

## 1. 一句话理解

ONNX（Open Neural Network Exchange）是一种深度学习模型的通用交换格式。

更工程一点地说：

> ONNX 是训练框架和推理框架之间的标准中间格式。

## 2. 为什么需要 ONNX

训练框架和推理框架往往不是同一套：

- 训练常见是 PyTorch、TensorFlow
- 推理常见是 TensorRT、ONNX Runtime、OpenVINO、MNN

如果没有中间格式，就会出现：

- PyTorch 模型不能直接喂给 TensorRT
- TensorFlow 模型不能直接喂给 OpenVINO

而 ONNX 的目标就是做“中间桥梁”。

典型流程：

```text
训练框架
  ↓ 导出
model.onnx
  ↓ 解析 / 优化 / 转换
推理框架
```

## 3. ONNX 文件里有什么

一个 `.onnx` 文件本质上包含：

- 计算图
- 权重
- 张量信息
- 算子版本信息

所以它不是单纯的“权重文件”，也不是单纯的“图文件”，而是两者结合。

## 4. ONNX 的作用

ONNX 主要解决这些问题：

- 训练框架和推理框架解耦
- 模型图结构标准化
- 便于做图优化、量化、算子融合和后端转换

常见链路可以写成：

```text
训练框架
  ↓ export
ONNX
  ↓
TensorRT / ONNX Runtime / MNN / OpenVINO / TVM
```

## 5. 为什么部署里经常先转 ONNX

原因通常有 3 个：

- 跨框架
- 图结构标准化
- 便于在 Graph 层做统一优化

例如常见的优化有：

- 算子融合
- 常量折叠
- layout 调整
- 量化前处理

所以在很多 CV 和通用模型部署链路里，ONNX 已经成了事实上的中间表示。

## 6. ONNX 在 LLM 里的位置要单独看

对于大模型，ONNX 不是不能用，但它并不是现在最主流的部署主线。

原因主要包括：

- 动态 shape 更复杂
- KV Cache 这类状态更难自然表达
- FlashAttention、Paged KV 这类高性能路径不容易直接映射成标准 ONNX 图

所以常见现象是：

- CV 模型和常规模型常走 ONNX
- LLM 更常走专用 runtime

例如不少 LLM 框架会直接读取：

- `safetensors`
- `gguf`

而不是把 ONNX 作为核心执行格式。

## 7. 工程上的一句话总结

> ONNX 是一个标准化的神经网络计算图文件格式，里面包含图结构、权重和必要的张量描述信息，用来衔接训练和推理。

## 8. 面试速记版

- ONNX 是模型交换格式
- 它的目标是解耦训练框架和推理框架
- `.onnx` 里通常包含计算图、权重和张量信息
- 在通用部署链路里 ONNX 很常见
- 但在 LLM 高性能推理里，很多框架会绕开 ONNX 走专用 runtime
