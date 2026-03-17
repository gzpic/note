# 从 Hugging Face 模型导出 ONNX

## 1. 主流方法有哪些

从 Hugging Face 下载的模型导出成 ONNX，常见有 4 类方法：

1. `Optimum`
2. `transformers.onnx`
3. `torch.onnx.export`
4. `tf2onnx`

如果按实际推荐顺序，通常可以这样选：

1. 先试 `Optimum`
2. 不行再试 `transformers.onnx`
3. 还不行就走 `torch.onnx.export`
4. 如果模型本来是 TensorFlow / Keras，就用 `tf2onnx`

## 2. 方法一：Optimum

这是现在比较推荐、也最省事的一种方式。

优点：

- 和 Hugging Face 生态贴合得更好
- 命令行和 Python 两种方式都方便
- 对常见 Transformers 模型支持较友好

命令行形式通常类似：

```bash
optimum-cli export onnx --model bert-base-uncased onnx_out/
```

适合：

- 常见 Transformer 模型
- 想少踩坑
- 想快速拿到可供 ONNX Runtime / 后端使用的模型

## 3. 方法二：transformers.onnx

这是 Transformers 自带的老牌导出方式。

形式通常类似：

```bash
python -m transformers.onnx --model=bert-base-cased onnx/bert-base-cased/
```

优点：

- 使用简单
- 对标准任务模型比较直接

缺点：

- 对新模型和复杂模型的支持有时不如 Optimum 顺手

## 4. 方法三：torch.onnx.export

这是最通用、也最灵活的方式。

适合：

- 你想完全控制输入输出
- 你需要自己指定 dynamic axes
- 前两种方法失败时兜底

它的代价是：

- 你要自己处理输入签名
- 自己处理动态维
- 自己处理一些兼容问题

特别是大模型里：

- `past_key_values`
- 动态 KV cache
- 多输出结构

都会让手动导出更复杂。

## 5. 方法四：tf2onnx

如果 Hugging Face 下来的模型本质上是 TensorFlow / Keras 格式，就应该走 `tf2onnx`。

这条路不适合 PyTorch checkpoint，但对 TF 模型是标准方案。

## 6. Mac 平台能不能做

可以。

在 Mac 上导出 ONNX 和 Linux 基本是同一套 Python 工具链，导出本身主要依赖：

- Python
- Transformers / Optimum / PyTorch
- ONNX 相关包

GPU 不参与导出，所以即使是 Mac，也完全可以做。

## 7. Mac 平台的注意点

### 7.1 GPU 不是关键

导出 ONNX 本质上主要是：

- tracing
- graph export
- graph serialization

所以不依赖 CUDA。

### 7.2 Python 版本要稳

一般更建议：

- Python 3.10
- Python 3.11

因为很多工具链对较新的 Python 版本支持可能没有那么稳定。

### 7.3 opset 版本别太老

通常更推荐较新的 opset，例如：

- `opset 17`

这样和后续推理框架兼容时通常更稳一些。

## 8. 为什么有些 Hugging Face 模型导不出来

不是所有模型都能一键成功导出 ONNX。

常见原因包括：

- 模型结构太新
- 导出器暂时不支持某些模块
- `past_key_values` 结构复杂
- 动态 shape 配置不完整
- 自定义算子 / 自定义层
- opset 不匹配

所以“能不能导出”不只取决于模型来源是不是 Hugging Face，还取决于：

- 模型类型
- 导出方法
- 目标任务

## 9. LLM 场景要更谨慎

对于大模型，ONNX 并不是总是最舒服的中间格式。

尤其是：

- `Llama`
- `Qwen`
- `Mixtral`

这类模型里经常涉及：

- `past_key_values`
- 动态 KV cache
- prefill / decode 路径差异

这些都会让 ONNX 导出和后续执行复杂很多。

所以常见现象是：

- 常规模型更容易走 ONNX
- LLM 更容易转向专用 runtime

## 10. 对部署链路的一个实用理解

如果目标是 Jetson 或后续 TensorRT 部署，常见链路可以理解成：

```text
Hugging Face model
  ↓
PyTorch / Transformers
  ↓ export
ONNX
  ↓
TensorRT / ONNX Runtime / MNN
```

但对于 LLM，要提前意识到：

- 不一定所有模型都适合这条路
- 有时需要拆 prefill / decode
- 有时需要绕开 ONNX，直接走专用推理框架

## 11. 面试速记版

- Hugging Face 导 ONNX 常见有 4 条路：`Optimum`、`transformers.onnx`、`torch.onnx.export`、`tf2onnx`
- 默认先试 `Optimum`
- Mac 平台可以导 ONNX，导出本身不依赖 GPU
- LLM 导出 ONNX 更容易被 `past_key_values` 和动态 shape 卡住
