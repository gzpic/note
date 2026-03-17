---

---

# network-MNIST-model

network可自己构建，也可以用parse从onnx格式文件直接加载的

## 自己构建network的方法

### step1 用builder构建实例 network

a.

```javascript
IBuilder* builder = createInferBuilder(logger);
INetworkDefinition* network = builder->createNetworkV2(flag);
```

b.

```plain text
auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(0);
```

### step2 添加input

```javascript
auto data = network->addInput(INPUT_BLOB_NAME, datatype, Dims4{1, 1, INPUT_H, INPUT_W});
```

### step3 构建网络层级

```javascript
auto conv1 = network->addConvolution(
*data->getOutput(0), 20, DimsHW{5, 5}, weightMap["conv1filter"], weightMap["conv1bias"]);
conv1->setStride(DimsHW{1, 1});

auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
pool1->setStride(DimsHW{2, 2});

int32_t const batch = input->getDimensions().d[0];
int32_t const mmInputs = input.getDimensions().d[1] * input.getDimensions().d[2] * input.getDimensions().d[3];
auto inputReshape = network->addShuffle(*input);
inputReshape->setReshapeDimensions(Dims{2, {batch, mmInputs}});

IConstantLayer* filterConst = network->addConstant(Dims{2, {nbOutputs, mmInputs}}, mWeightMap["ip1filter"]);
auto mm = network->addMatrixMultiply(*inputReshape->getOutput(0), MatrixOperation::kNONE, *filterConst->getOutput(0), MatrixOperation::kTRANSPOSE);

auto biasConst = network->addConstant(Dims{2, {1, nbOutputs}}, mWeightMap["ip1bias"]);
auto biasAdd = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
auto relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);

auto prob = network->addSoftMax(*relu1->getOutput(0));
```

### 
step4 设置输出

```javascript
prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
network->markOutput(*prob->getOutput(0));
```

## 从onnx格式文件parse

step1 构建好parse

```javascript
#include “NvOnnxParser.h”
using namespace nvonnxparser;
IParser* parser = createParser(*network, logger);
```


你手动把 **ONNX 模型的 protobuf 字节流**（不一定来自文件）传给 parser。


```javascript
parser->loadModelProto(modelData, modelSize);
```

给 parser 补充权重（initializer）。

为什么 initializer 是分开加载？

- static weights（比如 Conv weight, bias）在 ONNX Graph 的 initializer 字段中
- 你可以用这个 API **覆盖默认的权重** 或 **延迟加载权重**

很多大厂框架（TRT-LLM / DeepSpeed / TensorRT Wizard）拿来做 **外置权重加载**。

```javascript
parser->loadInitializer(name, data, dataSize);
```

**真正把模型转换为 TensorRT 网络**。

它会：

- 解析模型结构
- 把算子转成 TensorRT 对应的 layer（比如 Conv、MatMul、Relu）
- 把 initializer 填入权重
- 在你传入的 `network` 中添加所有层

```javascript
parser->parseModelProto()
```

ENGINE

用上面的network可以构建engine

step1.先设置config

```javascript
auto engine = builder->buildEngineWithConfig(*network, *config);
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
```

step2.用network来进行构建engine

```javascript
IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
```

step3 删除非必要配置

```javascript
delete parser;
delete network;
delete config;
delete builder;
//如果engine持久化储存，那也可以被释放调
delete serializedModel
```

EXECUTE PLAN

step1 

```javascript
IRuntime* runtime = createInferRuntime(logger);
```

step2 两种反序列化engine的方法

a.

```javascript
std::vector<char> modelData = readModelFromFile("model.plan");
ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
```

b.

```javascript
class MyStreamReaderV2 : public IStreamReaderV2 {
    // Custom implementation with support for device memory reading
};
MyStreamReaderV2 readerV2("model.plan");
ICudaEngine* engine = runtime->deserializeCudaEngine(readerV2);
```

step 3

```javascript
IExecutionContext *context = engine->createExecutionContext();
```

step4

```javascript
context->setTensorAddress(INPUT_NAME, inputBuffer);
context->setTensorAddress(OUTPUT_NAME, outputBuffer);
context->setInputShape(INPUT_NAME, inputDims);
```

step5

```javascript
context->enqueueV3(stream);
```

[[TRT特点实现]]