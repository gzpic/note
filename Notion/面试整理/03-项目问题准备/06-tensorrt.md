可以，我就按你这段里提到的“问题/能力点”一条一条讲，对应**实际处理方案**、**面试里该怎么说**、**做到什么程度算招聘里的熟练**。TensorRT 官方现在也还是把核心能力放在 Quick Start、dynamic shapes、best practices、troubleshooting、custom layers 这些模块里；当前文档首页明确把快速部署、动态 shape、量化、troubleshooting、plugin 扩展列成主线能力。

---

## 1）“会安装和验证 TensorRT”对应怎么处理

### 你要解决的问题

不是“装上包就完了”，而是要确认：

1. TensorRT 版本对 CUDA / driver / GPU 是否匹配
2. Python/C++ API 能不能正常调用
3. `trtexec` 能不能跑
4. ONNX parser、plugin 库这些是否齐全

### 处理方案

先做三层验证：

**第一层：环境验证**

- 看 TensorRT 版本、CUDA 版本、GPU 型号
- 确认 `trtexec` 是否存在
- 确认 Python 里 `import tensorrt as trt` 正常

**第二层：最小样例验证**

- 直接拿一个官方能过的 ONNX 小模型
- 用 `trtexec --onnx=... --saveEngine=...` 先生成 engine
- 再跑一次 engine 推理

**第三层：排兼容性**

- 如果 build 失败，先查 driver/CUDA/TRT 版本组合
- 如果 parser 报错，先区分是 ONNX 导出有问题，还是 TensorRT 不支持该 op
- 如果 runtime 报错，再看 engine 是否和当前机器绑定不一致

### 面试里怎么说

> 我会先做环境闭环验证，不是只看安装成功。一般会验证 Python/C++ API、`trtexec`、ONNX parser 和最小 ONNX→engine→runtime 推理链路，先确保问题不是环境层的。

### 到这一步算什么水平

这是“会用 TensorRT”的起点，不算“真熟练”，但这是必备基本功。Quick Start 里也明确把 `trtexec` 转换和 runtime 部署作为第一条主线。

---

## 2）“会把 ONNX 转成 engine”对应怎么处理

### 你要解决的问题

给你一个模型，你要把它变成可部署的 TensorRT engine。

### 处理方案

标准顺序是：

**第一步：先保证 ONNX 本身干净**

- 用 ONNXRuntime 先跑通，确认模型导出没问题
- 检查输入输出名、shape、dtype
- 看有没有奇怪的自定义节点、动态维度、shape op

**第二步：优先用 `trtexec` 跑首版**

- 因为它最快，最适合先拿到 baseline
- 先生成 engine，再 benchmark
- 如果 `trtexec` 都过不了，就先别急着写 API 代码

**第三步：再决定要不要写 builder/runtime 代码**

- 只做离线转换和简单部署：`trtexec` 足够
- 要嵌入自己业务系统、自己管理 buffer / stream / profile：再上 C++/Python API

### 面试里怎么说

> 我一般先走 ONNX 路径，因为这是 TensorRT 最通用的自动转换链路。先用 `trtexec` 做首版 engine 和 baseline，再看要不要接 builder/runtime API 进业务代码。官方 Quick Start 也是把 ONNX→`trtexec`→runtime 作为最常见路径。

### 到这一步算什么水平

这还是“熟悉”的前半段，不足以单独撑起“熟练”。

---

## 3）“什么时候该用 trtexec，什么时候该自己写 builder/runtime 代码”对应怎么处理

### `trtexec` 适合什么

适合：

- 快速验证 ONNX 是否能转
- 快速生成 engine
- 做 benchmark
- 看初步 profiling
- 快速验证 FP16/INT8/动态 shape 配置

官方 best practices 也明确把 `trtexec` 作为性能 benchmark 和 profiling 的基础工具。

### API 适合什么

适合：

- 业务里要自己控制 build 过程
- 需要多 profile、多个 stream、复杂内存管理
- 要自定义 logger / allocator / plugin 注册
- 要在程序里动态加载 engine、设置 tensor address、控制 context
- 要把推理集成进完整服务/端侧程序

### 处理原则

**先 `trtexec`，后 API。**  
因为：

- `trtexec` 先帮你证明模型本身可转、性能大概如何
- 之后再写 API，问题范围更小

### 面试里怎么说

> `trtexec` 更像首选诊断工具和 benchmark 工具；API 更像正式交付路径。前者用来快速试错，后者用来和业务系统结合。

这句话面试里很好用。

---

## 4）“builder 和 runtime 的基本角色”对应怎么处理

### 这是在问什么

很多人只会“转 engine”，但讲不清：

- builder 干什么
- runtime 干什么
- engine/context 是什么关系

### 处理方案

你要这样理解：

**builder 阶段**

- 读取网络
- 选择 tactic
- 做 layer fusion、precision 选择、profile 约束
- 最后生成 engine

**runtime 阶段**

- 反序列化 engine
- 创建 execution context
- 绑定输入输出地址
- enqueue 执行推理

也就是：

**builder = 编译/优化阶段**  
**runtime = 执行阶段**

Quick Start 和 migration guide 都在强调 engine 构建与 runtime 部署是两个阶段；10.x 里 API 也更强调 name-based IO 和 runtime 执行流程。

### 面试里怎么说

> Builder 负责把网络编译成适合目标硬件的 engine，runtime 负责加载 engine 并执行。调优大多发生在 builder 阶段，部署集成大多发生在 runtime 阶段。

这就是标准答法。

---

## 5）“能处理常见 FP16 部署”对应怎么处理

### 你要解决的问题

不是简单加个 `--fp16`，而是：

- 模型能不能安全降到 FP16
- 性能是不是变好了
- 精度有没有异常

### 处理方案

**第一步：先做 FP32 baseline**

- 先在 ONNXRuntime 或 TensorRT FP32 下跑基准
- 保存输出结果用于对比

**第二步：再切 FP16**

- 重新 build engine
- 重新测 latency / throughput
- 对比输出偏差

**第三步：如果性能没提升或精度掉点**  
排查顺序：

1. 某些层没真正走 FP16
2. 动态 shape 导致某些 fusion/tactic 不理想
3. 输入数据范围本来就对 FP16 敏感
4. batch/profile 设得不合理

TensorRT 文档明确支持 mixed precision，并把准确性与性能调优分开讲；best practices 也建议 benchmark 与 profiling 联动看。

### 面试里怎么说

> FP16 不是只看能不能跑，还要看精度和性能是否真的达标。我一般会先留一份 FP32 baseline，再切 FP16 做 A/B 对比。

这就显得你不是只会开开关。

---

## 6）“unsupported op”对应怎么处理

### 你要解决的问题

TensorRT 不能保证支持你模型里的所有算子。Quick Start 也明确说了：ONNX 导入要求算子受支持，不支持的要靠 plugin 补。

### 处理方案

遇到 unsupported op，不要上来就写 plugin，顺序应该是：

**第一层：确认是不是导出问题**

- 有时候不是 TensorRT 不支持，而是 ONNX 导出成了奇怪子图
- 先看 ONNX 节点名和 op 类型是否正常

**第二层：能不能图改写绕过去**

- 用 ONNX simplifier / GraphSurgeon / 手动改图
- 把某个复杂 op 拆成 TensorRT 支持的基础 op 组合
- 或者在 PyTorch 导出时改写前向

**第三层：再考虑 plugin**  
适合写 plugin 的情况：

- 这个 op 是核心瓶颈
- 图改写太丑或不可维护
- 业务长期要用
- 多个模型都会碰到

### 面试里怎么说

> 遇到 unsupported op，我会先判断是导出问题、图可改写问题，还是确实需要自定义层。plugin 是后手，不是第一手。

这句话很加分。

---

## 7）“shape mismatch”对应怎么处理

### 你要解决的问题

这个特别常见，很多乱码、崩溃、输出异常都可能从这里来。

### 处理方案

排查顺序很固定：

**先核对模型输入**

- 输入 name
- 输入 dtype
- 输入 layout（NCHW / NHWC）
- 输入 shape

**再核对运行时实际喂进去的数据**

- 真正送进 context 的 shape 是多少
- batch 是否和 profile 范围匹配
- 动态维是否先 set shape 再 enqueue

**再核对导出链路**

- PyTorch 导出时 dynamic axes 是否对
- ONNX 上的 shape 推导是否正确
- 是否某层 reshape / flatten 逻辑在不同 batch 下有坑

### 面试里怎么说

> shape mismatch 我会分三层查：模型定义、导出结果、运行时输入。尤其动态 shape 下，先确认 profile 覆盖，再确认 context 实际设定的输入 shape。

这就是成熟答法。

---

## 8）“输入预处理不一致”对应怎么处理

### 你要解决的问题

这是实际项目里非常高频、但很多人会忽略的问题。官方 troubleshooting 也明确把系统化 debug 和错误信息定位放在核心位置，并建议先按步骤定位而不是盲猜。

### 处理方案

重点查这几项：

- RGB / BGR 顺序
- NCHW / NHWC
- resize 方法是否一致
- normalize 的 mean/std 是否一致
- 是否除以 255
- dtype 是 `float32` 还是 `uint8`
- batch 维是否正确
- tokenizer / padding / mask 规则是否一致（NLP/LLM 场景）

最有效办法不是“看代码猜”，而是：

**同一份输入**  
分别喂给：

1. 原框架模型
2. ONNXRuntime
3. TensorRT

然后逐层比对，先看输入 tensor 是否一致，再看输出在哪一层开始偏。

### 面试里怎么说

> 精度异常我优先先排预处理，因为很多问题不在 TensorRT 本身，而在前后处理没对齐。我的做法是同一份输入做多后端对齐比较。

这个答法很实战。

---

## 9）“动态 shape 要怎么配 optimization profile”对应怎么处理

### 你要解决的问题

这是招聘里最容易问的 TensorRT 核心点之一。

TensorRT 支持 dynamic shapes，但不是说你把某一维写成 `-1` 就完了；还需要配置 optimization profile。文档和 troubleshooting 都明确提到：要为不同尺寸/批次优化，关键是 profile，尤其 `kOPT`/opt shape 是优化重点。

### 处理方案

### 第一步：明确哪些维度真要动态

不要全都动态。  
只把确实会变化的维度设成 `-1`，比如：

- batch
- seq_len
- image height/width

### 第二步：给 profile 配 min / opt / max

这三个不是乱填的：

- **min**：最小可能输入
- **opt**：最常见输入，也是 TensorRT 重点优化的点
- **max**：业务允许的最大输入

### 第三步：按业务分 profile

比如 LLM：

- 一个 profile 适合短序列
- 一个 profile 适合长序列

比如视觉：

- 一个 profile 适合 1x3x224x224
- 一个 profile 适合 1x3x640x640

### 关键原则

**opt 要贴近真实流量主峰。**  
因为 TensorRT 对 opt shape 的优化最好，官方 FAQ 也明确说，为多个 batch/尺寸优化时，应在 profile 的 `OPT` 维度上做设计。

### 面试里怎么说

> 动态 shape 的关键不是 `-1`，而是 optimization profile。`min/opt/max` 要按真实业务分布来定，尤其 `opt` 不能乱填，否则既影响性能也影响可用性。

这是标准答案。

---

## 10）“profile 的 min/opt/max 不是乱填的”对应怎么处理

### 常见错误

很多人会写：

- min=1
- opt=max/2
- max=最大可能值

这往往很虚。

### 正确处理

要按真实请求分布定：

#### 例子 1：视觉模型

线上 90% 都是 224x224，少量 384x384  
那 opt 就该偏 224，而不是瞎取中间值

#### 例子 2：LLM

大部分 prompt 长度在 128~512  
只有少量到 2k  
那 opt 不应该设 2k

### 原因

- tactic 选择会围绕 opt shape 更优化
- shape 差太大，可能导致某些尺寸性能很烂
- engine build 时间、workspace 需求也可能受影响

### 面试里怎么说

> `min/opt/max` 本质上是在告诉 builder 真实工作区间，其中 `opt` 最重要，应该贴近高频请求而不是数学中值。

---

## 11）“profiling 动态 shape 网络前要先指定 profile”对应怎么处理

### 你要解决的问题

动态 shape 下，不先选 profile、不先设 shape，profiling 数据就可能没意义，甚至直接不能跑。

### 处理方案

- 先选择/激活正确的 optimization profile
- 再设置本次 inference 的实际输入 shape
- 然后再 enqueue / profile

### 为什么

因为动态网络不是固定单一执行计划；TensorRT 需要知道当前输入落在哪个 profile 范围内，才能确定执行配置。

### 面试里怎么说

> 动态 shape 下先 profile 再设 shape 是不对的，必须先落到某个 optimization profile，并把 context 的实际输入 shape 设好。

---

## 12）“INT8 不是随便开个开关就完事”对应怎么处理

### 你要解决的问题

INT8 是高频面试点，但很多人只会说“更快更省显存”。

### 处理方案

INT8 你至少要会讲三个点：

#### 1. 量化前提

要有动态范围信息，或者校准结果，或者显式 Q/DQ 图。

#### 2. 性能不等于必然更好

- 某些模型 INT8 提升明显
- 某些模型受限于 memory / shape / 某些层不支持，提升不明显

#### 3. 精度会掉

所以一定要做：

- calibration / scale 检查
- 和 FP32/FP16 baseline 对比
- 按业务指标验收，不是只看单层误差

### 面试里怎么说

> INT8 不是简单加开关，而是要解决动态范围和精度验收问题。没有校准或量化信息时，即使能跑，精度也未必可信。

这就已经比只会背“INT8 更快”强很多。

---

## 13）“INT8 校准/动态范围问题怎么查”对应怎么处理

### 处理方案

顺序建议这样：

**第一步：确认模型量化路径**

- 是 PTQ 校准
- 还是 QAT 导出的 ONNX
- 还是显式 Q/DQ

**第二步：先保留 FP32 / FP16 baseline**

- 不然你不知道误差从哪里来的

**第三步：看误差是全局的还是局部层引入的**

- 如果整体都歪，先怀疑输入、scale、校准集
- 如果局部开始偏，重点看敏感层

**第四步：检查校准数据**

- 校准集是否覆盖真实分布
- 是否样本太少
- 是否预处理和实际推理不一致

### 面试里怎么说

> INT8 出问题我先分是 scale 问题、校准集问题，还是模型本身对量化敏感。不会一上来就说 TensorRT 有 bug。

---

## 14）“性能不如预期时怎么查”对应怎么处理

### 你要解决的问题

模型转成 TensorRT 了，但没加速，甚至更慢。

### 处理方案

按这条线查：

#### 1. 先区分 build 问题还是 runtime 问题

- 是 engine 本身 tactic/fusion 不好
- 还是你 runtime 里频繁 malloc、频繁 H2D/D2H、同步太多

#### 2. 用 `trtexec` 先测纯 engine 性能

- 如果 `trtexec` 很快，而你业务代码很慢，说明问题在 runtime 集成
- 如果 `trtexec` 也不快，说明问题在模型/shape/profile/build

#### 3. 看动态 shape/profile 是否不合理

- opt shape 不贴业务
- max 太大拖累 tactic
- 频繁跨 profile

#### 4. 看精度模式

- FP16 是否真正生效
- 某些层是否退回更慢实现

#### 5. 做 profiling

官方 best practices 明确建议用 `trtexec` 做 benchmark 和 profiling，再逐步做优化。

### 面试里怎么说

> 性能问题我会先用 `trtexec` 切分问题边界：如果裸 engine 快，业务慢，说明是集成问题；如果裸 engine 都慢，再看 profile、precision、tactic 和算子支持情况。

这句很好用。

---

## 15）“故障排查先开什么”对应怎么处理

### 处理方案

先开详细日志。  
官方 troubleshooting 的思路也是：先系统化搜错误信息，再收集 diagnostics，再按模块排查。

你自己工作里一般要做的是：

- 打开 verbose 日志
- 记录 parser/build/runtime 三段日志
- 保存失败的 ONNX、命令行、输入 shape
- 把问题缩成最小可复现

### 面试里怎么说

> 排 TensorRT 问题我不会上来盲猜，我会先拿 verbose 日志，把问题分到 parser、builder、runtime 三段，再决定是图、profile、plugin 还是输入问题。

---

## 16）“plugin 基本认知”对应怎么处理

### 你要解决的问题

招聘里写“熟练 TensorRT”，很多面试官默认你至少要知道：

- 什么情况下写 plugin
- plugin 基本组件有哪些
- 不是让你一定写过很复杂的

### 处理方案

你至少要掌握这条主线：

1. **实现 plugin 类**
2. **实现 creator**
3. **注册到 plugin registry**
4. **把 plugin 接到 network / ONNX parser 流程里**

NVIDIA 当前官方文档明确说，从 TensorRT 10.0 开始，推荐的新插件接口就是 `IPluginV3`，旧的 V2 系列都在弃用；而且插件能力被拆成 core/build/runtime 三部分。

### 面试里怎么说

> 现在新插件开发我会按 `IPluginV3` 路线理解，不再把老的 V2 接口当主线。插件开发至少包括 plugin 类、creator、注册、序列化/反序列化，以及接入 network/ONNX parser 这几个环节。

这句话非常像“真的看过并理解过最新文档”的答法。

---

## 17）“什么时候该写 plugin”对应怎么处理

### 判断标准

该写 plugin 的场景一般是：

- TensorRT 不支持该层
- 图改写很困难
- 这个 op 很关键，值得维护
- 性能要求高，必须自定义 kernel/实现

不该写 plugin 的场景一般是：

- 只是导出图有点丑，改一下前向就能解决
- 只是临时实验
- 团队没人能长期维护

### 面试里怎么说

> plugin 是为了补不支持层或拿更强的算子控制力，不是为了炫技。能改图解决的我优先改图，只有在可维护性和性能都值得时才写 plugin。

---

## 18）“做到什么程度才真的够写‘熟练 TensorRT’”对应怎么处理

### 我给你的实际标准

如果你简历写“熟练 TensorRT”，最低要能接住下面这几类活：

#### A. 常规部署

- ONNX → engine
- runtime 跑起来
- 会 `trtexec`
- 会 FP16

#### B. 动态 shape

- 会配 `-1`
- 会配 profile
- 知道 `min/opt/max` 怎么按业务设
- 知道动态 shape 下 profile/context 的使用顺序

#### C. 常见排障

- unsupported op
- shape mismatch
- 预处理不一致
- 精度异常
- 性能不如预期

#### D. plugin 基本认知

- 知道什么时候该上 plugin
- 知道现在主线接口是 `IPluginV3`，不是老接口硬背一堆名字

---

## 19）如果面试官按招聘导向追问，你最该准备的“标准题”

你至少要能完整回答这几题：

1. **给你一个 ONNX，怎么转 TensorRT？**
2. **为什么先用 `trtexec`，什么时候再上 API？**
3. **dynamic shape 和 optimization profile 怎么配？**
4. **`min/opt/max` 为什么不能乱填？**
5. **INT8 为什么不是开开关就完？**
6. **性能没提升你怎么排？**
7. **unsupported op 你怎么处理？**
8. **plugin 什么时候需要写？现在推荐哪个接口？**

---

## 20）你这段话里所有“问题”对应的最短总结版

我给你压成一句话一条：

- **安装验证**：做环境闭环，不只看包装没装上
- **ONNX→engine**：先保证 ONNX 干净，再用 `trtexec` 首转
- **`trtexec`**：首选验证、benchmark、profiling 工具
- **builder/runtime**：builder 负责编译优化，runtime 负责执行部署
- **FP16**：先留 FP32 baseline，再做精度/性能 A/B
- **unsupported op**：先查导出和改图，最后才写 plugin
- **shape mismatch**：按模型定义→导出结果→运行时输入三层排
- **预处理不一致**：多后端同输入对齐比较
- **dynamic shape**：核心不只是 `-1`，而是 optimization profile
- **min/opt/max**：按真实业务分布配，尤其 opt 要贴主流量
- **动态 shape profiling**：先选 profile、设 shape，再 profile
- **INT8**：核心是动态范围/校准/精度验收
- **性能排查**：先用 `trtexec` 切分是 engine 问题还是集成问题
- **plugin**：知道何时需要、基本开发路径、当前主线是 `IPluginV3`

---

如果你愿意，我下一条可以直接给你整理成 **“面试问：你为什么敢在简历里写熟练 TensorRT？——一段 2 分钟标准口述答案”**。