# 项目与性能排查

## 高频题

### 1. 部署先看哪些指标

- 时延：
  - `TTFT`
  - `Prefill latency`
  - `Decode latency`
- 吞吐：
  - `Prefill tokens/s`
  - `Decode tokens/s`
  - `QPS`
- 资源：
  - 峰值显存
  - 稳态显存
  - KV Cache 占用
  - GPU/CPU 利用率
- 稳定性：
  - 温度
  - 功耗
  - 长时间抖动
  - 输出退化

### 2. `TTFT`、`Prefill latency`、`Decode latency` 怎么区分

- `TTFT`
  - 用户从发请求到看到第一个 token 的总等待
- `Prefill latency`
  - prompt 进入模型后的计算耗时
- `Decode latency`
  - 生成阶段每步推进的耗时

一句话：

- `TTFT` 是用户视角
- `Prefill` 和 `Decode` 更偏模型内部视角

### 3. 定位问题的固定顺序

1. 先拆阶段
2. 再区分 CPU 侧还是 GPU 侧
3. 再区分算力、带宽还是调度
4. 最后决定抠 kernel 还是改系统

### 4. `perf`、`nsys`、`ncu` 怎么分工

- `perf`
  - 看 CPU 热点、系统调用、上下文切换
- `nsys`
  - 看整体 timeline、launch gap、同步、memcpy
- `ncu`
  - 看单个 kernel 的 occupancy、stall、L2、DRAM、Tensor Core

### 5. Prefill 慢优先看什么

- GEMM 是否走高性能路径
- Attention 实现是否合适
- Tensor Core 利用率
- kernel 是否过碎
- 是否有融合空间

### 6. Decode 慢优先看什么

- KV Cache 读写和布局
- 小 kernel 是否太多
- L2 命中率
- launch overhead
- 调度是否让 prefill 和 decode 互相干扰

### 7. OOM 和 page fault 怎么讲

- `OOM`
  - 内存放不下，申请失败或被系统杀掉
- `page fault`
  - 页访问时触发缺页处理
- 端侧常见来源：
  - 权重太大
  - KV Cache 持续增长
  - workspace 太大
  - `mmap` 首次访问
  - UMA 迁移

### 8. 什么时候该继续抠 kernel，什么时候该转系统层

- 适合继续抠 kernel：
  - 热点高度集中
  - 单个 kernel 占大头
  - `ncu` 还能看出明显空间
- 适合转系统层：
  - 小 kernel 很碎
  - launch gap 和同步多
  - CPU 提交和调度占比高
  - 单 kernel 再快整体收益也有限

### 9. kernel launch overhead 怎么减

- 算子融合
- 减少 tiny kernel
- 减少不必要同步
- 用 `CUDA Graph`
- 让 copy 和 compute overlap
- 增大单次工作量

## 高频口头答案

### 如果面试官问“你怎么定位慢”

- 我通常先做分阶段埋点，把请求拆成排队、tokenize、prefill、decode、sample、回传。
- 然后用 `nsys` 看整条时间线，确认是 CPU 提交慢、GPU 执行慢，还是同步和 memcpy 在关键路径上。
- 如果热点集中在某个 kernel，再用 `ncu` 下钻；如果怀疑 CPU 前后处理或调度问题，再用 `perf`。

### 如果问“吞吐小且时延大说明什么”

- 这通常说明系统既没把单请求做好，也没把整体资源吃满。
- 常见是调度、同步、kernel 碎片化、带宽瓶颈和前后处理开销叠在一起。

### 如果问“端侧部署最怕什么”

- 不只是算得慢，更怕：
  - 放不下
  - 跑不稳
  - 长时间抖
  - 一优化就掉精度

## 易错点

- 不要混淆用户指标和模型内部指标
- 不要一上来就盯某个 kernel，看不到系统层问题
- 不要把 GPU 利用率低直接等同于“GPU 不重要”
- 不要只做一次 benchmark 就下结论，长压测很关键

## 原笔记入口

- `[[Notion/面试整理/03-项目问题准备/01-项目部署问题]]`
- `[[Notion/面试整理/03-项目问题准备/02-推理框架问题]]`
- `[[Notion/面试整理/03-项目问题准备/03-推理算子问题]]`
- `[[Notion/面试整理/03-项目问题准备/04-问题定位方法]]`
