# llama.cpp Qwen3 测试接口改动整理

这份目录整理的是一次针对 `llama.cpp` 的定向改动：

**目标：** 给 `src/models/qwen3.cpp` 对应的**整条执行路径**包装一个**对外测试接口**，供测试程序直接调用。

## 改动概览

本次实际涉及的文件：

- `src/llama-context.h`
- `src/llama-context.cpp`
- `tests/test-qwen3-graph.cpp`
- `tests/CMakeLists.txt`

## 设计思路

没有重写 `qwen3.cpp` 的执行逻辑，而是复用现有：

- `llama_context`
- `memory`
- `llama_model::build_graph(...)`

然后新增一个 **internal test helper**：

```cpp
llama_internal_build_qwen3_graph_for_test(...)
```

这样测试代码可以直接拉起 **Qwen3 的完整 graph build 路径**，但不影响正常主执行逻辑。

## 文件说明

- `src/llama-context.h.snippet.cpp`
  - 展示头文件里新增的声明与接口
- `src/llama-context.cpp.snippet.cpp`
  - 展示实现部分新增的桥接代码
- `tests/test-qwen3-graph.cpp`
  - 新增的测试程序完整代码
- `tests/CMakeLists.txt.snippet.cmake`
  - 新增的测试目标注册片段

## 新增内容标注方式

我在代码片段里统一使用了这样的注释：

- `// [ADDED] ...`
- `// [ADDED BEGIN]`
- `// [ADDED END]`

用来明确指出这次新增的代码范围。
