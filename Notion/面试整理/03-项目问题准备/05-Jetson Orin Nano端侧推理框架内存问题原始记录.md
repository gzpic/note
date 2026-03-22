# Jetson Orin Nano端侧推理框架内存问题原始记录

如果要给jetson orin nano适配一个端侧的大模型推理框架那内存应该如何提前分配和安排呢                                    
                                                                                     
⏺ Jetson Orin Nano 端侧大模型推理内存规划                                                                               
                                                                                                                
  硬件约束                                                                                                              
                  
  Jetson Orin Nano 的核心特点是 统一内存架构（Unified Memory）：CPU 和 GPU 共享 8GB LPDDR5（Super 版为 8GB，标准版也是  
  8GB），没有独立显存。这意味着内存规划必须精打细算。

  ---
  内存预算分配（以 8GB 为例）

  ┌─────────────────────────────────────────────────┐
  │              总可用内存: ~8GB                      │
  ├─────────────────────────────────────────────────┤
  │  OS + 系统服务          │  ~1.0 - 1.5 GB         │
  │  模型权重 (Weights)      │  ~2.0 - 4.0 GB         │
  │  KV Cache               │  ~0.5 - 2.0 GB         │
  │  激活值 (Activations)    │  ~0.3 - 0.5 GB         │
  │  推理框架运行时开销       │  ~0.2 - 0.5 GB         │
  │  安全余量                │  ~0.5 GB               │
  └─────────────────────────────────────────────────┘

  ---
  各部分详细规划

  1. 系统保留（~1.0–1.5GB）

  - Linux 内核 + JetPack 驱动栈: ~500MB
  - CUDA/TensorRT 运行时库:      ~200-300MB
  - 桌面环境(如有):              ~300-500MB

  建议：使用无桌面的 headless 模式，可以省出 300–500MB。

  # 切换到无桌面模式
  sudo systemctl set-default multi-user.target

  2. 模型权重（最大开销）

  这是最关键的部分。量化是必须的：

  ┌──────────┬───────┬───────┬─────────────────┐
  │ 模型规模 │ FP16  │ INT8  │ INT4 (GPTQ/AWQ) │
  ├──────────┼───────┼───────┼─────────────────┤
  │ 1.5B     │ 3.0GB │ 1.5GB │ 0.75GB          │
  ├──────────┼───────┼───────┼─────────────────┤
  │ 3B       │ 6.0GB │ 3.0GB │ 1.5GB           │
  ├──────────┼───────┼───────┼─────────────────┤
  │ 7B       │ 14GB  │ 7.0GB │ 3.5GB           │
  ├──────────┼───────┼───────┼─────────────────┤
  │ 14B      │ 28GB  │ 14GB  │ 7.0GB           │
  └──────────┴───────┴───────┴─────────────────┘

  结论：在 8GB Orin Nano 上，实际可用的模型规模：
  - INT4 量化：最大可跑 7B 模型（紧凑配置）
  - INT8 量化：最大 3B 模型比较舒适
  - 推荐选择：Qwen2.5-3B-INT4、Llama-3.2-3B-INT4、或 SmolLM2-1.7B

  3. KV Cache 预分配（关键优化点）

  KV Cache 大小公式：

  KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes

  以 3B 模型（32层, 8 KV heads, head_dim=128）为例：

  每 token KV 内存 = 2 × 32 × 8 × 128 × 2 bytes(FP16) = 128 KB/token
                   = 2 × 32 × 8 × 128 × 1 byte(INT8)  =  64 KB/token

  上下文长度  │  FP16 KV    │  INT8 KV
  512 tokens │  64 MB      │  32 MB
  1024       │  128 MB     │  64 MB
  2048       │  256 MB     │  128 MB
  4096       │  512 MB     │  256 MB

  建议：
  - 限制 max_seq_len 为 1024–2048，不要盲目拉满
  - KV Cache 使用 FP8 或 INT8 量化
  - 采用 预分配 + 分页 策略（类似 vLLM 的 PagedAttention）

  4. 激活值内存

  Activations ≈ batch_size × seq_len × hidden_dim × 2 bytes

  - 对于 batch_size=1, seq_len=1024, hidden=3072: 约 6MB
  - Prefill 阶段会短暂峰值更高，需要预留 200–500MB

  ---
  内存管理策略

  策略一：静态预分配（推荐）

  // 启动时一次性分配，避免运行时 malloc/cudaMalloc 碎片
  typedef struct {
      void* model_weights;      // mmap 映射，按需加载
      void* kv_cache;           // cudaMallocManaged 预分配
      void* activation_buffer;  // 双缓冲，prefill/decode 复用
      void* scratch_buffer;     // 临时计算空间
  } InferenceMemoryPool;

  void init_memory_pool(InferenceMemoryPool* pool, ModelConfig* cfg) {
      // 模型权重: mmap 方式加载，利用统一内存零拷贝
      pool->model_weights = mmap(NULL, cfg->weight_size,
                                 PROT_READ, MAP_PRIVATE, weight_fd, 0);

      // KV Cache: 使用 CUDA managed memory
      cudaMallocManaged(&pool->kv_cache,
                        2 * cfg->num_layers * cfg->num_kv_heads *
                        cfg->head_dim * cfg->max_seq_len * sizeof(half));

      // 激活缓冲: 只需要最大单层的空间（逐层复用）
      size_t act_size = cfg->max_batch * cfg->max_seq_len *
                        cfg->hidden_dim * sizeof(half);
      cudaMallocManaged(&pool->activation_buffer, act_size);

      // Scratch: 用于 GEMM 临时空间
      cudaMallocManaged(&pool->scratch_buffer, 64 * 1024 * 1024); // 64MB
  }

  策略二：利用统一内存的零拷贝特性

  // Jetson 的统一内存优势：CPU/GPU 共享物理地址，无需显式拷贝
  // 但需要注意内存访问模式以避免频繁的 CPU-GPU 一致性开销

  // 推荐: 对模型权重使用 cudaMemAdvise 提示 GPU 优先访问
  cudaMemAdvise(pool->model_weights, weight_size,
                cudaMemAdviseSetPreferredLocation, 0); // GPU device 0
  cudaMemAdvise(pool->model_weights, weight_size,
                cudaMemAdviseSetAccessedBy, 0);

  // KV Cache 同理
  cudaMemAdvise(pool->kv_cache, kv_size,
                cudaMemAdviseSetPreferredLocation, 0);

  策略三：模型权重分层加载（极端省内存）

  # 如果模型权重接近内存上限，可以逐层加载计算
  # 牺牲速度换空间 (layer-by-layer offloading)
  class LayerWiseInference:
      def __init__(self, model_path, max_resident_layers=8):
          self.max_resident = max_resident_layers
          self.total_layers = 32
          # 只在内存中保留部分层的权重
          self.weight_buffer = allocate(max_resident_layers * per_layer_size)

      def forward(self, x):
          for i in range(0, self.total_layers, self.max_resident):
              # 加载当前批次的层权重到 buffer
              load_layers(i, min(i + self.max_resident, self.total_layers))
              for layer_idx in range(i, min(i + self.max_resident, self.total_layers)):
                  x = self.compute_layer(layer_idx, x)
          return x

  ---
  实用建议总结

  ┌────────────┬──────────────────────────────────────────────┐
  │    项目    │                   推荐做法                   │
  ├────────────┼──────────────────────────────────────────────┤
  │ 量化       │ 权重 INT4 (AWQ/GPTQ)，KV Cache INT8/FP8      │
  ├────────────┼──────────────────────────────────────────────┤
  │ 模型选择   │ 1.5B–3B 舒适，7B-INT4 极限可跑               │
  ├────────────┼──────────────────────────────────────────────┤
  │ 上下文长度 │ 限制 1024–2048 tokens                        │
  ├────────────┼──────────────────────────────────────────────┤
  │ batch size │ 固定为 1（端侧通常单用户）                   │
  ├────────────┼──────────────────────────────────────────────┤
  │ 权重加载   │ mmap + cudaMemAdvise，利用零拷贝             │
  ├────────────┼──────────────────────────────────────────────┤
  │ KV Cache   │ 启动时静态预分配，避免动态扩展               │
  ├────────────┼──────────────────────────────────────────────┤
  │ 激活值     │ 逐层复用同一块 buffer                        │
  ├────────────┼──────────────────────────────────────────────┤
  │ 系统       │ headless 模式，关闭不必要服务                │
  ├────────────┼──────────────────────────────────────────────┤
  │ Swap       │ 配置 zram swap 作为安全网（但不要依赖）      │
  ├────────────┼──────────────────────────────────────────────┤
  │ 框架选择   │ llama.cpp (GGUF INT4)、TensorRT-LLM、MLC-LLM │
  └────────────┴──────────────────────────────────────────────┘

  推荐的框架适配路径

  llama.cpp (最易上手，GGUF Q4_K_M 量化)
    └── 直接编译 aarch64 + CUDA，内存管理透明
    └── mmap 加载权重，自动利用统一内存

  TensorRT-LLM (性能最优)
    └── 需要离线 build engine，内存布局更可控
    └── 支持 Weight-Only Quantization (W4A16)

  MLC-LLM (灵活)
    └── 支持 Vulkan/CUDA 后端
    └── 自动内存规划

  如果需要，我可以帮你写一个具体的内存预分配模块或者某个框架的适配代码。

✻ Brewed for 57s

❯ 权重是编译器就知道大小的吗而激活值是在跑起来之后才知道大小的对不对呢                                                  
                                                                       
⏺ 大方向是对的，但需要更精确地区分几个阶段。严格来说并不是 C 编译器意义上的"编译期"，而是要分三个时间点来看：           
                                                                                                                        
  三个时间点
                                                                                                                        
  1. 框架编译期 (C/C++ compile time)
     → 两者的大小都不知道                                                                                               
                  
  2. 模型加载期 (model load time)
     → 权重大小已完全确定 ✓
     → 激活值的上界可以确定（如果你固定了 max_seq_len / max_batch）

  3. 推理运行期 (inference runtime)
     → 每次请求的激活值实际大小才真正确定

  权重 — 模型定义后就确定了

  权重的大小完全由模型结构超参数决定，跟输入无关：

  线性层权重: [hidden_dim, hidden_dim] × dtype_bytes
  QKV 权重:  [hidden_dim, 3 × num_heads × head_dim] × dtype_bytes
  ...

  这些超参数在模型训练完、保存成文件的那一刻就固定了。所以在加载模型时（而不是 C
  编译期），你可以直接从配置文件里算出精确的总大小：

  # 从 config.json 就能精确算出，不需要跑任何推理
  total_weight_bytes = 0
  for name, shape in model.named_parameters():
      total_weight_bytes += np.prod(shape) * dtype_size
  # 这个数字是确定性的，永远不会变

  激活值 — 要分两种情况

  Prefill 阶段（处理输入 prompt）

  激活值取决于输入序列长度，每次请求都不同：

  激活值大小 = batch_size × seq_len × hidden_dim × dtype_bytes
                           ^^^^^^^^
                           这是动态的，每次请求不一样

  用户发 10 个 token 和 1000 个 token，激活值差 100 倍。

  Decode 阶段（逐 token 生成）

  每步只算 1 个新 token，激活值反而是近似固定的：

  激活值大小 = batch_size × 1 × hidden_dim × dtype_bytes
                           ^^^
                           始终为 1

  所以实际工程中怎么做

  既然激活值大小是动态的，端侧通常的做法是按上界预分配：

  // 模型加载时：权重大小精确已知，直接分配
  void* weights = mmap(fd, EXACT_WEIGHT_SIZE, ...);

  // 激活值：按你允许的最大输入长度预分配上界
  // 大小 = max_batch × max_seq_len × hidden_dim × sizeof(half)
  size_t max_act_size = 1 * 2048 * 3072 * 2;  // 约 12MB
  cudaMallocManaged(&activation_buf, max_act_size);

  // 实际推理时，只使用其中一部分
  // 输入 512 tokens → 用了 25%
  // 输入 2048 tokens → 用满 100%

  逐层复用同一块 buffer 可以进一步省内存，因为每层的激活值算完之后就不再需要了（推理不需要反向传播）：

  Layer 0: 读 activation_buf → 写 activation_buf_swap
  Layer 1: 读 activation_buf_swap → 写 activation_buf
  Layer 2: 读 activation_buf → 写 activation_buf_swap
  ...（乒乓复用，只需两块 buffer）

  总结

  ┌──────────┬─────────────────┬──────────────────────┬─────────────────────────┐
  │          │  大小何时确定   │      取决于什么      │     能否精确预分配      │
  ├──────────┼─────────────────┼──────────────────────┼─────────────────────────┤
  │ 权重     │ 模型定义/训练后 │ 模型结构超参数       │ 能，精确值              │
  ├──────────┼─────────────────┼──────────────────────┼─────────────────────────┤
  │ KV Cache │ 运行时逐步增长  │ seq_len（动态）      │ 按上界预分配            │
  ├──────────┼─────────────────┼──────────────────────┼─────────────────────────┤
  │ 激活值   │ 每次请求时      │ 输入 seq_len（动态） │ 按上界预分配 + 逐层复用 │
  └──────────┴─────────────────┴──────────────────────┴─────────────────────────┘

  所以你的理解核心是对的：权重是静态的，激活值是动态的。只是"静态"不是 C 编译器意义上的 compile-time
  constant，而是模型确定后就不变了。

✻ Churned for 37s                
                                           
❯ 推理框架的运行时包含了什么东西占内存如何减小运行时占的内存                                                            
                                                                                                                        
⏺ 推理框架运行时内存拆解                                                                                                
                                                                                                                        
  运行时到底包含哪些东西
                                                                                                                        
  把运行时占用从大到小排列：
                                                                                                                        
  推理框架运行时 (~200-500MB)
  │
  ├── 1. CUDA/GPU 运行时栈        ~100-200MB   ← 大头
  │   ├── CUDA Runtime (libcudart)
  │   ├── cuBLAS (GEMM 计算库)
  │   ├── cuDNN (可选)
  │   └── GPU context (页表、命令队列)
  │
  ├── 2. 计算临时缓冲区            ~50-100MB
  │   ├── GEMM workspace (矩阵乘法临时空间)
  │   ├── Softmax / LayerNorm 中间结果
  │   └── AllReduce 通信缓冲 (多卡才有)
  │
  ├── 3. 内存分配器元数据           ~20-50MB
  │   ├── 空闲块链表 / 伙伴系统簿记
  │   ├── 碎片 (已分配但未使用的间隙)
  │   └── 对齐填充 (CUDA 要求 256B/512B 对齐)
  │
  ├── 4. 计算图 / 算子描述          ~10-30MB
  │   ├── 每层的算子元信息 (输入输出 shape、dtype)
  │   ├── cuBLAS handle / 算子描述符
  │   └── TensorRT 的 engine 序列化缓存
  │
  ├── 5. Tokenizer                 ~5-20MB
  │   ├── 词表 (vocab, 通常 32K-150K 条)
  │   ├── BPE merge 规则表
  │   └── 特殊 token 映射
  │
  ├── 6. 采样器状态                 ~1-5MB
  │   ├── logits 缓冲 (vocab_size × sizeof(float))
  │   ├── top-k/top-p 排序临时数组
  │   └── 随机数生成器状态
  │
  ├── 7. 线程池 / 调度器            ~1-5MB
  │   ├── 线程栈 (每线程 ~1MB)
  │   └── 任务队列
  │
  └── 8. 如果用了 Python...         ~30-80MB  ← 可完全避免
      ├── Python 解释器
      ├── PyTorch / numpy 等库
      └── GC 元数据、引用计数

  各部分详细分析与优化

  1. CUDA Runtime — 最大的隐形开销

  这是很多人忽略的大头。CUDA 初始化时会分配一个 GPU context：

  $ nvidia-smi 显示的 "GPU Memory Used" 里，
    哪怕你什么都没做，一个空的 CUDA 进程就吃掉 ~100-200MB

  这里面包含：
  - GPU 页表 (管理虚拟地址 → 物理地址映射)
  - 命令队列 (command buffer / stream)
  - cuBLAS 内部预分配的 workspace
  - JIT 编译缓存 (PTX → SASS)

  优化手段：

  # 限制 CUDA context 预留内存
  export CUDA_MODULE_LOADING=LAZY          # 延迟加载内核，省几十 MB
  export CUDA_AUTO_BOOST=0                 # 关闭频率自动提升相关缓存

  # 如果用了 cuBLAS，限制其 workspace
  cublasSetWorkspace(handle, workspace, SMALL_SIZE);  # 默认会贪心分配

  // 手动控制 cuBLAS workspace 而不是让它自己分配
  // 默认 cuBLAS 可能分配 32-256MB workspace
  size_t cublas_workspace = 4 * 1024 * 1024;  // 只给 4MB
  cublasSetWorkspace(handle, my_buffer, cublas_workspace);

  2. 计算临时缓冲区

  GEMM（矩阵乘法）需要临时空间。cuBLAS 的策略是越大越快：

  // cuBLAS 内部：我要一块 workspace 来做分块乘法
  // 给我越大，我能选的算法越多，越快
  cublasGemmEx(...);
  // 默认行为：尽可能多吃内存

  // 优化：手动选择小 workspace 的算法
  cublasLtMatmulAlgoGetHeuristic(..., &algo, maxWorkspace=4MB);
  // 性能可能降 5-10%，但省几十 MB

  真正省内存的做法 — 全局共享一块 scratch buffer：

  // 所有算子共享同一块临时空间（它们不会同时执行）
  void* global_scratch = cudaMalloc(MAX_SCRATCH_SIZE);

  // Layer 0 的 GEMM 用 global_scratch
  // Layer 0 的 Softmax 也用 global_scratch（串行执行，不冲突）
  // Layer 1 的 GEMM 还是用 global_scratch
  // ...

  3. 内存分配器开销

  框架通常自己维护一个内存池来避免反复调 cudaMalloc（很慢），但分配器本身也有开销：

  问题：
    cudaMalloc(512 bytes)  → 实际占用 4096 bytes（最小分配粒度）
    反复 malloc/free       → 碎片化，空闲块簿记膨胀

  优化 — 用 arena 分配器替代通用分配器：

  // Arena 分配器：一次分配大块，内部线性推进，零碎片
  typedef struct {
      void* base;
      size_t capacity;
      size_t offset;    // 当前分配位置
  } Arena;

  void* arena_alloc(Arena* a, size_t size) {
      size = ALIGN_UP(size, 256);  // CUDA 对齐
      void* ptr = (char*)a->base + a->offset;
      a->offset += size;
      return ptr;
  }

  void arena_reset(Arena* a) {
      a->offset = 0;  // 一次性"释放"全部，零开销
  }

  // 每次推理请求:
  //   arena_alloc() 分配激活值、临时缓冲
  //   推理完成后 arena_reset() 一次性回收
  //   没有碎片，没有簿记开销

  4. 计算图 / 算子描述

  每一层都需要存一些元信息：

  // 每层大约占:
  struct LayerMeta {
      cublasLtMatmulDesc_t  matmul_desc;   // ~几百字节
      cublasLtMatrixLayout_t layouts[3];    // A, B, C 描述
      void* algo_workspace_ptr;
      size_t input_shape[4];
      size_t output_shape[4];
  };
  // 32 层 × 每层 ~几 KB = 很小，通常不是瓶颈

  但 TensorRT engine 是个例外：序列化后的 engine 可以占 几十到上百 MB，因为它内部存了多种算法候选和 profiling 数据。

  # TensorRT 优化：build 时指定更小的 workspace
  trtexec --workspace=64  # 限制 64MB（默认可能 1GB+）

  5. Tokenizer

  # SentencePiece / tiktoken 词表
  # 32K 词表 × 平均 token 长 10 bytes ≈ 320KB (很小)
  # 但 BPE merge 规则表可能有 50K+ 条 ≈ 几 MB

  # HuggingFace tokenizer 的问题：会加载整个 tokenizers 库
  # 占 ~15-20MB

  # 优化：用纯 C 实现的 tokenizer
  # llama.cpp 自带的 tokenizer < 2MB
  # sentencepiece C 库 < 5MB

  6. Python 解释器（如果存在的话）

  Python 进程本身:          ~10MB
  import torch:             ~300MB  ← 灾难性
  import transformers:      ~100MB
  import numpy:             ~20MB

  这是最值得优化的点之一：端侧推理完全不需要 Python。

  ---
  优化策略总结

  按效果从大到小排列：

  效果排名    策略                           节省内存
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1       去掉 Python，用纯 C/C++ 推理     ~200-400MB
    2       CUDA_MODULE_LOADING=LAZY         ~50-80MB
    3       限制 cuBLAS workspace            ~30-100MB
    4       全局共享 scratch buffer           ~30-50MB
    5       Arena 分配器替代通用 allocator     ~20-40MB
    6       用 C tokenizer 替代 HF tokenizer  ~15-20MB
    7       用 CUDA Graph 减少命令队列开销     ~10-20MB
    8       关闭 headless 图形桌面            ~300-500MB (系统级)

  极致方案：单进程极简运行时

  // 理想的端侧推理进程内存构成:
  //
  // 代码段 (.text):    ~2MB    (推理引擎二进制)
  // CUDA context:     ~80MB   (LAZY 模式，最小化)
  // 模型权重:          N MB    (mmap 映射)
  // KV Cache:         预分配
  // 1个 scratch buf:  ~16MB   (所有算子共用)
  // Tokenizer:        ~2MB    (C 实现)
  // 采样 logits buf:  ~0.5MB
  // ─────────────────────────
  // 运行时总开销:      ~100MB  (不含权重和 KV Cache)

  所以回答你的核心问题：运行时里 CUDA 栈和计算库是大头（不可避免但可压缩），Python
  是最大的浪费（可完全去除），剩下的分配器碎片和临时缓冲通过 arena + 共享 buffer 可以压到很小。

✻ Cogitated for 1m 4s                    
                                                       
❯ 我问的是如果要设计一个端侧推理框架运行时哪些占内存如何优化减小运行时占的内存                                                        
                                                                                                                        
⏺ 自研端侧推理框架：运行时内存设计                                                                                      
                                                                                                                        
  抛开 CUDA/cuBLAS 这些外部依赖不谈，你自己写的运行时本身有这几块东西吃内存：
                                                                                                                        
  运行时核心组成  
                                                                                                                        
  你写的推理框架运行时
  │
  ├── 1. 计算图表示 (Graph IR)
  ├── 2. 内存分配器 (Allocator)
  ├── 3. 算子注册表 + kernel 调度
  ├── 4. Tensor 元数据描述
  ├── 5. 调度器 / 执行器
  ├── 6. Tokenizer
  └── 7. Sampler

  ---
  1. 计算图表示

  你得有一个数据结构描述"先算什么后算什么"：

  朴素做法（PyTorch 风格，吃内存）：

  // 每个节点是一个堆分配的对象，互相用指针连接
  struct Node {
      std::string name;              // 堆分配字符串
      std::string op_type;           // 又一个
      std::vector<Node*> inputs;     // 堆分配动态数组
      std::vector<Node*> outputs;
      std::map<std::string, Attr> attrs;  // 红黑树，每个 KV 都是堆分配
  };
  std::vector<Node*> graph;  // 指针数组

  // 问题：
  //   32 层 transformer，每层 ~20 个算子，共 ~640 个 Node
  //   每个 Node: string 分配 + vector 分配 + map 分配
  //   大量小对象 → malloc 头部开销 + 碎片 + cache miss
  //   实际占用可能 5-10MB（数据本身才几百 KB）

  优化做法 — 扁平化表示：

  // 所有节点存在一个连续数组里，用索引而不是指针
  struct Op {
      uint16_t op_type;         // 枚举值，不用字符串
      uint16_t input_start;     // 在 edges[] 数组中的起始位置
      uint8_t  input_count;
      uint16_t output_start;
      uint8_t  output_count;
  };

  struct Graph {
      Op      ops[MAX_OPS];        // 连续内存，零堆分配
      int16_t edges[MAX_EDGES];    // 所有边存一个数组
      int     num_ops;
  };

  // 640 个算子 × 12 bytes = 7.5KB
  // 全部运行时图表示 < 64KB

  更极端 — 根本不需要图：

  LLM 的计算流程是线性的（逐层串行），完全可以硬编码：

  // 直接写死执行流程，连图表示都省了
  void forward(Model* m, int* tokens, int seq_len) {
      embed(m->embed_weight, tokens, seq_len, buf_a);
      for (int l = 0; l < m->n_layers; l++) {
          rmsnorm(buf_a, m->layers[l].attn_norm, buf_b);
          attention(m, l, buf_b, buf_a);   // 包含 QKV 投影 + softmax + output 投影
          rmsnorm(buf_a, m->layers[l].ffn_norm, buf_b);
          ffn(m, l, buf_b, buf_a);         // 包含 gate + up + down
      }
      rmsnorm(buf_a, m->final_norm, buf_b);
      logits(m->lm_head, buf_b, out_logits);
  }

  // 图表示内存占用: 0
  // llama.cpp 就是这么干的

  ---
  2. 内存分配器

  你的框架内部也需要管理 GPU 内存的分配和释放，这个分配器自身有簿记开销：

  通用分配器的问题：

  每次 malloc 的隐形成本：
    ┌──────────┬────────────────────┐
    │ 16B 头部  │   用户数据           │
    │ (size,   │                    │
    │  flags)  │                    │
    └──────────┴────────────────────┘

    空闲链表节点、分裂合并逻辑、对齐填充...
    管理 1000 个小块 → 簿记本身可能占几 MB
    还有碎片问题：总空闲 100MB 但最大连续块只有 20MB

  端侧推理的特点：所有 buffer 大小在模型加载时就可以算出来。所以：

  // 最优解：静态偏移量分配，运行时零开销
  //
  // 在模型加载阶段做离线规划：
  // 分析每个 tensor 的生命周期，算出互不冲突的偏移量

  struct TensorSlot {
      size_t offset;   // 在大 buffer 中的偏移
      size_t size;
  };

  // 离线规划结果（编译期或加载期算好）：
  //   buf_a: offset=0,       size=12MB    (生命周期: layer input/output)
  //   buf_b: offset=12MB,    size=12MB    (生命周期: 中间结果)
  //   qkv:   offset=0,       size=8MB     (和 buf_a 复用！生命周期不重叠)

  // 运行时只需要一次分配
  void* pool = cudaMalloc(total_planned_size);

  // 之后所有 tensor 都是 pool + offset，无需任何分配器
  float* buf_a = (float*)(pool + slots[0].offset);
  float* buf_b = (float*)(pool + slots[1].offset);

  // 分配器簿记开销: 0
  // 碎片: 0
  // 运行时 malloc 调用: 0

  生命周期分析的核心思路：

  时间轴 →
  算子1:  [===buf_a===]
  算子2:       [===buf_b===]
  算子3:                     [===buf_c===]

  buf_a 和 buf_c 生命周期不重叠 → 可以共享同一段内存
  规划后：
    pool: |----buf_a/buf_c----|----buf_b----|
    总共只需要 max(a,c) + b，而不是 a + b + c

  ---
  3. 算子注册表 + kernel 调度

  你的框架需要知道"遇到 matmul 该调哪个函数"：

  重型做法（PyTorch/TensorFlow 风格）：

  // 全局注册表，支持任意算子、任意 dtype、任意 device 的组合分发
  std::unordered_map<
      std::string,                          // 算子名
      std::unordered_map<
          DeviceType,
          std::unordered_map<
              DType,
              std::function<void(...)>      // 每个 function 有堆分配
          >
      >
  > global_registry;

  // 几百个算子 × 多种 dtype × 多种 device = 几千个条目
  // 字符串 key + unordered_map 桶 + function 包装 = 不少内存

  端侧做法 — LLM 只需要固定几个算子：

  // LLM 推理总共就这几个算子，直接函数指针数组
  typedef void (*KernelFn)(void* args);

  enum OpType : uint8_t {
      OP_EMBED,
      OP_RMSNORM,
      OP_MATMUL,        // QKV 投影、FFN 都是它
      OP_ROPE,
      OP_SOFTMAX,
      OP_SILU_MUL,
      OP_ADD,
      OP_COUNT           // = 7
  };

  // 分发表：一个 7 元素的数组
  KernelFn dispatch[OP_COUNT] = {
      kernel_embed,
      kernel_rmsnorm,
      kernel_matmul_int4,
      kernel_rope,
      kernel_softmax,
      kernel_silu_mul,
      kernel_add,
  };

  // 内存开销: 7 × 8 bytes = 56 bytes

  如果用前面说的硬编码执行流程，连这个分发表都不需要。

  ---
  4. Tensor 元数据

  每个中间结果都需要描述自己的 shape、dtype、指针：

  重型做法：

  class Tensor {
      std::vector<int64_t> shape;    // 堆分配
      std::vector<int64_t> stride;   // 堆分配
      std::shared_ptr<Storage> storage;  // 引用计数、control block
      Device device;
      DType dtype;
      bool requires_grad;
      AutogradMeta* grad_fn;         // 推理根本不需要
      // ...
  };
  // sizeof ≈ 100-200 bytes，还有间接堆分配
  // PyTorch 的 TensorImpl 实际上更大

  端侧做法：

  struct Tensor {
      void*    data;       // 8 bytes，直接指向 pool + offset
      uint32_t shape[4];   // 16 bytes，LLM 最多 4 维够了
      uint16_t stride[4];  // 8 bytes
      uint8_t  ndim;       // 1 byte
      uint8_t  dtype;      // 1 byte
  };
  // sizeof = 34 bytes，零堆分配

  // 甚至可以更极端：推理时 shape 是已知的，不需要存
  // 直接用裸指针 + 调用时传 shape 参数
  void matmul(void* A, void* B, void* C, int M, int N, int K);

  ---
  5. 调度器

  要不要线程池？

  端侧推理的特点：
    - GPU kernel 是异步提交的，CPU 只负责发射
    - batch_size=1，没有请求级并发
    - 单 stream 串行执行就够了

  结论：不需要复杂调度器

  // 极简调度：一个 for 循环
  void run_inference(Model* m, ...) {
      for (int l = 0; l < n_layers; l++) {
          launch_attn_kernel<<<grid, block, 0, stream>>>(args);
          launch_ffn_kernel<<<grid, block, 0, stream>>>(args);
      }
      cudaStreamSynchronize(stream);
  }

  // 如果需要 CPU-GPU 重叠（比如边 decode 边做 tokenize），
  // 最多两个线程就够了，不需要线程池

  ---
  总结对比

  ┌─────────────┬────────────────────────────────┬──────────────────────────┬──────────────────────┐
  │    组件     │          通用框架做法          │       端侧优化做法       │       省了什么       │
  ├─────────────┼────────────────────────────────┼──────────────────────────┼──────────────────────┤
  │ 计算图      │ 堆分配节点 + 指针图            │ 硬编码执行流程           │ 去掉图表示，省几 MB  │
  ├─────────────┼────────────────────────────────┼──────────────────────────┼──────────────────────┤
  │ 内存分配器  │ 通用 malloc + 空闲链表         │ 离线规划偏移量，一次分配 │ 去掉簿记和碎片       │
  ├─────────────┼────────────────────────────────┼──────────────────────────┼──────────────────────┤
  │ 算子分发    │ 字符串哈希表 + std::function   │ 枚举数组 / 直接调用      │ 去掉注册表开销       │
  ├─────────────┼────────────────────────────────┼──────────────────────────┼──────────────────────┤
  │ Tensor 描述 │ shared_ptr + vector + autograd │ 固定大小 struct / 裸指针 │ 去掉堆分配和引用计数 │
  ├─────────────┼────────────────────────────────┼──────────────────────────┼──────────────────────┤
  │ 调度器      │ 线程池 + 任务队列              │ 单线程 for 循环          │ 去掉线程栈和队列     │
  └─────────────┴────────────────────────────────┴──────────────────────────┴──────────────────────┘

  核心思路就一句话：LLM 推理的计算模式是完全静态可预测的，所以通用框架里那些为"灵活性"付出的动态开销（堆分配、哈希表、引
  用计数、图调度）在端侧全部可以去掉，换成编译期/加载期确定的静态结构。
