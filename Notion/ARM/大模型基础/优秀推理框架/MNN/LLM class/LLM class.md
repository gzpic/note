---

---
```c++
public:
    enum Stage {
        Prefill,
        Decode
    };
    static Llm* createLLM(const std::string& config_path);
    static void destroy(Llm* llm);// For Windows RT mode should use destroy
    Llm(std::shared_ptr<LlmConfig> config);
    virtual ~Llm();
    virtual bool load();
    virtual Express::VARP gen_attention_mask(int seq_len);
    virtual Express::VARP gen_position_ids(int seq_len);
    virtual Express::VARP embedding(const std::vector<int>& input_ids);
    virtual int sample(Express::VARP logits, int offset = 0, int size = 0); // 决定下一个token
    std::vector<Express::VARP> getOutputs() const;
    int getOutputIndex(const std::string& name) const;
    void reset();
    void tuning(TuneType type, std::vector<int> candidates);
    virtual std::vector<Express::VARP> forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs = {});
    Express::VARP forward(const std::vector<int>& input_ids, bool is_prefill = true);
    Express::VARP forward(MNN::Express::VARP input_embeds);
    void switchMode(Stage stage);
    void setKVCacheInfo(size_t add, size_t remove, int* reserve = nullptr, int n_reserve = 0);
    size_t getCurrentHistory() const;
    void eraseHistory(size_t begin, size_t end);
    virtual void response(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    void response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    void response(const ChatMessages& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    void response(MNN::Express::VARP input_embeds, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    virtual void generate_init(std::ostream* os = nullptr, const char* end_with = nullptr);
    void generate(int max_token);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    std::vector<int> generate(MNN::Express::VARP input_embeds, int max_tokens = -1);
    bool stoped();
    bool reuse_kv();
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    Llm* create_lora(const std::string& lora_path);
    // tokenier function
    bool is_stop(int token);
    std::string tokenizer_decode(int token);
    virtual std::vector<int> tokenizer_encode(const std::string& query);
    friend class Pipeline;
    virtual std::vector<int> tokenizer_encode(const MultimodalPrompt& multimodal_input);
    // ptompt functions
    std::string apply_chat_template(const std::string& user_content) const;
    std::string apply_chat_template(const ChatMessages& chat_prompts) const;
    void response(const MultimodalPrompt& multimodal_input,
                  std::ostream* os = &std::cout,
                  const char* end_with = nullptr,
                  int max_new_tokens = -1);
    const LlmContext* getContext() const {
        return mContext.get();
    }
    virtual void setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) {}
    virtual void generateWavform() {}
protected:
    void initRuntime();
    void setRuntimeHint(std::shared_ptr<Express::Executor::RuntimeManager> &rtg);
    std::shared_ptr<LlmContext> mContext;
    std::shared_ptr<KVMeta> mMeta;
    std::shared_ptr<LlmConfig> mConfig;
    std::shared_ptr<Prompt> mPrompt;
    std::shared_ptr<Tokenizer> mTokenizer;
    std::shared_ptr<DiskEmbedding> mDiskEmbedding;
    std::shared_ptr<Sampler> mSampler;
    std::shared_ptr<Express::Executor::RuntimeManager> mRuntimeManager, mProcessorRuntimeManager;
    std::shared_ptr<Express::Module> mModule;
    /**
     key: <seq_len, all_logists>
     value : module
     note: prefill share one module, seq_len = 100 for example
     */
    const int mPrefillKey = 100;
    std::map<std::pair<int, bool>, std::shared_ptr<Express::Module>> mModulePool;
    const Express::Module* mBaseModule = nullptr;
    Express::VARP inputsEmbeds, attentionMask, positionIds;
    std::vector<Express::VARP> mAttentionMaskVarVec, mPositionIdsVarVec;
    Express::VARP logitsAllIdx, logitsLastIdx;
    int mSeqLenIndex = 0;
protected:
    friend class ArGeneration;
    friend class LookaheadGeneration;
    friend class MtpGeneration;
    friend class EagleGeneration;
    std::vector<Express::VARP> forwardVec(const std::vector<int>& input_ids);
    std::vector<Express::VARP> forwardVec(MNN::Express::VARP input_embeds);
private:
    std::shared_ptr<Generation> mGenerationStrategy;
    void setSpeculativeConfig();
    void updateContext(int seq_len, int gen_len);

private:
    bool mInSpec = false;
    int mDraftLength = 4;
    std::shared_ptr<GenerationParams> mGenerateParam;
    bool mAsync = true;
    int mBlockSize = 0;
    std::vector<int> mValidBlockSize;
};
```

[[sample]]