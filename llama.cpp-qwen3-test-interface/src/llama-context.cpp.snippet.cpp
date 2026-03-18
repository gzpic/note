// [ADDED BEGIN] helper implementation added to llama-context.cpp

ggml_cgraph * llama_context::graph_build_for_test(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, bool split_only, size_t * sizes) {
    auto mctx = memory ? memory->init_full() : nullptr;
    return graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get(), split_only, sizes);
}

ggml_cgraph * llama_internal_build_qwen3_graph_for_test(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs,
        bool split_only,
        size_t * sizes) {
    GGML_ASSERT(ctx != nullptr);

    const auto * model = llama_get_model(ctx);
    GGML_ASSERT(model != nullptr);
    GGML_ASSERT(model->arch == LLM_ARCH_QWEN3);

    return ctx->graph_build_for_test(n_tokens, n_seqs, n_outputs, split_only, sizes);
}

// [ADDED END]
