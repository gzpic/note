// [ADDED BEGIN] test-facing helper declaration added to llama-context.h
// Internal test helper: build the full Qwen3 execution graph through the normal
// llama_context -> llama_model::build_graph path, but expose a thin wrapper that
// tests can call directly.
//
// This intentionally reuses graph_reserve() so tests exercise the same end-to-end
// graph construction logic as normal execution without duplicating Qwen3-specific
// graph assembly code.
ggml_cgraph * llama_internal_build_qwen3_graph_for_test(
        struct llama_context * ctx,
        uint32_t n_tokens = 1,
        uint32_t n_seqs = 1,
        uint32_t n_outputs = 1,
        bool split_only = false,
        size_t * sizes = nullptr);

// [ADDED] helper method declared inside llama_context for internal tests
ggml_cgraph * graph_build_for_test(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, bool split_only = false, size_t * sizes = nullptr);
// [ADDED END]
