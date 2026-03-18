#include "llama.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// [ADDED BEGIN] new standalone test program for Qwen3 full graph build
int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen3-model.gguf>\n", argv[0]);
        return EXIT_FAILURE;
    }

    llama_backend_init();

    int exit_code = EXIT_FAILURE;

    try {
        llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap = false;

        llama_model * model = llama_model_load_from_file(argv[1], mparams);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model");
        }

        if (model->arch != LLM_ARCH_QWEN3) {
            throw std::runtime_error("model is not Qwen3");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("failed to create context");
        }

        ggml_cgraph * gf = llama_internal_build_qwen3_graph_for_test(ctx, 1, 1, 1);
        if (gf == nullptr) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("failed to build qwen3 graph");
        }

        std::fprintf(stderr, "qwen3 graph built successfully\n");
        (void) gf;

        llama_free(ctx);
        llama_model_free(model);
        exit_code = EXIT_SUCCESS;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
    }

    llama_backend_free();
    return exit_code;
}
// [ADDED END]
