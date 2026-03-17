# Codex GEMM Trial

This is a small CUDA FP16 GEMM trial project aimed at Jetson Orin Nano.

It includes:

- a naive kernel
- a simple shared-memory tiled kernel
- a CPU reference check
- a lightweight benchmark

The kernels use row-major storage:

`C[M, N] = A[M, K] x B[K, N]`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Default size is `1024 x 1024 x 1024`:

```bash
./build/orin_nano_gemm_codex
```

You can also pass `M N K iters`:

```bash
./build/orin_nano_gemm_codex 512 1024 768 20
```

## Notes

- This version is intentionally conservative and easy to read.
- It does not use Tensor Cores, WMMA, or cuBLAS.
- The tiled kernel accumulates in `float` and writes back to `half`.
