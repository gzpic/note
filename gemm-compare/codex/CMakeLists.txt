cmake_minimum_required(VERSION 3.20)
project(orin_nano_gemm_codex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA arch for Jetson Orin Nano")

add_executable(orin_nano_gemm_codex src/main.cu)

target_compile_options(orin_nano_gemm_codex PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
