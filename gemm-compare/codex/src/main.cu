#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;

inline void checkCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

__global__ void hgemmNaiveKernel(const half* a, const half* b, half* c, int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  c[row * n + col] = __float2half_rn(acc);
}

__global__ void hgemmTiledKernel(const half* a, const half* b, half* c, int m, int n, int k) {
  __shared__ half tileA[kTile][kTile];
  __shared__ half tileB[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * kTile + tx;
  const int row = blockIdx.y * kTile + ty;

  float acc = 0.0f;
  for (int tileK = 0; tileK < k; tileK += kTile) {
    const int aCol = tileK + tx;
    const int bRow = tileK + ty;

    tileA[ty][tx] = (row < m && aCol < k) ? a[row * k + aCol] : __float2half(0.0f);
    tileB[ty][tx] = (bRow < k && col < n) ? b[bRow * n + col] : __float2half(0.0f);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk) {
      acc += __half2float(tileA[ty][kk]) * __half2float(tileB[kk][tx]);
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = __float2half_rn(acc);
  }
}

std::vector<float> cpuReference(const std::vector<half>& a,
                                const std::vector<half>& b,
                                int m,
                                int n,
                                int k) {
  std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
      }
      out[row * n + col] = acc;
    }
  }
  return out;
}

float maxError(const std::vector<half>& gpu, const std::vector<float>& ref) {
  float maxErr = 0.0f;
  for (size_t i = 0; i < gpu.size(); ++i) {
    maxErr = std::max(maxErr, std::fabs(__half2float(gpu[i]) - ref[i]));
  }
  return maxErr;
}

template <typename Kernel>
float runBenchmark(Kernel kernel,
                   const half* dA,
                   const half* dB,
                   half* dC,
                   int m,
                   int n,
                   int k,
                   int iters) {
  dim3 block(kTile, kTile);
  dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);

  for (int i = 0; i < 3; ++i) {
    kernel<<<grid, block>>>(dA, dB, dC, m, n, k);
  }
  checkCuda(cudaGetLastError(), "warmup launch");
  checkCuda(cudaDeviceSynchronize(), "warmup sync");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    kernel<<<grid, block>>>(dA, dB, dC, m, n, k);
  }
  checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  checkCuda(cudaGetLastError(), "benchmark launch");

  float ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return ms / static_cast<float>(iters);
}

void printResult(const std::string& name, float ms, int m, int n, int k, float err) {
  const double flops = 2.0 * static_cast<double>(m) * n * k;
  const double gflops = flops / (static_cast<double>(ms) * 1.0e6);
  std::cout << std::left << std::setw(10) << name
            << " time=" << std::fixed << std::setprecision(3) << ms << " ms"
            << " gflops=" << std::setprecision(2) << gflops
            << " max_err=" << std::setprecision(6) << err << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const int m = (argc > 1) ? std::stoi(argv[1]) : 1024;
  const int n = (argc > 2) ? std::stoi(argv[2]) : 1024;
  const int k = (argc > 3) ? std::stoi(argv[3]) : 1024;
  const int iters = (argc > 4) ? std::stoi(argv[4]) : 10;

  std::cout << "Jetson Orin Nano FP16 GEMM trial\n";
  std::cout << "M=" << m << " N=" << n << " K=" << k << " iters=" << iters << '\n';

  const size_t sizeA = static_cast<size_t>(m) * k;
  const size_t sizeB = static_cast<size_t>(k) * n;
  const size_t sizeC = static_cast<size_t>(m) * n;

  std::vector<half> hA(sizeA);
  std::vector<half> hB(sizeB);
  std::vector<half> hCNaive(sizeC);
  std::vector<half> hCTiled(sizeC);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (half& v : hA) {
    v = __float2half_rn(dist(rng));
  }
  for (half& v : hB) {
    v = __float2half_rn(dist(rng));
  }

  const auto hRef = cpuReference(hA, hB, m, n, k);

  half* dA = nullptr;
  half* dB = nullptr;
  half* dC = nullptr;
  checkCuda(cudaMalloc(&dA, sizeA * sizeof(half)), "cudaMalloc(dA)");
  checkCuda(cudaMalloc(&dB, sizeB * sizeof(half)), "cudaMalloc(dB)");
  checkCuda(cudaMalloc(&dC, sizeC * sizeof(half)), "cudaMalloc(dC)");

  checkCuda(cudaMemcpy(dA, hA.data(), sizeA * sizeof(half), cudaMemcpyHostToDevice), "copy A");
  checkCuda(cudaMemcpy(dB, hB.data(), sizeB * sizeof(half), cudaMemcpyHostToDevice), "copy B");

  const float naiveMs = runBenchmark(hgemmNaiveKernel, dA, dB, dC, m, n, k, iters);
  checkCuda(cudaMemcpy(hCNaive.data(), dC, sizeC * sizeof(half), cudaMemcpyDeviceToHost), "copy C naive");
  const float naiveErr = maxError(hCNaive, hRef);

  const float tiledMs = runBenchmark(hgemmTiledKernel, dA, dB, dC, m, n, k, iters);
  checkCuda(cudaMemcpy(hCTiled.data(), dC, sizeC * sizeof(half), cudaMemcpyDeviceToHost), "copy C tiled");
  const float tiledErr = maxError(hCTiled, hRef);

  printResult("naive", naiveMs, m, n, k, naiveErr);
  printResult("tiled", tiledMs, m, n, k, tiledErr);

  checkCuda(cudaFree(dA), "cudaFree(dA)");
  checkCuda(cudaFree(dB), "cudaFree(dB)");
  checkCuda(cudaFree(dC), "cudaFree(dC)");
  return 0;
}
