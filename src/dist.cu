#include "dist.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void dnn::cuda::euclidean_dist(float *A, float *B, float *result,
                                          int n, int m, int l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < m) {
    float sum = 0.0f;
    for (int k = 0; k < l; ++k) {
      float diff = A[i * l + k] - B[j * l + k];
      sum += diff * diff;
    }
    result[i * m + j] = sqrtf(sum);
  }
}

__global__ void dnn::cuda::cos_dist(float *A, float *B, float *result, int n,
                                    int m, int l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= m) {
    return;
  }
  float dotProduct = 0.0f;
  float magnitudeA = 0.0f;
  float magnitudeB = 0.0f;

  for (int k = 0; k < l; ++k) {
    dotProduct += A[i * l + k] * B[j * l + k];
    magnitudeA += A[i * l + k] * A[i * l + k];
    magnitudeB += B[j * l + k] * B[j * l + k];
  }

  result[i * m + j] =
      1.0f - dotProduct / (sqrtf(magnitudeA) * sqrtf(magnitudeB));
}

std::vector<float> dnn::cuda::dist(std::vector<float> va, std::vector<float> vb,
                                   const uint64_t vlen) {
  const uint64_t alen = va.size() / vlen;
  const uint64_t blen = vb.size() / vlen;
  const uint64_t resultslen = alen * blen;

  float *devA, *devB, *devRes;
  cudaMalloc((void **)&devA, va.size() * sizeof(float));
  cudaMalloc((void **)&devB, vb.size() * sizeof(float));
  cudaMalloc((void **)&devRes, resultslen * sizeof(float));
  cudaMemcpy(devA, va.data(), va.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(devB, vb.data(), vb.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 threadPerBlock(16, 16);
  dim3 numBlocks((alen + threadPerBlock.x - 1) / threadPerBlock.x,
                 (blen + threadPerBlock.y - 1) / threadPerBlock.y);
  cos_dist<<<numBlocks, threadPerBlock>>>(devA, devB, devRes, alen, blen, vlen);

  std::vector<float> hostRes(resultslen);
  cudaMemcpy(hostRes.data(), devRes, hostRes.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devRes);

  return hostRes;
}

std::vector<float> dnn::cuda::dist(std::vector<float> va, float *devb,
                                   const uint64_t devbsize,
                                   const uint64_t vlen) {

  const uint64_t alen = va.size() / vlen;
  const uint64_t blen = devbsize / vlen;
  const uint64_t resultslen = alen * blen;

  float *devA, *devRes;
  cudaMalloc((void **)&devA, va.size() * sizeof(float));
  cudaMalloc((void **)&devRes, resultslen * sizeof(float));
  cudaMemcpy(devA, va.data(), va.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 threadPerBlock(16, 16);
  dim3 numBlocks((alen + threadPerBlock.x - 1) / threadPerBlock.x,
                 (blen + threadPerBlock.y - 1) / threadPerBlock.y);
  cos_dist<<<numBlocks, threadPerBlock>>>(devA, devb, devRes, alen, blen, vlen);

  std::vector<float> hostRes(resultslen);
  cudaMemcpy(hostRes.data(), devRes, hostRes.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(devA);
  cudaFree(devRes);

  return hostRes;
}
