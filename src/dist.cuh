#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dnn {
namespace cuda {
__global__ void euclidean_dist(float *A, float *B, float *result, int n, int m,
                               int l);
__global__ void cos_dist(float *A, float *B, float *result, int n, int m,
                         int l);
std::vector<float> dist(std::vector<float> va, std::vector<float> vb,
                        const uint64_t vlen);
std::vector<float> dist(std::vector<float> va, float *devb,
                        const uint64_t devbsize, const uint64_t vlen);

} // namespace cuda
} // namespace dnn
