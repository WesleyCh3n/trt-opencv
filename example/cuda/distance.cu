#include "dist.cuh"
#include "fmt/core.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  int len = 5;
  std::vector<float> va = {1.0,  2.0,  3.0,  4.0,  5.0,  //
                           6.0,  7.0,  8.0,  9.0,  10.0, //
                           11.0, 12.0, 13.0, 14.0, 15.0};
  std::vector<float> vb = {10.0, 11.0, 12.0, 13.0, 14.0, //
                           15.0, 16.0, 17.0, 18.0, 19.0, //
                           20.0, 21.0, 22.0, 23.0, 24.0, //
                           25.0, 26.0, 27.0, 28.0, 29.0};
  float *devB;
  cudaMalloc((void **)&devB, vb.size() * sizeof(float));
  cudaMemcpy(devB, vb.data(), vb.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // auto res = dnn::cuda::dist(va, vb, len);
  auto res = dnn::cuda::dist(va, devB, vb.size(), len);
  cudaFree(devB);
  for (int i = 0; i < va.size() / len; ++i) {
    for (int j = 0; j < vb.size() / len; ++j) {
      fmt::println("Distance between vector {} and vector {}: {}", i, j,
                   res[i * vb.size() / len + j]);
    }
  }
  return 0;
}
