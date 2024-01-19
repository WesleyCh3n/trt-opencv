#pragma once

#include <filesystem>

#include "trt.hpp"

namespace dnn {

void letterbox(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output_image,
               const cv::Size &target_size);
cv::cuda::GpuMat blob_from_gpumat(const cv::cuda::GpuMat &input,
                                  const std::array<float, 3> &std,
                                  const std::array<float, 3> &mean, bool swapBR,
                                  bool normalize);
cv::cuda::GpuMat blob_from_gpumat(const std::vector<cv::cuda::GpuMat> &inputs,
                                  const std::array<float, 3> &std,
                                  const std::array<float, 3> &mean, bool swapBR,
                                  bool normalize);

struct Object {
  cv::Rect rect;
  float conf;
};
class Yolo {
  std::unique_ptr<trt::Engine> model_;
  std::vector<uint32_t> input_dim_;
  std::vector<uint32_t> output_dim_;
  std::vector<float> raw_output_;
  std::vector<std::vector<cv::Rect>> rects_;
  std::vector<std::vector<float>> confs_;
  std::vector<std::vector<int>> idxes_;

  std::vector<Object> post_process(float *raw_results,
                                   const cv::cuda::GpuMat &input,
                                   const float &confidence_threshold_,
                                   const float &nms_threshold_);

  std::vector<std::vector<Object>>
  post_process(float *raw_results, const std::vector<cv::cuda::GpuMat> &inputs,
               const float &confidence_threshold_, const float &nms_threshold_);

public:
  Yolo(std::filesystem::path model_path, const trt::EngineOption option);

  [[nodiscard]] std::vector<Object>
  predict(const cv::cuda::GpuMat &gmat,
          const float &confidence_threshold = 0.25,
          const float &nms_threshold = 0.45);

  [[nodiscard]] std::vector<std::vector<Object>>
  predict(const std::vector<cv::cuda::GpuMat> &gmats,
          const float &confidence_threshold = 0.25,
          const float &nms_threshold = 0.45);
};

class FeatureExtractor {
  std::unique_ptr<trt::Engine> model_;
  std::vector<uint32_t> input_dim_;
  std::vector<uint32_t> output_dim_;
  std::vector<float> raw_output_;

public:
  FeatureExtractor(std::filesystem::path model_path,
                   const trt::EngineOption option);
  std::vector<float> predict(const cv::cuda::GpuMat &gmat,
                             const std::array<float, 3> &std = {0.5, 0.5, 0.5},
                             const std::array<float, 3> &mean = {0.5, 0.5,
                                                                 0.5});
  std::vector<float> predict(const std::vector<cv::cuda::GpuMat> &gmat,
                             const std::array<float, 3> &std = {0.5, 0.5, 0.5},
                             const std::array<float, 3> &mean = {0.5, 0.5,
                                                                 0.5});
};

} // namespace dnn
