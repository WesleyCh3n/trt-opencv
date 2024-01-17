#pragma once

#include <filesystem>
#include <opencv2/cudaarithm.hpp>  // cv::cuda::split
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudawarping.hpp> // cv::cuda::resize
#include <opencv2/dnn.hpp>         // cv::dnn::NMSBoxes

#include "trt.hpp"

namespace dnn {

void letterbox(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output_image,
               const cv::Size &new_size, const cv::Size &target_size);
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
                                   const float &confidence_threshold_,
                                   const float &nms_threshold_,
                                   const float &scale, const cv::Size &pad,
                                   const uint32_t img_cols,
                                   const uint32_t img_rows);

  std::vector<std::vector<Object>>
  post_process(float *raw_results, const uint32_t batch_size,
               const float &confidence_threshold_, const float &nms_threshold_,
               const float &scale, const cv::Size &pad, const uint32_t img_cols,
               const uint32_t img_rows);

public:
  Yolo(std::filesystem::path model_path, const uint32_t max_batch_size);

  std::vector<Object> predict(const cv::cuda::GpuMat &gmat,
                              const float &confidence_threshold = 0.25,
                              const float &nms_threshold = 0.45);

  std::vector<std::vector<Object>>
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
                   const uint32_t max_batch_size);
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
