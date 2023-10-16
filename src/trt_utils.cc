#include "trt.hpp"

#include <opencv2/cudaarithm.hpp>  // cv::cuda::split
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudawarping.hpp> // cv::cuda::resize
#include <opencv2/dnn.hpp>         // cv::dnn::NMSBoxes
#include <spdlog/spdlog.h>
using namespace trt::utils;

//==============================================================================
// blob_from_gpumats
// =============================================================================

cv::cuda::GpuMat trt::utils::blob_from_gpumat(
    cv::cuda::GpuMat &input, const std::array<uint32_t, 2> &size,
    const std::array<float, 3> &std, const std::array<float, 3> &mean,
    bool swapBR, bool normalize) {
  if (swapBR)
    cv::cuda::cvtColor(input, input, cv::COLOR_BGR2RGB);
  cv::cuda::resize(input, input, cv::Size(size[0], size[1]), 0, 0,
                   cv::INTER_LINEAR);
  cv::cuda::GpuMat blob(1, input.rows * input.cols, CV_8UC3);
  size_t continuous_length = input.rows * input.cols;
  std::vector<cv::cuda::GpuMat> rgb{
      cv::cuda::GpuMat(input.rows, input.cols, CV_8U, &(blob.ptr()[0])),
      cv::cuda::GpuMat(input.rows, input.cols, CV_8U,
                       &(blob.ptr()[continuous_length])),
      cv::cuda::GpuMat(input.rows, input.cols, CV_8U,
                       &(blob.ptr()[continuous_length * 2])),
  };
  cv::cuda::split(input, rgb);
  if (normalize) {
    blob.convertTo(blob, CV_32FC3, 1.f / 255.f);
  } else {
    blob.convertTo(blob, CV_32FC3);
  }
  cv::cuda::subtract(blob, cv::Scalar(mean[0], mean[1], mean[2]), blob,
                     cv::noArray(), -1);
  cv::cuda::divide(blob, cv::Scalar(std[0], std[1], std[2]), blob, 1, -1);
  return blob;
}

cv::cuda::GpuMat trt::utils::blob_from_gpumats(
    std::vector<cv::cuda::GpuMat> &inputs, const std::array<uint32_t, 2> &size,
    const std::array<float, 3> &std, const std::array<float, 3> &mean,
    bool swapBR, bool normalize) {
  for (uint32_t i = 0; i < (uint32_t)inputs.size(); i++) {
    if (swapBR)
      cv::cuda::cvtColor(inputs[i], inputs[i], cv::COLOR_BGR2RGB);
    cv::cuda::resize(inputs[i], inputs[i], cv::Size(size[0], size[1]), 0, 0,
                     cv::INTER_LINEAR);
  }
  cv::cuda::GpuMat blob(1, inputs[0].rows * inputs[0].cols * inputs.size(),
                        CV_8UC3);
  size_t continuous_length = inputs[0].rows * inputs[0].cols;
  for (uint32_t i = 0; i < (uint32_t)inputs.size(); i++) {
    std::vector<cv::cuda::GpuMat> rgb{
        cv::cuda::GpuMat(inputs[0].rows, inputs[0].cols, CV_8U,
                         &(blob.ptr()[0 + continuous_length * 3 * i])),
        cv::cuda::GpuMat(
            inputs[0].rows, inputs[0].cols, CV_8U,
            &(blob.ptr()[continuous_length + continuous_length * 3 * i])),
        cv::cuda::GpuMat(
            inputs[0].rows, inputs[0].cols, CV_8U,
            &(blob.ptr()[continuous_length * 2 + continuous_length * 3 * i])),
    };
    cv::cuda::split(inputs[i], rgb);
  }
  if (normalize) {
    blob.convertTo(blob, CV_32FC3, 1.f / 255.f);
  } else {
    blob.convertTo(blob, CV_32FC3);
  }
  cv::cuda::subtract(blob, cv::Scalar(mean[0], mean[1], mean[2]), blob,
                     cv::noArray(), -1);
  cv::cuda::divide(blob, cv::Scalar(std[0], std[1], std[2]), blob, 1, -1);
  return blob;
}

//==============================================================================
// YoloUtility
// =============================================================================
YoloUtility YoloUtility::create() { return YoloUtility(); };

YoloUtility &YoloUtility::set_original_size(const cv::Size &size) {
  original_size_ = size;
  return *this;
}

YoloUtility &YoloUtility::set_input_size(const cv::Size &size) {
  input_size_ = size;
  return *this;
}

YoloUtility &YoloUtility::set_conf_threshold(const float &threshold) {
  confidence_threshold_ = threshold;
  return *this;
}

YoloUtility &YoloUtility::set_nms_threshold(const float &threshold) {
  nms_threshold_ = threshold;
  return *this;
}

YoloUtility &YoloUtility::is_xywhs() {
  bbox_dim_ = 5;
  return *this;
}

YoloUtility &YoloUtility::is_xywhsc() {
  bbox_dim_ = 6;
  return *this;
};

YoloUtility &YoloUtility::set_num_bbox(const uint32_t &num_bbbox) {
  num_bbbox_ = num_bbbox;
  return *this;
}

void YoloUtility::show() {
  spdlog::info("[yolo] original w: {} h: {}", original_size_.width,
               original_size_.height);
  spdlog::info("[yolo] input w: {} h: {}", input_size_.width,
               input_size_.height);
  spdlog::info("[yolo] output format: {}", bbox_dim_ == 5 ? "xywhs" : "xywhsc");
  spdlog::info("[yolo] num bbox: {}", num_bbbox_);
  spdlog::info("[yolo] conf threshold: {}", confidence_threshold_);
  spdlog::info("[yolo] nms threshold: {}", nms_threshold_);
}

void trt::utils::YoloUtility::post_process(float *raw_results,
                                           std::vector<cv::Rect> &rects,
                                           std::vector<float> &confs,
                                           std::vector<int> &idxes) {
  if (bbox_dim_ >= 6) {
    throw std::runtime_error("xywhsc is not supported yet");
  }
  // ---num_bbbox_---
  // [xc..........xc] [yc..........yc] [w............w] [h............h]
  // [s............s]
  for (int i = 0; i < num_bbbox_; i++) {
    if (raw_results[4 * num_bbbox_ + i] > confidence_threshold_) {
      uint32_t xc =
          std::round(raw_results[i] / input_size_.width * original_size_.width);
      uint32_t yc = std::round(raw_results[1 * num_bbbox_ + i] /
                               input_size_.height * original_size_.height);
      uint32_t dw = std::round(raw_results[2 * num_bbbox_ + i] / 2 /
                               input_size_.width * original_size_.width);
      uint32_t dh = std::round(raw_results[3 * num_bbbox_ + i] / 2 /
                               input_size_.height * original_size_.height);

      uint32_t tlx = std::max(0, (int)(xc - dw));
      uint32_t tly = std::max(0, (int)(yc - dh));
      uint32_t brx = tlx + dw * 2 < original_size_.width ? tlx + dw * 2
                                                         : original_size_.width;
      uint32_t bry = tly + dh * 2 < original_size_.height
                         ? tly + dh * 2
                         : original_size_.height;
      rects.emplace_back(cv::Rect(cv::Point(tlx, tly), cv::Point(brx, bry)));
      confs.emplace_back(raw_results[4 * num_bbbox_ + i]);
    }
  }
  if (!rects.empty()) {
    cv::dnn::NMSBoxes(rects, confs, confidence_threshold_, nms_threshold_,
                      idxes);
  }
};

void trt::utils::YoloUtility::batch_post_process(
    std::vector<float> &raw_results, const uint32_t batch_size,
    std::vector<std::vector<cv::Rect>> &rects,
    std::vector<std::vector<float>> &confs,
    std::vector<std::vector<int>> &idxes) {
#pragma omp parallel for
  for (int b = 0; b < batch_size; b++) {
    rects[b].clear();
    confs[b].clear();
    idxes[b].clear();
    this->post_process(raw_results.data() + b * num_bbbox_ * bbox_dim_,
                       rects[b], confs[b], idxes[b]);
  }
};
