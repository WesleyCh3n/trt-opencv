#include "dnn.hpp"

#include <opencv2/cudaarithm.hpp>  // cv::cuda::split
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudawarping.hpp> // cv::cuda::resize
#include <opencv2/dnn.hpp>         // cv::dnn::NMSBoxes

// =============================================================================
// Utility functions
// =============================================================================
void dnn::letterbox(const cv::cuda::GpuMat &input,
                    cv::cuda::GpuMat &output_image,
                    const cv::Size &target_size) {
  float scale = std::min((float)target_size.width / input.cols,
                         (float)target_size.height / input.rows);
  cv::Size new_size =
      cv::Size(static_cast<int>(std::round(input.cols * scale)),
               static_cast<int>(std::round(input.rows * scale)));
  cv::cuda::resize(input, output_image, new_size, 0, 0, cv::INTER_AREA);
  float padh = (target_size.height - new_size.height) / 2.;
  float padw = (target_size.width - new_size.width) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  cv::cuda::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114.));
}

cv::cuda::GpuMat dnn::blob_from_gpumat(const cv::cuda::GpuMat &input,
                                       const std::array<float, 3> &std,
                                       const std::array<float, 3> &mean,
                                       bool swapBR, bool normalize) {
  if (swapBR)
    cv::cuda::cvtColor(input, input, cv::COLOR_BGR2RGB);
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

cv::cuda::GpuMat
dnn::blob_from_gpumat(const std::vector<cv::cuda::GpuMat> &inputs,
                      const std::array<float, 3> &std,
                      const std::array<float, 3> &mean, bool swapBR,
                      bool normalize) {
  if (swapBR) {
    for (uint32_t i = 0; i < (uint32_t)inputs.size(); i++) {
      cv::cuda::cvtColor(inputs[i], inputs[i], cv::COLOR_BGR2RGB);
    }
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

// =============================================================================
// Yolo Class
// =============================================================================
dnn::Yolo::Yolo(std::filesystem::path model_path,
                const trt::EngineOption option) {
  model_ = std::make_unique<trt::Engine>(model_path.string(), option);
  input_dim_ = model_->get_input_dims();
  output_dim_ = model_->get_output_dims();
}

std::vector<dnn::Object> dnn::Yolo::predict(const cv::cuda::GpuMat &gmat,
                                            const float &confidence_threshold,
                                            const float &nms_threshold) {
  cv::cuda::GpuMat input;
  letterbox(gmat, input,
            {static_cast<int>(input_dim_[2]), static_cast<int>(input_dim_[1])});
  auto blob = blob_from_gpumat(input,                         // input gpumats
                               std::array<float, 3>{1, 1, 1}, // std factor
                               std::array<float, 3>{0, 0, 0}, // mean
                               true, true);

  model_->run(blob, 1, raw_output_);
  return post_process(raw_output_.data(), gmat, confidence_threshold,
                      nms_threshold);
}

std::vector<std::vector<dnn::Object>>
dnn::Yolo::predict(const std::vector<cv::cuda::GpuMat> &gmats,
                   const float &confidence_threshold,
                   const float &nms_threshold) {
  uint32_t batch_size = gmats.size();
  std::vector<cv::cuda::GpuMat> inputs(batch_size);
  for (int i = 0; i < gmats.size(); i++) {
    letterbox(
        gmats[i], inputs[i],
        {static_cast<int>(input_dim_[2]), static_cast<int>(input_dim_[1])});
  }
  auto blob = blob_from_gpumat(inputs,                        // input gpumats
                               std::array<float, 3>{1, 1, 1}, // std factor
                               std::array<float, 3>{0, 0, 0}, // mean
                               true, true);
  model_->run(blob, batch_size, raw_output_);
  return post_process(raw_output_.data(), gmats, confidence_threshold,
                      nms_threshold);
}

void dnn::Yolo::predict(const std::vector<cv::cuda::GpuMat> &gmats,
                        std::vector<std::vector<cv::Rect>> &rects,
                        std::vector<std::vector<float>> &confs,
                        const float &confidence_threshold,
                        const float &nms_threshold) {
  uint32_t batch_size = gmats.size();
  std::vector<cv::cuda::GpuMat> inputs(batch_size);
  for (int i = 0; i < gmats.size(); i++) {
    letterbox(
        gmats[i], inputs[i],
        {static_cast<int>(input_dim_[2]), static_cast<int>(input_dim_[1])});
  }
  auto blob = blob_from_gpumat(inputs,                        // input gpumats
                               std::array<float, 3>{1, 1, 1}, // std factor
                               std::array<float, 3>{0, 0, 0}, // mean
                               true, true);
  model_->run(blob, batch_size, raw_output_);
  rects.clear();
  confs.clear();
  post_process(raw_output_.data(), gmats, confidence_threshold, nms_threshold,
               rects, confs);
}

void dnn::Yolo::post_process(float *raw_results, const cv::cuda::GpuMat &input,
                             const float &confidence_threshold_,
                             const float &nms_threshold_,
                             std::vector<cv::Rect> &rects,
                             std::vector<float> &confs) {

  if (output_dim_[0] >= 6) {
    throw std::runtime_error("xywhsc is not supported yet");
  }
  const float scale = std::min((float)input_dim_[2] / input.cols,
                               (float)input_dim_[1] / input.rows);
  const float padw = (std::round(input_dim_[2] - input.cols * scale) / 2 - 0.1);
  const float padh = (std::round(input_dim_[1] - input.rows * scale) / 2 - 0.1);
  std::vector<cv::Rect> rects_all;
  std::vector<float> confs_all;
  std::vector<int> idxes;
  for (int i = 0; i < output_dim_[1]; i++) {
    if (raw_results[4 * output_dim_[1] + i] > confidence_threshold_) {
      float xc = raw_results[i];
      float yc = raw_results[1 * output_dim_[1] + i];
      float dw = raw_results[2 * output_dim_[1] + i] / 2;
      float dh = raw_results[3 * output_dim_[1] + i] / 2;
      float x1 = xc - dw;
      float y1 = yc - dh;
      float x2 = xc + dw;
      float y2 = yc + dh;

      x1 = std::max((x1 - padw) / scale, (float)0.0);
      y1 = std::max((y1 - padh) / scale, (float)0.0);
      x2 = std::min((x2 - padw) / scale, (float)input.size().width);
      y2 = std::min((y2 - padh) / scale, (float)input.size().height);

      rects_all.emplace_back(
          cv::Rect(cv::Point(std::round(x1), std::round(y1)),
                   cv::Point(std::round(x2), std::round(y2))));
      confs_all.emplace_back(raw_results[4 * output_dim_[1] + i]);
    }
  }
  if (!rects_all.empty()) {
    cv::dnn::NMSBoxes(rects_all, confs_all, confidence_threshold_,
                      nms_threshold_, idxes);
  }
  for (auto &i : idxes) {
    rects.emplace_back(rects_all[i]);
    confs.emplace_back(confs_all[i]);
  }
}

void dnn::Yolo::post_process(float *raw_results,
                             const std::vector<cv::cuda::GpuMat> &inputs,
                             const float &confidence_threshold_,
                             const float &nms_threshold_,
                             std::vector<std::vector<cv::Rect>> &rects,
                             std::vector<std::vector<float>> &confs) {
  rects.resize(inputs.size());
  confs.resize(inputs.size());
  for (int b = 0; b < inputs.size(); b++) {
    post_process(raw_results + b * output_dim_[0] * output_dim_[1], inputs[b],
                 0.25, 0.45, rects[b], confs[b]);
  }
}

std::vector<dnn::Object>
dnn::Yolo::post_process(float *raw_results, const cv::cuda::GpuMat &input,
                        const float &confidence_threshold_,
                        const float &nms_threshold_) {
  if (output_dim_[0] >= 6) {
    throw std::runtime_error("xywhsc is not supported yet");
  }
  const float scale = std::min((float)input_dim_[2] / input.cols,
                               (float)input_dim_[1] / input.rows);
  const float padw = (std::round(input_dim_[2] - input.cols * scale) / 2 - 0.1);
  const float padh = (std::round(input_dim_[1] - input.rows * scale) / 2 - 0.1);
  std::vector<cv::Rect> rects;
  std::vector<float> confs;
  std::vector<int> idxes;
  for (int i = 0; i < output_dim_[1]; i++) {
    if (raw_results[4 * output_dim_[1] + i] > confidence_threshold_) {
      float xc = raw_results[i];
      float yc = raw_results[1 * output_dim_[1] + i];
      float dw = raw_results[2 * output_dim_[1] + i] / 2;
      float dh = raw_results[3 * output_dim_[1] + i] / 2;
      float x1 = xc - dw;
      float y1 = yc - dh;
      float x2 = xc + dw;
      float y2 = yc + dh;

      x1 = std::max((x1 - padw) / scale, (float)0.0);
      y1 = std::max((y1 - padh) / scale, (float)0.0);
      x2 = std::min((x2 - padw) / scale, (float)input.size().width);
      y2 = std::min((y2 - padh) / scale, (float)input.size().height);

      rects.emplace_back(cv::Rect(cv::Point(std::round(x1), std::round(y1)),
                                  cv::Point(std::round(x2), std::round(y2))));
      confs.emplace_back(raw_results[4 * output_dim_[1] + i]);
    }
  }
  if (!rects.empty()) {
    cv::dnn::NMSBoxes(rects, confs, confidence_threshold_, nms_threshold_,
                      idxes);
  }
  std::vector<Object> objs;
  for (auto &i : idxes) {
    objs.emplace_back(Object{rects[i], confs[i]});
  }
  return objs;
};

std::vector<std::vector<dnn::Object>> dnn::Yolo::post_process(
    float *raw_results, const std::vector<cv::cuda::GpuMat> &inputs,
    const float &confidence_threshold_, const float &nms_threshold_) {
  std::vector<std::vector<Object>> results(inputs.size());
  for (int b = 0; b < inputs.size(); b++) {
    results[b] = post_process(raw_results + b * output_dim_[0] * output_dim_[1],
                              inputs[b], 0.25, 0.45);
  }
  return results;
};

// =============================================================================
// FeatureExtractor
// =============================================================================

dnn::FeatureExtractor::FeatureExtractor(std::filesystem::path model_path,
                                        const trt::EngineOption option) {
  model_ = std::make_unique<trt::Engine>(model_path.string(), option);
  input_dim_ = model_->get_input_dims();
  output_dim_ = model_->get_output_dims();
}

std::vector<float>
dnn::FeatureExtractor::predict(const cv::cuda::GpuMat &gmat,
                               const std::array<float, 3> &std,
                               const std::array<float, 3> &mean) {
  cv::cuda::GpuMat resized;
  cv::cuda::resize(gmat, resized, cv::Size(input_dim_[1], input_dim_[2]), 0, 0,
                   cv::INTER_AREA);
  auto blob = blob_from_gpumat(resized, std, mean, true, true);
  model_->run(blob, 1, raw_output_);
  return std::move(raw_output_);
}

std::vector<float>
dnn::FeatureExtractor::predict(const std::vector<cv::cuda::GpuMat> &gmats,
                               const std::array<float, 3> &std,
                               const std::array<float, 3> &mean) {
  std::vector<cv::cuda::GpuMat> resized(gmats.size());
  for (int i = 0; i < gmats.size(); i++) {
    cv::cuda::resize(gmats[i], resized[i],
                     cv::Size(input_dim_[1], input_dim_[2]), 0, 0,
                     cv::INTER_AREA);
  }

  auto blob = blob_from_gpumat(resized, std, mean, true, true);

  model_->run(blob, gmats.size(), raw_output_);
  return std::move(raw_output_);
}
