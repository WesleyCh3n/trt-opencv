#ifndef __TRT_HPP__
#define __TRT_HPP__

#include <NvInfer.h>
#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>

namespace trt {

class CudaException : public std::exception {
  std::string msg_;

public:
  CudaException(const std::string &msg) : msg_(msg) {}
  char const *what() const noexcept override { return msg_.c_str(); }
};

/** @brief check cuda error */
inline void check_cuda_err(cudaError_t code) {
  if (code != cudaSuccess) {
    throw CudaException(fmt::format("[{}] {}: {}", (int)code,
                                    cudaGetErrorName(code),
                                    cudaGetErrorString(code)));
  }
}

class EngineException : public std::exception {
  std::string msg_;

public:
  EngineException(const std::string &msg) : msg_(msg) {}
  char const *what() const noexcept override { return msg_.c_str(); }
};

/** @brief trt logger */
class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override;
};

/** @brief engine option */
struct EngineOption {
  uint32_t max_batch_size;
};

class Engine {
  EngineOption option_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<void *> io_tensors_buf_;
  std::vector<const char *> io_tensors_name_;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
  uint64_t single_output_len_;

public:
  /** @brief construct a trt engine from a model file.
   * @param model_path model file path
   * @param option engine option
   */
  Engine(const std::string_view model_path, const EngineOption &option);
  Engine() = delete;
  ~Engine();

  /** @brief inference a flatten batch of cv::cuda::GpuMat, return a vector of
   * float. User should reorder output yourself.
   * @param flatten_inputs flatten batch of cv::cuda::GpuMat
   * @param batch_size number of batch
   * @param outputs output vector of float
   */
  void run(cv::cuda::GpuMat &flatten_inputs, const uint32_t &batch_size,
           std::vector<float> &outputs);

  /** @brief get input dims. omitting batch size, so if input size is {-1, 3,
   * 224, 224}, the return vector will be {3, 224, 224}.
   * @return vector of dimension
   */
  std::vector<uint32_t> get_input_dims();

  /** @brief get output dims. omitting batch size, so if output size is {-1,
   * 512}, the return vector will be {512}.
   * @return vector of dimension
   */
  std::vector<uint32_t> get_output_dims();

  /** @brief run model inference with cv::Mat??
   * @param inputs input
   * @param outputs output
   */
  [[maybe_unused]] void run(std::vector<cv::Mat> &inputs,
                            std::vector<std::vector<float>> &outputs){};
};

namespace utils {

cv::cuda::GpuMat blob_from_gpumat(cv::cuda::GpuMat &inputs,
                                  const std::array<uint32_t, 2> &size,
                                  const std::array<float, 3> &std,
                                  const std::array<float, 3> &mean,
                                  bool swapBR = true, bool normalize = true);

/** @brief preprocess batch of cv::cuda::GpuMat and change the order from
 * [B,H,W,C] to [B,C,H,W] then export to 1D continuous GpuMat.
 *
 * @param inputs batch of cv::cuda::GpuMat
 * @param size resized image
 * @param std image std
 * @param mean image mean
 * @param swapBR whether swap BGR to RGB
 * @param normalize whether normalize to [0, 1]
 * @return cv::cuda::GpuMat 1D GpuMat
 *
 * @note if inputs dimension is `[2, 1280, 720, 3]: CV_8UC3`, resized to
 * `[640, 640]`, first function will resize images to `[2, 640, 640, 3]`, then
 * reorder channel to `[2, 3, 640, 640]`. finally, output should be 1D GpuMat
 * `[1, 2 * 640 * 640 ]: CV_32FC3`
 */
cv::cuda::GpuMat blob_from_gpumats(std::vector<cv::cuda::GpuMat> &inputs,
                                   const std::array<uint32_t, 2> &size,
                                   const std::array<float, 3> &std,
                                   const std::array<float, 3> &mean,
                                   bool swapBR = true, bool normalize = true);

class YoloUtility {
  cv::Size original_size_;
  cv::Size input_size_;
  float confidence_threshold_;
  float nms_threshold_ = 0.5;
  uint32_t bbox_dim_ = 5;
  uint32_t num_bbbox_ = 8400;

public:
  YoloUtility() = default;
  ~YoloUtility() = default;

  /** @brief create yolo utility instance with builder pattern */
  static YoloUtility create();

  /** @brief set original image size
   * @param size 2d image size
   * @return *this
   */
  YoloUtility &set_original_size(const cv::Size &size);

  /** @brief set model image size
   * @param size 2d image size
   * @return *this
   */
  YoloUtility &set_input_size(const cv::Size &size);

  /** @brief set confidence threshold
   * @param threshold confidence threshold
   * @return *this
   */
  YoloUtility &set_conf_threshold(const float &threshold);

  /** @brief set nms threshold
   * @param threshold nms threshold
   * @return *this
   */
  YoloUtility &set_nms_threshold(const float &threshold);

  /** @brief set maximun number of bounding box per output
   * @param num_bbbox number of bounding box
   * @return *this
   */
  YoloUtility &set_num_bbox(const uint32_t &num_bbbox);

  /** @brief set whether output is xywhs (single class only)
   * @return *this
   */
  YoloUtility &is_xywhs();

  /** @brief set whether output is xywhsc (multi class)
   * @return *this
   */
  YoloUtility &is_xywhsc();

  /** @brief show yolo utility info */
  void show();

  /** @brief post process single image raw results
   * @param raw_results floats of array
   * @param rects output of results bounding boxes
   * @param confs output of results confidence
   * @param idxes output of selected results index after non-maximum suppression
   */
  void post_process(float *raw_results, std::vector<cv::Rect> &rects,
                    std::vector<float> &confs, std::vector<int> &idxes);

  /** @brief post process batch images raw results
   * @param raw_results floats of array
   * @param batch_size batch size
   * @param rects 2d output of results bounding boxes
   * @param confs 2d output of results confidence
   * @param idxes 2d output of selected results index after non-maximum
   * suppression
   *
   * note: rects.size() == confs.size() == idxes.size() == batch_size
   */
  void batch_post_process(std::vector<float> &raw_results,
                          const uint32_t batch_size,
                          std::vector<std::vector<cv::Rect>> &rects,
                          std::vector<std::vector<float>> &confs,
                          std::vector<std::vector<int>> &idxes);
};
} // namespace utils

} // namespace trt

#endif
