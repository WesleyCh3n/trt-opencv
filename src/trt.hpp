#ifndef __TRT_HPP__
#define __TRT_HPP__

#include <NvInfer.h>
#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include <torch/types.h>

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

  torch::Tensor run(cv::cuda::GpuMat &flatten_inputs,
                    const uint32_t &batch_size);

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

} // namespace trt

#endif
