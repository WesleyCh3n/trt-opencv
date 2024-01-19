#include "trt.hpp"

#include <fstream>
#include <numeric> //std::accumulate

//===================================================================================
// Engine
//===================================================================================

trt::Engine::Engine(const std::string_view model_path,
                    const EngineOption &option)
    : option_(option) {
  model_path_ = std::string(model_path);
  std::ifstream file(std::string(model_path), std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw EngineException("Unable to read model file");
  }

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(option_.logger));
  if (runtime_ == nullptr) {
    throw EngineException("Failed to create runtime");
  }
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (engine_ == nullptr) {
    throw EngineException("Failed to create engine");
  }
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  if (context_ == nullptr) {
    throw EngineException("Failed to create context");
  }

  int32_t num_io_tensors = engine_->getNbIOTensors();
  io_tensors_buf_.resize(num_io_tensors);
  io_tensors_name_.resize(num_io_tensors);

  cudaStream_t stream;
  check_cuda_err(cudaStreamCreate(&stream));

  option_.logger.log(
      nvinfer1::ILogger::Severity::kINFO,
      ("Number of IO tensors: " + std::to_string(num_io_tensors)).c_str());
  option_.logger.log(
      nvinfer1::ILogger::Severity::kINFO,
      ("Max batch size: " + std::to_string(option_.max_batch_size)).c_str());

  for (int i = 0; i < engine_->getNbIOTensors(); i++) {
    const char *tensor_name = engine_->getIOTensorName(i);
    io_tensors_name_[i] = tensor_name;
    const auto tensor_shape = engine_->getTensorShape(tensor_name);
    const auto tensor_type = engine_->getTensorIOMode(tensor_name);

    const auto tensor_max_bytes = std::accumulate(
        tensor_shape.d + 1, tensor_shape.d + tensor_shape.nbDims,
        sizeof(float) * option_.max_batch_size, std::multiplies<uint64_t>());

    if (tensor_type == nvinfer1::TensorIOMode::kINPUT) {
      input_dims_ = tensor_shape;
      check_cuda_err(
          cudaMallocAsync(&io_tensors_buf_[i], tensor_max_bytes, stream));
    } else if (tensor_type == nvinfer1::TensorIOMode::kOUTPUT) {
      output_dims_ = tensor_shape;
      single_output_len_ = std::accumulate(tensor_shape.d + 1,
                                           tensor_shape.d + tensor_shape.nbDims,
                                           1, std::multiplies<uint64_t>());
      check_cuda_err(
          cudaMallocAsync(&io_tensors_buf_[i], tensor_max_bytes, stream));
    } else {
      throw EngineException("Unsupported tensor type");
    }

    // log the basic information
    std::string tensor_type_str =
        tensor_type == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT";
    std::string dim_str;
    dim_str += '[';
    for (int n_i = 0; n_i < tensor_shape.nbDims; n_i++) {
      dim_str += std::to_string(tensor_shape.d[n_i]);
      if (n_i < tensor_shape.nbDims - 1) {
        dim_str += ", ";
      }
    }
    dim_str += ']';
    option_.logger.log(nvinfer1::ILogger::Severity::kINFO,
                       ("Tensor idx: " + std::to_string(i) +
                        ", name: " + tensor_name + ", type: " + tensor_type_str)
                           .c_str());
    option_.logger.log(nvinfer1::ILogger::Severity::kINFO,
                       ("    Dimensions: " + dim_str).c_str());
    option_.logger.log(
        nvinfer1::ILogger::Severity::kINFO,
        ("    Max bytes: " + std::to_string(tensor_max_bytes)).c_str());
  }

  // Synchronize and destroy the cuda stream
  check_cuda_err(cudaStreamSynchronize(stream));
  check_cuda_err(cudaStreamDestroy(stream));
};

trt::Engine::~Engine() {
  for (auto &buf : io_tensors_buf_) {
    check_cuda_err(cudaFree(buf));
  }

  io_tensors_buf_.clear();
}

void trt::Engine::run(cv::cuda::GpuMat &flatten_inputs,
                      const uint32_t &batch_size, std::vector<float> &outputs) {
  input_dims_.d[0] = batch_size; // Define the batch size
  context_->setInputShape(io_tensors_name_[0], input_dims_);

  cudaStream_t stream;
  check_cuda_err(cudaStreamCreate(&stream));

  auto *data_ptr = flatten_inputs.ptr<void>();
  check_cuda_err(cudaMemcpyAsync(io_tensors_buf_[0], data_ptr,
                                 flatten_inputs.rows * flatten_inputs.cols *
                                     flatten_inputs.channels() * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
  if (context_->inferShapes(1, io_tensors_name_.data()) != 0) {
    throw EngineException("Failed to infer shapes");
  }
  if (!context_->allInputDimensionsSpecified()) {
    throw EngineException("Not all input dimensions are specified");
  }
  for (uint32_t i = 0; i < (uint32_t)io_tensors_buf_.size(); i++) {
    if (!context_->setTensorAddress(io_tensors_name_[i], io_tensors_buf_[i])) {
      throw EngineException("Failed to set tensor address");
    }
  }
  if (!context_->enqueueV3(stream)) {
    throw EngineException("Failed to enqueue");
  }

  outputs.resize(batch_size * single_output_len_);
  check_cuda_err(
      cudaMemcpyAsync(outputs.data(), static_cast<char *>(io_tensors_buf_[1]),
                      batch_size * single_output_len_ * sizeof(float),
                      cudaMemcpyDeviceToHost, stream));

  check_cuda_err(cudaStreamSynchronize(stream));
  check_cuda_err(cudaStreamDestroy(stream));
}

std::vector<uint32_t> trt::Engine::get_input_dims() {
  auto input_dims = engine_->getTensorShape(io_tensors_name_[0]);
  return std::vector<uint32_t>(input_dims.d + 1,
                               input_dims.d + 1 + input_dims.nbDims - 1);
}

std::vector<uint32_t> trt::Engine::get_output_dims() {
  auto output_dims = engine_->getTensorShape(io_tensors_name_[1]);
  return std::vector<uint32_t>(output_dims.d + 1,
                               output_dims.d + 1 + output_dims.nbDims - 1);
}
