#include <opencv2/opencv.hpp>

#include <cxxopts.hpp>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "dnn.hpp"

namespace global {
std::string model_path;
std::string image_path;
uint32_t batch_size;
uint32_t max_batch_size;
} // namespace global

void parse_args(int argc, char *argv[]) {
  cxxopts::Options options("resnet_single",
                           "TensorRT Resnet50 single image example");
  // clang-format off
  options.add_options()
    ("model", "model path", cxxopts::value<std::string>())
    ("input", "single image path", cxxopts::value<std::string>())
    ("b,batch", "batch size", cxxopts::value<int>()->default_value("256"))
    ("m,maxbatch", "max batch size of model", cxxopts::value<int>()->default_value("256"))
    ("h,help", "help");
  options.parse_positional({"model", "input"});
  options.positional_help("<model path> <input img path>");
  options.show_positional_help();
  auto result = options.parse(argc, argv);
  // clang-format on
  try {
    if (result.count("help")) {
      std::cout << options.help();
      exit(0);
    }
    global::model_path = result["model"].as<std::string>().c_str();
    global::image_path = result["input"].as<std::string>().c_str();
    global::batch_size = result["batch"].as<int>();
    global::max_batch_size = result["maxbatch"].as<int>();
    assert(global::batch_size <= global::max_batch_size);
  } catch (const cxxopts::exceptions::exception &e) {
    spdlog::error(e.what());
    spdlog::info(options.help());
    exit(1);
  }
}

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR) {
      spdlog::error("TRT INTERNAL ERROR: {}", msg);
    } else if (severity == Severity::kERROR) {
      spdlog::error("TRT ERROR: {}", msg);
    } else if (severity == Severity::kWARNING) {
      spdlog::warn("TRT WARNING: {}", msg);
    } else if (severity == Severity::kINFO) {
      spdlog::info("TRT INFO: {}", msg);
    }
  }
};

void process_single_img() {
  Logger logger;
  trt::EngineOption options{global::max_batch_size, logger};
  dnn::FeatureExtractor model(global::model_path, options);

  auto cpumat = cv::imread(global::image_path);
  auto gpumat = cv::cuda::GpuMat(cpumat);

  // set grayscale
  auto r = model.predict({gpumat, gpumat}, true, true, true);

  std::cout << r.size() << '\n';
  for (int b = 0; b < int(r.size() / 512); b++) {
    fmt::println("r[{}][:10]:\n{}", b,
                 std::vector(r.begin() + b * 512, r.begin() + b * 512 + 10));
  }
}

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::trace);
  try {
    parse_args(argc, argv);
    trt::check_cuda_err(cudaSetDevice(0));
    process_single_img();
  } catch (trt::EngineException &e) {
    spdlog::error("trt::EngineException: {}", e.what());
    return 1;
  } catch (trt::CudaException &e) {
    spdlog::error("trt::CudaException: {}", e.what());
    return 1;
  } catch (std::exception &e) {
    spdlog::error("std::exception: {}", e.what());
    return 1;
  }
  return 0;
}
