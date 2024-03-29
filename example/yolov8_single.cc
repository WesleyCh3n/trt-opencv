#include <opencv2/opencv.hpp>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <vector>

#include "dnn.hpp"
#include "trt.hpp"

namespace global {
std::string model_path;
std::string image_path;
std::string output_image_path;
uint32_t batch_size;
uint32_t max_batch_size;
} // namespace global

void parse_args(int argc, char *argv[]) {
  cxxopts::Options options("yolov8_single",
                           "TensorRT Yolov8 single image example");
  // clang-format off
  options.add_options()
    ("model", "model path", cxxopts::value<std::string>())
    ("input", "single image path", cxxopts::value<std::string>())
    ("output", "output image path", cxxopts::value<std::string>())
    ("b,batch", "batch size", cxxopts::value<int>()->default_value("64"))
    ("m,maxbatch", "max batch size of model", cxxopts::value<int>()->default_value("64"))
    ("h,help", "help");
  options.parse_positional({"model", "input","output"});
  options.positional_help("<model path> <input img path> <output img path>");
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
    global::output_image_path = result["output"].as<std::string>().c_str();
    global::batch_size = result["batch"].as<int>();
    global::max_batch_size = result["maxbatch"].as<int>();
    assert(global::batch_size <= global::max_batch_size);
  } catch (const cxxopts::exceptions::exception &e) {
    spdlog::error(e.what());
    spdlog::info(options.help());
    exit(2);
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

  auto model = dnn::Yolo(global::model_path, options);
  auto cpumat = cv::imread(global::image_path);
  auto gpumat = cv::cuda::GpuMat(cpumat);

  // test multiple inputs
  std::vector<cv::cuda::GpuMat> inputs;
  for (int i = 0; i < global::max_batch_size; i++) {
    inputs.emplace_back(gpumat.clone());
  }
  std::vector<std::vector<cv::Rect>> rects;
  std::vector<std::vector<float>> confs;
  model.predict(inputs, rects, confs);
  for (const auto &rect : rects) {
    fmt::println("=== {}", rect.size());
    for (const auto &r : rect) {
      fmt::println("{} {} {} {}", r.x, r.y, r.width, r.height);
    }
  }

  /* for (const auto &r : rs) {
    // fmt::println("=== {}", r.size());
    for (const auto &b : r) {
      // fmt::println("{} {} {} {}", b.rect.x, b.rect.y, b.rect.width,
      //              b.rect.height);
    }
  } */

  cv::Mat larger_mat;
  cv::resize(cpumat, larger_mat, cv::Size(2048, 1556));
  // auto results = model.predict({gpumat, cv::cuda::GpuMat(larger_mat)});
  model.predict({gpumat, cv::cuda::GpuMat(larger_mat)}, rects, confs);

  std::vector<cv::Mat> mats{cpumat, larger_mat};
  int i = 0;
  for (const auto &rect : rects) {
    for (const auto &r : rect) {
      cv::rectangle(mats[i], r, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(fmt::format("{}-size.jpg", i), mats[i]);
    i++;
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
