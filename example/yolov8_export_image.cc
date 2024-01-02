#include <filesystem>

#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "dnn.hpp"
#include "trt.hpp"
#include "utils.hpp"

namespace global {
std::string model_path;
std::string dir_path;
std::string output_dir;
uint32_t batch_size;
uint32_t max_batch_size;
} // namespace global

void parse_args(int argc, char *argv[]) {
  cxxopts::Options options(
      "yolov8_export_image",
      "TensorRT Resnet50 batch predict and save image example");
  // clang-format off
  options.add_options()
    ("model", "model path", cxxopts::value<std::string>())
    ("input", "input image root dir w/ sub-folder classes", cxxopts::value<std::string>())
    ("output", "output dir path", cxxopts::value<std::string>())
    ("b,batch", "batch size", cxxopts::value<int>()->default_value("64"))
    ("m,maxbatch", "max batch size of model", cxxopts::value<int>()->default_value("64"))
    ("h,help", "help");
  options.parse_positional({"model", "input","output"});
  options.positional_help("<model path> <input dir path> <output dir path>");
  options.show_positional_help();
  auto result = options.parse(argc, argv);
  // clang-format on
  try {
    if (result.count("help")) {
      std::cout << options.help();
      exit(0);
    }
    global::model_path = result["model"].as<std::string>().c_str();
    global::dir_path = result["input"].as<std::string>().c_str();
    global::output_dir = result["output"].as<std::string>().c_str();
    global::batch_size = result["batch"].as<int>();
    global::max_batch_size = result["maxbatch"].as<int>();
    assert(global::batch_size <= global::max_batch_size);
  } catch (const cxxopts::exceptions::exception &e) {
    spdlog::error(e.what());
    spdlog::info(options.help());
    exit(1);
  }
}

std::vector<std::string> get_images_path(const std::string &root_dir) {
  std::vector<std::string> image_paths;
  for (const auto &entry : std::filesystem::directory_iterator(root_dir)) {
    if (entry.path().extension() == ".jpg") {
      image_paths.emplace_back(entry.path());
    }
  }
  return image_paths;
}

std::vector<cv::Mat> get_images_mat(std::vector<std::string> &image_paths) {
  std::vector<cv::Mat> images(image_paths.size());
#pragma omp parallel for
  for (int i = 0; i < image_paths.size(); i++) {
    images[i] = cv::imread(image_paths[i]);
  }
  return images;
}

void process_batch_img() {
  auto model = dnn::Yolo(global::model_path, global::max_batch_size);
  auto images_paths = get_images_path(global::dir_path);
  auto cpu_images = get_images_mat(images_paths);

  // pre-allocate gpu mats
  const uint32_t batch_size = global::batch_size;   // num of images per batch
  std::vector<cv::cuda::GpuMat> inputs(batch_size); // pre-allocated gpu mats
  std::vector<float> outputs;

  uint32_t num_iteration =
      images_paths.size() % batch_size == 0
          ? (int)(images_paths.size() / batch_size)
          : (int)(std::floor(images_paths.size() / batch_size) + 1);

  spdlog::info("num_iteration: {}", num_iteration);
  for (int iteration = 0; iteration < num_iteration; iteration++) {
    // get this batch ending index of images
    int batch_end_idx = iteration == num_iteration - 1
                            ? images_paths.size()
                            : (iteration + 1) * batch_size;
    spdlog::info("processing batch {} ({}/{})", iteration, batch_end_idx,
                 images_paths.size());

    for (int idx = iteration * batch_size; idx < batch_end_idx; idx++) {
      inputs[idx - iteration * batch_size] = cv::cuda::GpuMat(cpu_images[idx]);
    }

    // if idx is last image(last batch), resize inputs to fit last batch
    if (batch_end_idx == images_paths.size())
      inputs.resize(images_paths.size() - iteration * batch_size);

    {
      Timer t;
      auto results = model.predict(inputs);
      for (int i = 0; i < results.size(); i++) {
        fmt::println("results size: {}", results[i].size());
        for (const auto &r : results[i]) {
          cv::rectangle(cpu_images[iteration * batch_size + i], r.rect,
                        cv::Scalar(0, 255, 0));
        }
        // save result image
        cv::imwrite(fmt::format("{}/img-{:02d}-{:02d}.jpg", global::output_dir,
                                iteration, i),
                    cpu_images[iteration * batch_size + i]);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::trace);
  try {
    parse_args(argc, argv);
    trt::check_cuda_err(cudaSetDevice(0));
    process_batch_img();
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
