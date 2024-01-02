#include <filesystem>
#include <fstream>

#include <cxxopts.hpp>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <opencv2/opencv.hpp>

#include "trt.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;
namespace global {
std::string model_path;
std::string dir_path;
std::string output_path;
uint32_t batch_size;
uint32_t max_batch_size;
} // namespace global

void parse_args(int argc, char *argv[]) {
  cxxopts::Options options("yolov8_batch",
                           "TensorRT Resnet50 batch predict example");
  // clang-format off
  options.add_options()
    ("model", "model path", cxxopts::value<std::string>())
    ("input", "input image root dir w/ sub-folder classes", cxxopts::value<std::string>())
    ("o,output", "output txt dir", cxxopts::value<std::string>())
    ("b,batch", "batch size", cxxopts::value<int>()->default_value("64"))
    ("m,maxbatch", "max batch size of model", cxxopts::value<int>()->default_value("64"))
    ("h,help", "help");
  options.parse_positional({"model", "input"});
  options.positional_help("<model path> <input dir path>");
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
    global::batch_size = result["batch"].as<int>();
    global::max_batch_size = result["maxbatch"].as<int>();
    global::output_path = result["output"].as<std::string>().c_str();
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
  trt::EngineOption model_option{.max_batch_size = global::max_batch_size};
  trt::Engine model(global::model_path, model_option);
  auto in_size = model.get_input_dims();
  auto out_size = model.get_output_dims();
  // in my case input are images with 3 channels, output are 512 feature
  assert(in_size.size() == 3);                // CHW
  assert(in_size[0] == 3 || in_size[0] == 1); // channel first

  // get all image paths
  auto images_paths = get_images_path(global::dir_path);
  // get all image mats
  // auto cpu_images = get_images_mat(images_paths);

  // init yolo utility
  auto yolo_util = trt::utils::YoloUtility::create()
                       .set_original_size(cv::Size(1920, 1080))
                       .set_input_size(cv::Size(in_size[2], in_size[1]))
                       .set_conf_threshold(0.2)
                       .set_nms_threshold(0.5)
                       .set_num_bbox(out_size[1]);
  if (out_size[0] == 5)
    yolo_util.is_xywhs();
  yolo_util.show();

  // pre-allocate gpu mats
  const uint32_t batch_size =
      images_paths.size() < global::batch_size
          ? images_paths.size()
          : global::batch_size;                     // num of images per batch
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
    // load batch images into gpu mats
    for (int idx = iteration * batch_size; idx < batch_end_idx; idx++) {
      inputs[idx - iteration * batch_size] =
          cv::cuda::GpuMat(cv::imread(images_paths[idx]));
    }

    // if idx is last image(last batch), resize inputs to fit last batch
    if (batch_end_idx == images_paths.size())
      inputs.resize(images_paths.size() - iteration * batch_size);

    {
      Timer t;
      // pre-process batch images
      auto blob = trt::utils::blob_from_gpumats(
          inputs,                                          // input gpu mats
          std::array<uint32_t, 2>{in_size[1], in_size[2]}, // resize
          std::array<float, 3>{1, 1, 1},                   // std factor
          std::array<float, 3>{0, 0, 0});                  // mean factor
      // run model inference
      model.run(blob, inputs.size(), outputs);
    }

    std::vector<std::vector<cv::Rect>> rects(batch_size);
    std::vector<std::vector<float>> confs(batch_size);
    std::vector<std::vector<int>> idxes(batch_size);
    {
      Timer t; // 0.796ms
      // post process output
      yolo_util.batch_post_process(outputs, inputs.size(), rects, confs, idxes);
    }
    for (int i = 0; i < idxes.size(); i++) {
      if (idxes[i].size() == 0)
        continue;
      auto txt_path = fmt::format(
          "{}.txt", (global::output_path /
                     fs::path(images_paths[iteration * batch_size + i]).stem())
                        .string());
      std::ofstream ofs(txt_path);
      for (auto &idx : idxes[i]) {
        ofs << rects[i][idx].x << " " << rects[i][idx].y << " "
            << rects[i][idx].x + rects[i][idx].width << " "
            << rects[i][idx].y + rects[i][idx].height << " " << confs[i][idx]
            << '\n';
      }
    }

    // for (int i = 0; i < inputs.size(); i++) {
    //   fmt::println("{: >2}: rect size: {}", i, idxes[i].size());
    // }
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
