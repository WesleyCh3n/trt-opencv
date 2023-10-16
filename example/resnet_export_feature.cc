#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "trt.hpp"
#include "utils.hpp"

namespace global {
std::string model_path;
std::string dir_path;
uint32_t batch_size;
uint32_t max_batch_size;
std::string feature_path;
std::string class_path;
std::vector<std::string> classes;
} // namespace global

void parse_args(int argc, char *argv[]) {
  cxxopts::Options options("resnet_export_feature",
                           "TensorRT Resnet50 batch predict example");
  // clang-format off
  options.add_options()
    ("model", "model path", cxxopts::value<std::string>())
    ("input", "input root dir contain sub-folder classes images", cxxopts::value<std::string>())
    ("feature", "output feature path", cxxopts::value<std::string>())
    ("class", "output class path", cxxopts::value<std::string>())
    ("b,batch", "batch size", cxxopts::value<int>()->default_value("512"))
    ("m,maxbatch", "max batch size of model", cxxopts::value<int>()->default_value("512"))
    ("h,help", "help");
  options.parse_positional({"model", "input", "feature", "class"});
  options.positional_help("<model path> <input path> <feature path> <class path>");
  auto result = options.parse(argc, argv);
  // clang-format on

  try {
    if (result.count("help")) {
      std::cout << options.help();
      exit(0);
    }
    global::model_path = result["model"].as<std::string>().c_str();
    global::dir_path = result["input"].as<std::string>().c_str();
    global::feature_path = result["feature"].as<std::string>();
    global::class_path = result["class"].as<std::string>();
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
    if (entry.is_directory()) {
      for (const auto &img_entry :
           std::filesystem::directory_iterator(entry.path())) {
        if (img_entry.path().extension() == ".jpg") {
          image_paths.emplace_back(img_entry.path());
          global::classes.emplace_back(entry.path().stem());
        }
      }
    }
  }
  return image_paths;
}

void export_batch_feature() {
  std::ofstream output_file(global::feature_path);
  std::ofstream classes_file(global::class_path);

  trt::EngineOption model_option{.max_batch_size = global::max_batch_size};
  trt::Engine model(global::model_path, model_option);
  auto in_size = model.get_input_dims();
  auto out_size = model.get_output_dims();
  // in my case input are images with 3 channels, output are 512 feature
  assert(in_size.size() == 3);                // CHW
  assert(in_size[0] == 3 || in_size[0] == 1); // channel first

  // read image into gpu
  const uint32_t batch_size = global::batch_size;   // num of images per batch
  std::vector<cv::cuda::GpuMat> inputs(batch_size); // pre-allocated gpu mats
  std::vector<float> outputs;

  // get all image paths
  auto images_paths = get_images_path(global::dir_path);
  // write classes to file
  for (int i = 0; i < global::classes.size(); i++) {
    classes_file << global::classes[i] << "\n";
  }
  classes_file.close();

  uint32_t num_iteration =
      images_paths.size() % batch_size == 0
          ? (int)(images_paths.size() / batch_size)
          : (int)(std::floor(images_paths.size() / batch_size) + 1);
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

    // pre-process batch images
    auto blob = trt::utils::blob_from_gpumats(
        inputs,                                          // input gpu mats
        std::array<uint32_t, 2>{in_size[1], in_size[2]}, // resize
        std::array<float, 3>{0.5, 0.5, 0.5},             // scalefactor
        std::array<float, 3>{0.5, 0.5, 0.5});            // mean factor
    {
      Timer t;
      // run model inference
      model.run(blob, inputs.size(), outputs);
    }

    // get output information
    assert(outputs.size() == out_size[0] * inputs.size());

    // write batch features to tsv file
    for (int i = 0; i < inputs.size(); i++) {
      // write feature[:-2] as tab separated
      for (int n = 0; n < out_size[0] - 1; n++) {
        output_file << fmt::format("{:.5f}\t", outputs[out_size[0] * i + n]);
      }
      // write last feature[:-1] and add a newline
      output_file << fmt::format("{:.5f}\n",
                                 outputs[out_size[0] * (i + 1) - 1]);
    }
  }
}

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::trace);
  try {
    parse_args(argc, argv);
    trt::check_cuda_err(cudaSetDevice(0));
    export_batch_feature();
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
