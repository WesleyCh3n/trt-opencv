#include <opencv2/opencv.hpp>

#include <cxxopts.hpp>

#include "trt.hpp"
#include "utils.hpp"

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
    exit(1);
  }
}

void process_single_img() {
  trt::EngineOption model_option{.max_batch_size = global::max_batch_size};
  trt::Engine model(global::model_path, model_option);
  auto in_size = model.get_input_dims();
  auto out_size = model.get_output_dims();
  // in my case input are images with 3 channels, output are 512 feature
  assert(in_size.size() == 3);                // CHW
  assert(in_size[0] == 3 || in_size[0] == 1); // channel first

  // read image into gpu
  std::vector<float> outputs;
  auto cpumat = cv::imread(global::image_path);
  auto gpumat = cv::cuda::GpuMat(cpumat);

  // init yolo utility
  auto yolo_util = trt::utils::YoloUtility::create()
                       .set_original_size(cpumat.size())
                       .set_input_size(cv::Size(in_size[2], in_size[1]))
                       .set_conf_threshold(0.5)
                       .set_nms_threshold(0.5)
                       .set_num_bbox(out_size[1]);
  if (out_size[0] == 5)
    yolo_util.is_xywhs();
  yolo_util.show();

  // transform image
  auto blob = trt::utils::blob_from_gpumat(
      gpumat,                                          // input gpumats
      std::array<uint32_t, 2>{in_size[1], in_size[2]}, // resize
      std::array<float, 3>{1, 1, 1},                   // std factor
      std::array<float, 3>{0, 0, 0}                    // mean
  );

  // run model inference
  model.run(blob, 1, outputs);

  // post process output
  std::vector<cv::Rect> rects;
  std::vector<float> confs;
  std::vector<int> idxes;
  yolo_util.post_process(outputs.data(), rects, confs, idxes);

  // draw result
  for (const auto &idx : idxes) {
    cv::rectangle(cpumat, rects[idx], cv::Scalar(0, 255, 0));
  }

  // save result image
  cv::imwrite(global::output_image_path, cpumat);
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
