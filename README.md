# [**T**]ensor[**RT**]-cpp with OpenCV

A example wrapper for inferencing TensorRT with OpenCV as input. And some other
utilities for preprocessing and postprocessing images.

## Tested Environment

- Ubuntu 22.04
- [CUDA 12.2.1](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Cudnn 8.9.4.25](https://developer.nvidia.com/cudnn)
- [TensorRT 8.6.1.6](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
- OpenCV 4.8.0 with cuda enable

## Quick start

### Building Library

```sh
cmake -DTensorRT_DIR=/opt/TensorRT-8.6.1.6 \ # change this to your tensorrt root
    -B build .
cmake --build build --parallel
```

### Install Library

```sh
sudo cmake --install build --prefix /opt/trt # use prefix to install where you want
```

## Usage

All api are located in `src/trt.hpp`.

### TensorRT API: `trt::Engine`

```cpp
Engine(const std::string_view model_path,
       const EngineOption &option);
```
- @brief construct a trt engine from a model file.
- @param model_path model file path
- @param option engine option

```cpp
void run(cv::cuda::GpuMat &flatten_inputs,
         const uint32_t &batch_size,
         std::vector<float> &outputs)
```
- @brief inference a flatten batch of cv::cuda::GpuMat, return a vector of
float. User should reorder output yourself.
- @param flatten_inputs flatten batch of cv::cuda::GpuMat
- @param batch_size number of batch
- @param outputs output vector of float

```cpp
std::vector<uint32_t> get_input_dims()
```
- @brief get input dims. Omitting batch size, so if input size is {-1, 3, 224,
224}, the return vector will be {3, 224, 224}.
- @return vector of dimension

```cpp
std::vector<uint32_t> get_output_dims()
```
- @brief get output dims. Omitting batch size, so if output size is {-1,
512}, the return vector will be {512}.
- @return vector of dimension

### Utilities: `trt::utils`

#### Pre-process

```cpp
cv::cuda::GpuMat blob_from_gpumat(cv::cuda::GpuMat &inputs,
                                  const std::array<uint32_t, 2> &size,
                                  const std::array<float, 3> &std,
                                  const std::array<float, 3> &mean,
                                  bool swapBR = true, bool normalize = true);
cv::cuda::GpuMat blob_from_gpumats(std::vector<cv::cuda::GpuMat> &inputs,
                                   const std::array<uint32_t, 2> &size,
                                   const std::array<float, 3> &std,
                                   const std::array<float, 3> &mean,
                                   bool swapBR = true, bool normalize = true);
```
- @brief preprocess 1/batch of cv::cuda::GpuMat and change the order from
[B,H,W,C] to [B,C,H,W] then export to 1D continuous GpuMat.
- @param inputs batch of cv::cuda::GpuMat
- @param size resized image
- @param std image std
- @param mean image mean
- @param swapBR whether swap BGR to RGB
- @param normalize whether normalize to [0, 1]
- @return cv::cuda::GpuMat 1D GpuMat
- @note if inputs dimension is `[2, 1280, 720, 3]: CV_8UC3`, resized to
`[640, 640]`, first function will resize images to `[2, 640, 640, 3]`, then
reorder channel to `[2, 3, 640, 640]`. finally, output should be 1D GpuMat
`[1, 2 * 640 * 640 ]: CV_32FC3`

#### Post-process

- Yolov8 post-processing

Create utility instance with builder:
```cpp
auto yolo_util = trt::utils::YoloUtility::create()
                   .set_original_size(cv::Size(1280, 720)) // original image size
                   .set_input_size(cv::Size(640, 640)) // model input size
                   .set_conf_threshold(0.5) // confidence threshold
                   .set_nms_threshold(0.5) // nms threshold
                   .set_num_bbox(8400) // number of bbox
                   .is_xywhs(); // output only one class
```

Post-process:
```cpp
// single image
std::vector<cv::Rect> rects;
std::vector<float> confs;
std::vector<int> idxes;
yolo_util.post_process(outputs.data(), rects, confs, idxes);

// batch images
std::vector<std::vector<cv::Rect>> rects(batch_size);
std::vector<std::vector<float>> confs(batch_size);
std::vector<std::vector<int>> idxes(batch_size);
yolo_util.batch_post_process(outputs, inputs.size(), rects, confs, idxes);
```

## Example

Read a single image and inferencing to yolov8 model
```cpp
trt::EngineOption model_option{.max_batch_size = 64};
trt::Engine model("yolov8n.trt", model_option);

// get input image
auto cpumat = cv::imread("your/image.jpg");
auto gpumat = cv::cuda::GpuMat(cpumat);

auto yolo_util = trt::utils::YoloUtility::create()
                   .set_original_size(cpu_images.size()) // original image size
                   .set_input_size(cv::Size(640, 640)) // model input size
                   .set_conf_threshold(0.5) // confidence threshold
                   .set_nms_threshold(0.5) // nms threshold
                   .set_num_bbox(8400) // number of bbox
                   .is_xywhs(); // output only one class

// transform image
auto blob = trt::utils::blob_from_gpumat(
  gpumat,                                          // input gpumats
  std::array<uint32_t, 2>{640, 640},               // resize
  std::array<float, 3>{1, 1, 1},                   // std factor
  std::array<float, 3>{0, 0, 0}                    // mean
);

// run model inference
std::vector<float> outputs;
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
```


## References

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api)
