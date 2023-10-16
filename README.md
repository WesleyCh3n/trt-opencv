# [**T**]ensor[**RT**] with OpenCV

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
cmake
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \ # change this to your cuda toolkit root
    -DTensorRT_DIR=/opt/TensorRT-8.6.1.6 \ # change this to your tensorrt root
    -B build .
cmake --build build --parallel
```

### Running Examples

Navigate to `build/` directory. There are 6 examples:
- `resnet_single`: reading a single image and inferencing to a single feature
    - `resnet_single` [model path] [input img path]
- `resnet_batch`: reading a batch of images and inferencing to a batch of features
    - resnet_batch [model path] [input path]
- `resnet_export_feature`: exporting a batch of features and classes to tsv files
    - resnet_export_feature [model path] [input path] [feature path] [class path]
- `yolov8_single`: reading a single image and export bounding boxes result to output image
    - yolov8_single [model path] [input img path] [output img path]
- `yolov8_batch`: reading a batch of images
    - yolov8_batch [model path] [input dir path]
- `yolov8_export_image`: reading a batch of images and export bounding boxes result to output image
    - yolov8_export_image [model path] [input dir path] [output dir path]

Every example with `-h` option, you can get the help message. For example:

```sh
$ yolov8_export_image -h
TensorRT Resnet50 batch predict and save image example
Usage:
  yolov8_export_image [OPTION...] <model path> <input dir path> <output dir path>

      --model arg     model path
      --input arg     input image root dir w/ sub-folder classes
      --output arg    output dir path
  -b, --batch arg     batch size (default: 64)
  -m, --maxbatch arg  max batch size of model (default: 64)
  -h, --help          help
```

Walk through the examples will help you have a better understanding of how to use this library.

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

## References

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api)
