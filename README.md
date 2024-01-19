# [**T**]ensor[**RT**] with OpenCV

`libtrt`: A example wrapper for inferencing using TensorRT with OpenCV GpuMat as input.
`libdnn`: Class Yolov8 and Resnet with dynamic batch size model based on `libtrt`
to predict  OpenCV GpuMat(s).

## Tested Environment

- Ubuntu 22.04
- [CUDA 12.3](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Cudnn 8.9.4.25](https://developer.nvidia.com/cudnn)
- [TensorRT 8.6.1.6](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
- OpenCV 4.8.0 with CUDA enable

## Quick start

### Building Library

```sh
cmake -B build .
cmake --build build --parallel
```

### Install Library

```sh
sudo cmake --install build --prefix /opt/trt # use prefix to install where you want
```

### Example

```sh
build/yolov8_single model/yolov8.trt 20231212-161000.740_00_126.jpg output.jpg
build/resnet_single model/512-model-fp16.trt 20240114-041024.091_00_126_6_0.jpg output.jpg
```
