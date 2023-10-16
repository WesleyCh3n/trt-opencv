# **T**ensor**RT** with OpenCV

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
    - `resnet_single` [OPTION...] <model path> <input img path>
- `resnet_batch`: reading a batch of images and inferencing to a batch of features
    - resnet_batch [OPTION...] <model path> <input path>
- `resnet_export_feature`: exporting a batch of features and classes to tsv files
    - resnet_export_feature [OPTION...] <model path> <input path> <feature path> <class path>
- `yolov8_single`: reading a single image and export bounding boxes result to output image
    - yolov8_single [OPTION...] <model path> <input img path> <output img path>
- `yolov8_batch`: reading a batch of images
    - yolov8_batch [OPTION...] <model path> <input dir path>
- `yolov8_export_image`: reading a batch of images and export bounding boxes result to output image
    - yolov8_export_image [OPTION...] <model path> <input dir path> <output dir path>

exery example with `-h` option, you can get the help message.

## References

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api)
