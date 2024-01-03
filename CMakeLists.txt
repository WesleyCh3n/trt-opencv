cmake_minimum_required(VERSION 3.18)

project(trt)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#==============================================================================
# Cmake Options
#==============================================================================
set(TensorRT_ROOT "/opt/TensorRT-8.6.1.6" CACHE STRING "TensorRT library directory")
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.3/" CACHE STRING "CUDA toolkit directory")
option(BUILD_FMT "Build fmt library" ON)
option(BUILD_SPDLOG "Build spdlog library" ON)
option(BUILD_EXAMPLES "Build examples" ON)

#==============================================================================
# Third Party Dependencies
#==============================================================================
set(CAFFE2_USE_CUDNN ON)
find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED PATHS /opt/libtorch)
find_package(TorchVision REQUIRED PATHS /opt/libtorchvision)
list(APPEND TORCH_LIBRARIES libcupti.so)

find_package(TensorRT REQUIRED)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

include(cmake/CPM.cmake)
if(BUILD_FMT)
  CPMAddPackage(
    NAME fmt
    GIT_TAG 10.1.1
    GITHUB_REPOSITORY fmtlib/fmt
    OPTIONS "FMT_INSTALL YES"
  )
else()
  find_package(fmt REQUIRED)
endif()
if(BUILD_SPDLOG)
  CPMAddPackage(
    NAME spdlog
    GIT_TAG v1.12.0
    GITHUB_REPOSITORY gabime/spdlog
    OPTIONS "SPDLOG_FMT_EXTERNAL YES"
    "SPDLOG_INSTALL YES"
  )
else()
  find_package(spdlog REQUIRED)
endif()
add_dependencies(spdlog fmt)

#==============================================================================
# Build Libraries
#==============================================================================
add_library(trt
  src/trt.cpp
  )
target_link_libraries(trt PUBLIC
  TensorRT::TensorRT
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  fmt::fmt
  spdlog::spdlog
)
target_include_directories(trt PUBLIC
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(trt PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(trt PROPERTIES PUBLIC_HEADER "src/trt.hpp")

add_library(dnn
  src/dnn.cpp
)
target_link_libraries(dnn PUBLIC
  TorchVision::TorchVision
  trt
)
target_include_directories(dnn PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(dnn PROPERTIES PUBLIC_HEADER "src/dnn.hpp")
list(APPEND INSTALL_TARGETS
  trt
  dnn
  )

set(CONFIG_EXPORT_NAME trt-config)
set(TRT_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/trt)
set(TRT_SHARE_DIR share/trt)
set(PROJECT_CONFIG ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_EXPORT_NAME}.cmake)

install(TARGETS ${INSTALL_TARGETS} EXPORT ${CONFIG_EXPORT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trt)
export(TARGETS ${INSTALL_TARGETS}
       NAMESPACE trt::
       FILE ${PROJECT_CONFIG})
install(EXPORT ${CONFIG_EXPORT_NAME}
        DESTINATION "${TRT_CMAKE_DIR}"
        NAMESPACE trt::
)
install(FILES ./cmake/FindTensorRT.cmake DESTINATION ${TRT_SHARE_DIR})


#==============================================================================
# Build Examples
#==============================================================================
if(BUILD_EXAMPLES)
  CPMAddPackage("gh:jarro2783/cxxopts@3.1.1")
  list(APPEND EXAMPLES
    # resnet_single
    # resnet_batch
    # resnet_export_feature
    yolov8_single
    # yolov8_batch
    yolov8_export_image
    # benchmark_yolo
  )
  foreach(example ${EXAMPLES})
    add_executable(${example}
      example/${example}.cc
      )
    target_link_libraries(${example} PRIVATE
      dnn
      OpenMP::OpenMP_CXX
      cxxopts)
    target_include_directories(${example} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      )
  endforeach()

endif()
