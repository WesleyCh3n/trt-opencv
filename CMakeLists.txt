cmake_minimum_required(VERSION 3.22)

project(trt)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#==============================================================================
# Cmake Options
#==============================================================================
set(TensorRT_DIR "/opt/TensorRT-8.6.1.6" CACHE STRING "TensorRT library directory")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE STRING "CUDA toolkit directory")
option(BUILD_FMT "Build fmt library" ON)
option(BUILD_SPDLOG "Build spdlog library" ON)
option(BUILD_EXAMPLES "Build examples" ON)

#==============================================================================
# Third Party Dependencies
#==============================================================================
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

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
    OPTIONS "SPDLOG_INSTALL YES"
    "SPDLOG_FMT_EXTERNAL YES"
  )
else()
  find_package(spdlog REQUIRED)
endif()
add_dependencies(spdlog fmt)

#==============================================================================
# Build Libraries
#==============================================================================
add_library(trt
  src/trt_engine.cc
  src/trt_utils.cc
  )
target_link_libraries(trt PUBLIC TensorRT::TensorRT ${CUDA_LIBRARIES} ${OpenCV_LIBS} fmt::fmt spdlog::spdlog)
target_include_directories(trt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set_target_properties(trt PROPERTIES PUBLIC_HEADER "src/trt.hpp")
INSTALL(TARGETS trt
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

#==============================================================================
# Build Examples
#==============================================================================
if(BUILD_EXAMPLES)
  CPMAddPackage("gh:jarro2783/cxxopts@3.1.1")
  list(APPEND EXAMPLES
    resnet_single
    resnet_batch
    resnet_export_feature
    yolov8_single
    yolov8_batch
    yolov8_export_image
  )
  foreach(example ${EXAMPLES})
    add_executable(${example} example/${example}.cc)
    target_link_libraries(${example} PRIVATE trt OpenMP::OpenMP_CXX cxxopts)
    target_include_directories(${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endforeach()

endif()
