# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()

project(transformer_engine_tensorflow LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CMAKE_CUDA_FLAGS "-G")
endif()

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)

add_library(_pywrap_transformer_engine SHARED
  src/pywrap_transformer_engine.cu)

# Includes
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())" 
                OUTPUT_VARIABLE Pybind_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())" 
                OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy as np; print(np.get_include())" 
                OUTPUT_VARIABLE Numpy_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
target_include_directories(_pywrap_transformer_engine PRIVATE "${PROJECT_SOURCE_DIR}/../common/include")
target_include_directories(_pywrap_transformer_engine PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(_pywrap_transformer_engine PRIVATE ${Pybind_INCLUDE_DIRS})
target_include_directories(_pywrap_transformer_engine PRIVATE ${Tensorflow_INCLUDE_DIRS})
target_include_directories(_pywrap_transformer_engine PRIVATE ${Numpy_INCLUDE_DIRS})

# Libraries
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.__file__)" 
                OUTPUT_VARIABLE Tensorflow_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(Tensorflow_LIB_PATH ${Tensorflow_LIB_PATH} DIRECTORY)
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/libtensorflow_framework.so.2")
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/python/_pywrap_tensorflow_internal.so")

target_link_libraries(_pywrap_transformer_engine
  ${PYTHON_LIBRARIES}
  ${TF_LINKER_LIBS}
  transformer_engine
  cudart)

set_target_properties(_pywrap_transformer_engine PROPERTIES PREFIX "")
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 
                OUTPUT_VARIABLE PYTHON_EXT_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
set_target_properties(_pywrap_transformer_engine PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")