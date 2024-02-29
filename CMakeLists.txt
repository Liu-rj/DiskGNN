cmake_minimum_required(VERSION 3.5)
project(offgs CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")

# Pytorch 
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND python3 ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND OFFGS_LINKER_LIBS ${TORCH_LIBRARIES} uring)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
message(STATUS "Torch dir: ${Torch_DIR}")

# Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
list(APPEND OFFGS_INCLUDES ${Python_INCLUDE_DIRS})
list(APPEND OFFGS_LINKER_LIBS ${Python_LIBRARIES})

message(STATUS "OFFGS linker: ${OFFGS_LINKER_LIBS}")

file(GLOB offgs_src
          src/*.cc
          src/cuda/*.cu)

include_directories(src ${OFFGS_INCLUDES})

add_library(offgs SHARED ${offgs_src})
set_target_properties(offgs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(offgs PROPERTIES CUDA_STANDARD 14)
set_target_properties(offgs PROPERTIES CXX_STANDARD 14)
set_target_properties(offgs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(offgs ${OFFGS_LINKER_LIBS})
