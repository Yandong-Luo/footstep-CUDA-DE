cmake_minimum_required(VERSION 3.18)
project(Parallel_DiffEvolutionSolver LANGUAGES CXX CUDA)

# 删除这行，因为使用现代CMake不需要手动指定CUDA路径
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)

# 使用现代CMake方式查找CUDA
find_package(CUDAToolkit REQUIRED)

# mathdx 配置
set(CMAKE_FIND_PACKAGE_SCOPE global)
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS "/usr/local/mathdx/24.08/")

find_package(yaml-cpp REQUIRED)

find_package(cudss QUIET)

find_package(Eigen3 REQUIRED)

# NVTX配置保持不变
add_library(nvtx3-cpp INTERFACE)
target_include_directories(nvtx3-cpp INTERFACE 
    ${PROJECT_SOURCE_DIR}/thirdparty/nvtx/c/include
)
if(UNIX)
    target_link_libraries(nvtx3-cpp INTERFACE ${NVTOOLSEXT_LIBRARY})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11)

# 删除这两行，不再需要全局包含
# include_directories(${CUDA_INCLUDE_DIRS})
# link_directories(${CUDA_LIBRARIES})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -fPIC -lineinfo")

# 改用target_include_directories替代全局include_directories
file(GLOB_RECURSE ALL_CUDA "${PROJECT_SOURCE_DIR}/src/*.cu")

pybind11_add_module(cuda_solver 
    ${PROJECT_SOURCE_DIR}/src/solver_center/solver_center.cpp
)

add_library(cuda_DE STATIC ${ALL_CUDA})

# 专门禁用 20012-D 警告
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 20012")
endif()

target_include_directories(cuda_DE SYSTEM PUBLIC
    ${EIGEN3_INCLUDE_DIR}
)

# 为cuda_DE添加包含目录
target_include_directories(cuda_DE PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/nvtx/c/include
    # ${EIGEN3_INCLUDE_DIR}
    "/usr/local/mathdx/24.08/include"
    "/usr/local/mathdx/24.08/include/cublasdx/include"
    "/usr/local/mathdx/24.08/external/cutlass/include"
)

# CUDA架构设置保持不变
set_property(TARGET cuda_DE PROPERTY
    CUDA_ARCHITECTURES 86
)

set_target_properties(cuda_DE PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)

set_target_properties(cuda_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    OUTPUT_NAME "DE_cuda_solver"
    PREFIX ""
    SUFFIX ".so"
)

# 修改链接库，使用现代CMake的CUDA targets
target_link_libraries(cuda_solver PRIVATE
    cuda_DE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    nvtx3-cpp
    yaml-cpp::yaml-cpp
    mathdx::cublasdx
    cudss
    Eigen3::Eigen
)

# 保持这个属性不变
set_target_properties(cuda_solver PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)