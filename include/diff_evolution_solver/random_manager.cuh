#ifndef CUDAPROCESS_RANDOM_MANAGER_H
#define CUDAPROCESS_RANDOM_MANAGER_H


#include <curand.h>
#include <curand_kernel.h>
#include "utils/utils.cuh"

namespace cudaprocess{
    class CudaRandomManager {
        private:
            
        public:
            CudaRandomManager(int gpu_device);
            ~CudaRandomManager();
            void Generate();
            float* GetUniformData() { return uniform_data_; }
            float* GetNormalData() { return normal_data_; }


            curandGenerator_t gen;
            float* uniform_data_;
            float* normal_data_;
            const uint uniform_size_ = CUDA_MAX_ROUND_NUM * (CUDA_SOLVER_POP_SIZE + 1) * (2*CUDA_PARAM_MAX_SIZE + 5);     // or 512000
            const uint normal_size_ = (CUDA_SOLVER_POP_SIZE + 1) * 3 * CUDA_MAX_ROUND_NUM;
            cudaStream_t stream;
    };
}

#endif