#ifndef CUDAPROCESS_RANDOM_MANAGER_H
#define CUDAPROCESS_RANDOM_MANAGER_H


#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <ctime>
#include "utils/utils.cuh"

namespace cudaprocess{
    class CudaRandomManager {
        private:
            static unsigned long long GenerateRandomSeed();
        public:
            CudaRandomManager(int gpu_device);
            ~CudaRandomManager();
            void Generate();
            void Regenerate();
            void Regenerate(unsigned long long seed);
            float* GetUniformData() { return uniform_data_; }
            float* GetNormalData() { return normal_data_; }

            curandGenerator_t gen;
            float* uniform_data_;
            float* normal_data_;
            const unsigned long long uniform_size_ = REGENRATE_RANDOM_FREQUENCE * (CUDA_SOLVER_POP_SIZE + 1) * (2*CUDA_PARAM_MAX_SIZE + 5);     // or 512000
            const unsigned long long normal_size_ = (CUDA_SOLVER_POP_SIZE + 1) * 3 * REGENRATE_RANDOM_FREQUENCE;
            cudaStream_t stream;
    };
}

#endif