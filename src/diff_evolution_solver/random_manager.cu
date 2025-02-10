#include "diff_evolution_solver/random_manager.cuh"

namespace cudaprocess{
    CudaRandomManager::CudaRandomManager(int gpu_device) {
        CHECK_CUDA(cudaSetDevice(gpu_device));
        CHECK_CUDA(cudaStreamCreate(&stream));
        
        CHECK_CUDA(cudaMalloc(&uniform_data_, sizeof(float) * uniform_size_));
        CHECK_CUDA(cudaMalloc(&normal_data_, sizeof(float) * normal_size_));
        
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
        CURAND_CHECK(curandSetStream(gen, stream));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 0));
        
        Generate();
    }

    CudaRandomManager::~CudaRandomManager() {
        cudaStreamSynchronize(stream);
        CURAND_CHECK(curandDestroyGenerator(gen));
        CHECK_CUDA(cudaFree(uniform_data_));
        CHECK_CUDA(cudaFree(normal_data_));
        CHECK_CUDA(cudaStreamDestroy(stream));
    }

    void CudaRandomManager::Generate() {
        CURAND_CHECK(curandGenerateUniform(gen, uniform_data_, uniform_size_));
        CURAND_CHECK(curandGenerateNormal(gen, normal_data_, normal_size_, 0.0f, 1.0f));
        cudaStreamSynchronize(stream);
    }

    void CudaRandomManager::Regenerate(unsigned long long seed) {
        // 重置随机数生成器的种子
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        
        // 同步确保之前的操作完成
        cudaStreamSynchronize(stream);
        
        // 重新生成随机数
        Generate();
    }

    void CudaRandomManager::Regenerate() {
        Regenerate(GenerateRandomSeed());
    }

    unsigned long long CudaRandomManager::GenerateRandomSeed() {
        // 方法1：使用时间戳和硬件随机数结合
        auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::random_device rd;  // 硬件随机数生成器
        
        // 将时间戳和随机数组合生成种子
        unsigned long long seed = static_cast<unsigned long long>(timestamp) ^ 
                                (static_cast<unsigned long long>(rd()) << 32);
        return seed;
    }
}