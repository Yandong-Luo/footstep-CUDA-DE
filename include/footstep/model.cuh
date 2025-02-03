#ifndef CUDAPROCESS_FOOTSTEP_MODEL_H
#define CUDAPROCESS_FOOTSTEP_MODEL_H

#include "diff_evolution_solver/data_type.h"
#include "footstep/footstep_utils.cuh"
#include <cublasdx.hpp>

// using namespace cublasdx;

namespace footstep{
    // For matrix E times x
    using Ex_GEMM = decltype(cublasdx::Size< row_bigE, 1, col_bigE>()
                        + cublasdx::Precision<float>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + cublasdx::SM<860>()
                        + cublasdx::Block());

    // For matrix F times u
    using Fu_GEMM = decltype(cublasdx::Size< row_bigF, 1, col_bigF>()
                        + cublasdx::Precision<float>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + cublasdx::SM<860>()
                        + cublasdx::Block());

    template<class GEMM>
    __device__ void gemm_kernel(const float  alpha,
                                const float* a,
                                const float* b,
                                const float  beta,
                                float* c,
                                char* smem) {
        // extern __shared__ __align__(16) char smem[];

        // Make global memory tensor
        auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
        auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
        auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

        // Make shared memory tensor
        auto [smem_a, smem_b, smem_c] = GEMM::slice_shared_memory(smem);
        auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
        auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
        auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

        // Load data from global memory tensor to shared memory tensor
        using alignment = cublasdx::alignment_of<GEMM>;
        cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
        cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
        cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
        cublasdx::copy_wait();

        // Execute GEMM
        GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
        __syncthreads();

        // Store data from shared memory tensor to global memory tensor
        cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
    }

    // Generative N step state and evaluate
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateStateAndEvaluate(cudaprocess::CudaParamClusterData<T> *cluster_data, const float *bigE, const float *bigF, float *cluster_state){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;

        // current individual control input (N step)
        float *cur_individual_param = cluster_data->all_param + blockIdx.x * CUDA_PARAM_MAX_SIZE;

        float *N_states = cluster_state + blockIdx.x * N * state_dims;
        
        extern __shared__ __align__(16) char smem[];

        // matrix bigE times init_states, and record the result at N_states
        gemm_kernel<Ex_GEMM>(1.0f, bigE, init_state, 0.0f, N_states, smem);

        __syncthreads();

        // matrix bigE times init_states, and plus the result at F
        gemm_kernel<Fu_GEMM>(1.0f, bigF, cur_individual_param, 1.0f, N_states, smem);

        // __syncthreads();
        
    }
}

#endif