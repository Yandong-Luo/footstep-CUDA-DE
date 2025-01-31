#ifndef CUDAPROCESS_FOOTSTEP_MODEL_H
#define CUDAPROCESS_FOOTSTEP_MODEL_H

#include "diff_evolution_solver/data_type.h"
#include "footstep/footstep_utils.cuh"
#include <cublasdx.hpp>

using namespace cublasdx;

namespace footstep{
    // For matrix E times x
    using Ex_GEMM = decltype(Size< row_bigE, col_init_state, col_bigE>()
                        + Precision<float>()
                        + Type<type::real>()
                        + Function<function::MM>()
                        + Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + Block());

    // For matrix F times u
    using Fu_GEMM = decltype(Size< row_bigF, 1, col_bigF>()
                        + Precision<float>()
                        + Type<type::real>()
                        + Function<function::MM>()
                        + Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + Block());


    // Generative N step state and evaluate
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateStateAndEvaluate(cudaprocess::CudaParamClusterData<T> *cluster_data){
        if(blockIdx.x > 0)  return;

        float all_state[N * state_dims] = {0.0f};
        // float Fu_result[N * state_dims] = {0.0f};

        auto gt_bigE = cublasdx::make_tensor(bigE, Ex_GEMM::get_layout_gmem_a());   // global tensor (gt)
        auto gt_init_state = cublasdx::make_tensor(init_state, Ex_GEMM::get_layout_gmem_b());
        auto gt_all_state = cublasdx::make_tensor(all_state, Ex_GEMM::get_layout_gmem_c());

        Ex_GEMM().execute(1.0f, gt_bigE, gt_init_state, 0.0f, gt_all_state);
        __syncthreads();

        auto gt_bigF = cublasdx::make_tensor(bigF, Fu_GEMM::get_layout_gmem_a());   // global tensor (gt)

        float *cur_individual_param = cluster_data->all_param + threadIdx.x * CUDA_PARAM_MAX_SIZE;
        auto gt_u = cublasdx::make_tensor(cur_individual_param, Fu_GEMM::get_layout_gmem_b());

        // auto gt_Fu_result = cublasdx::make_tensor(Fu_result, Fu_GEMM::get_layout_gmem_c());

        Fu_GEMM().execute(1.0f, gt_bigF, gt_u, 1.0f, gt_all_state);

        __syncthreads();
    }
}

#endif