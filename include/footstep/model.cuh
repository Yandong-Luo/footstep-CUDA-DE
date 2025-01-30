#ifndef CUDAPROCESS_FOOTSTEP_MODEL_H
#define CUDAPROCESS_FOOTSTEP_MODEL_H

#include "diff_evolution_solver/data_type.h"
#include "footstep/footstep_utils.cuh"
#include <cublasdx.hpp>
// #include <cublasdx/cublasdx.hpp>
// #include <cublasdx/include/cublasdx.hpp>



using namespace cublasdx;

namespace footstep{
    // For matrix E times x
    using Ex_GEMM = decltype(Size< row_E, col_init_state, col_E>()
                        + Precision<float>()
                        + Type<type::real>()
                        + Function<function::MM>()
                        + Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + Block());

    // For matrix F times u
    using Fu_GEMM = decltype(Size< row_F, 1, col_F>()
                        + Precision<float>()
                        + Type<type::real>()
                        + Function<function::MM>()
                        + Arrangement<cublasdx::row_major, cublasdx::row_major>()
                        + Block());


    // Generative N step state and evaluate
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateStateAndEvaluate(cudaprocess::CudaParamClusterData<T> *cluster_data){
        if(blockIdx.x > 0)  return;

        // float last_state[5] = init_state;
        float last_state[5] = {0.0f};        
        for(int j = 0; j < state_dims; ++j) {
            last_state[j] = init_state[j];
        }
        for(int i = 0; i < N; ++i){
            float state[5] = {0.0f};

            // Ex_GEMM().execute(1.0f, E, last_state, 1.0f, state);

            // float current_u[3] = {
            //     cluster_data->all_param[threadIdx.x * CUDA_PARAM_MAX_SIZE + i*control_dims + 0],
            //     cluster_data->all_param[threadIdx.x * CUDA_PARAM_MAX_SIZE + i*control_dims + 1],
            //     cluster_data->all_param[threadIdx.x * CUDA_PARAM_MAX_SIZE + i*control_dims + 2]
            // };
            // __syncthreads();
            // Fu_GEMM().execute(1.0f, F, current_u, 1.0f, state);
        }
    }
}

#endif