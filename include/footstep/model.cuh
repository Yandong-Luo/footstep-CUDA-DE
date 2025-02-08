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
    __device__ __forceinline__ void gemm_kernel(const float  alpha,
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

    // Generative N step state
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateState(cudaprocess::CudaParamClusterData<T> *cluster_data, const float *bigE, const float *bigF, float *cluster_state){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
        // ########### 
        // Update State
        // ###########
        // current individual control input (N step)
        float *cur_individual_param = cluster_data->all_param + blockIdx.x * CUDA_PARAM_MAX_SIZE;

        // for(int i = 0; i < CUDA_PARAM_MAX_SIZE && blockIdx.x >= CUDA_SOLVER_POP_SIZE-10; i++) {
        //     printf("blockIdx.x: %d, param[%d]: %f\n", blockIdx.x, i, cur_individual_param[i]);
        // }

        float *N_states = cluster_state + blockIdx.x * N * state_dims;
        
        extern __shared__ __align__(16) char smem[];

        // matrix bigE times init_states, and record the result at N_states
        gemm_kernel<Ex_GEMM>(1.0f, bigE, init_state, 0.0f, N_states, smem);

        __syncthreads();

        // matrix bigE times init_states, and plus the result at F
        gemm_kernel<Fu_GEMM>(1.0f, bigF, cur_individual_param, 1.0f, N_states, smem);

        __syncthreads();
    }

    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void EvaluateModel(cudaprocess::CudaParamClusterData<T> *cluster_data, float *cluster_state, float *score, float *sol_score = nullptr){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
        if(threadIdx.x >= 32)    return;

        // 32 threads don't need share memory.
        // __shared__ ALIGN(64) float N_score_sum[32];
        // __shared__ ALIGN(64) float N_obj_score_sum[32];
        // N_score_sum[threadIdx.x] = 0.0f;
        // N_obj_score_sum[threadIdx.x] = 0.0f;

        // current_step_score
        float cs_score = 0.0f;

        // current_step_constraint_score
        float cs_constraint_score = 0.0f;

        // current_step_objective_score
        float cs_obj_score = 0.0f;

        float cs_u_theta_cost = 0.0f;
        
        float cs_dist_to_target = 0.0f;
        float last_dist_to_target = 0.0f;

        if(threadIdx.x < N){
            float *current_state = cluster_state + blockIdx.x * N * state_dims + threadIdx.x * state_dims;

            float *cur_individual_param = cluster_data->all_param + blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x * control_dims;

            // #################
            // region constraint
            // #################
            // if robot doesn't stay at any region will get a huge penalty
            // calculate the region
            int current_region = -1;
            for(int i = 0; i < num_regions; ++i){
                if(current_state[0] >= all_region[i].x && current_state[0] <= all_region[i].y && current_state[1] >= all_region[i].z && current_state[1] <= all_region[i].w){
                    current_region = i;
                    break;
                }
            }

            if(current_region == -1)    cs_constraint_score += pos_penalty;

            // ##########################
            // speed and theta constraint
            // ##########################
            // if robot doesn't stay at any region will get a huge penalty

            // state[0]: position x
            // state[1]: position y
            // state[2]: dot x
            // state[3]: dot y
            // state[4]: theta
            cs_constraint_score += fabsf(current_state[2]) > speed_x_ub ? state_penalty : 0.0f;
            cs_constraint_score += fabsf(current_state[3]) > speed_y_ub ? state_penalty : 0.0f;
            cs_constraint_score += fabsf(current_state[4]) > theta_ub ? state_penalty : 0.0f;

            // ##########################
            // control constraint
            // ##########################
            // cur_individual_param[0]: ux
            // cur_individual_param[1]: uy
            // cur_individual_param[2]: u_theta
            cs_constraint_score += fabsf(cur_individual_param[0]) > ux_ub ? control_penalty : 0.0f;
            cs_constraint_score += fabsf(cur_individual_param[1]) > uy_ub ? control_penalty : 0.0f;
            cs_constraint_score += fabsf(cur_individual_param[2]) > utheta_ub ? control_penalty : 0.0f;

            if(fabsf(cur_individual_param[0]) > ux_ub || fabsf(cur_individual_param[1]) > uy_ub || fabsf(cur_individual_param[2]) > utheta_ub){
                printf("current param exceed the range:%f %f %f\n", cur_individual_param[0], cur_individual_param[1], cur_individual_param[2]);
            }

            // ##########################
            // foothold constraint
            // ##########################
            // foothold bound

            const float2 *cur_circle = (threadIdx.x & 1 != first_step_num) ? circles2 : circles;

            // I think should not use last state. The correct one is next state I think
            float *last_state = nullptr;
            if(threadIdx.x == 0){
                last_state = init_state;
            }
            else{
                last_state = cluster_state + blockIdx.x * N * state_dims + (threadIdx.x - 1) * state_dims;
            }

            // equation 3 and 4 in https://ieeexplore.ieee.org/document/7041373
            for(int i = 0; i < circle_num; ++i){
                if (fabsf(current_state[0] - (last_state[0] + __cosf(last_state[4]) * cur_circle[i].x - __sinf(last_state[4]) * cur_circle[i].y)) > radii[i]){
                    cs_constraint_score += state_penalty;
                    break;
                }
                else if (fabsf(current_state[1] - (last_state[1] + __sinf(last_state[4]) * cur_circle[i].x - __cosf(last_state[4]) * cur_circle[i].y)) > radii[i]){
                    cs_constraint_score += state_penalty;
                    break;
                }
            }

            // ##########################
            // objective function
            // ##########################
            float2 cur_fk = fk;
            if (threadIdx.x & 1 != first_step_num){
                cur_fk = fk2;
            }

            // float dist_to_target_x = fabsf(current_state[0] - (__cosf(current_state[4]) * cur_target_circle.x - __sinf(current_state[4]) * cur_target_circle.y));
            // float dist_to_target_y = fabsf(current_state[1] - (__sinf(current_state[4]) * cur_target_circle.x - __cosf(current_state[4]) * cur_target_circle.y));

            // equal to uRu^T
            cs_u_theta_cost = 5.0f * cur_individual_param[2] * cur_individual_param[2];

            float forceTrackingPenalty_x = fabsf(cur_individual_param[0] - (__cosf(current_state[4]) * cur_fk.x - __sinf(current_state[4]) * cur_fk.y));

            float forceTrackingPenalty_y = fabsf(cur_individual_param[1] - (__sinf(current_state[4]) * cur_fk.x - __cosf(current_state[4]) * cur_fk.y));

            cs_obj_score = cs_u_theta_cost + 5.0f * (forceTrackingPenalty_x * forceTrackingPenalty_x + forceTrackingPenalty_y * forceTrackingPenalty_y);

            cs_dist_to_target = target_weight * sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));

            if(threadIdx.x == N-1){
                last_dist_to_target = sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));    
                // printf("blockIdx.x:%d dist_to_target:%f\n", blockIdx.x, dist_to_target);
            }
            cs_score = cs_obj_score + cs_constraint_score + cs_dist_to_target;
            // cs_score = cs_obj_score + cs_constraint_score;
        }
        last_dist_to_target = __shfl_sync(0xffffffff, last_dist_to_target, N-1);
        
        // N_obj_score_sum[threadIdx.x] = cs_obj_score;

        __syncthreads();

        // sum up N step
        if (threadIdx.x < 32) {
            // float N_step_score = N_score_sum[threadIdx.x];
            cs_score += __shfl_down_sync(0xffffffff, cs_score, 16);
            cs_score += __shfl_down_sync(0xffffffff, cs_score, 8);
            cs_score += __shfl_down_sync(0xffffffff, cs_score, 4);
            cs_score += __shfl_down_sync(0xffffffff, cs_score, 2);
            cs_score += __shfl_down_sync(0xffffffff, cs_score, 1);

            // float obj_score = N_obj_score_sum[threadIdx.x];
            cs_obj_score += __shfl_down_sync(0xffffffff, cs_obj_score, 16);
            cs_obj_score += __shfl_down_sync(0xffffffff, cs_obj_score, 8);
            cs_obj_score += __shfl_down_sync(0xffffffff, cs_obj_score, 4);
            cs_obj_score += __shfl_down_sync(0xffffffff, cs_obj_score, 2);
            cs_obj_score += __shfl_down_sync(0xffffffff, cs_obj_score, 1);

            cs_constraint_score += __shfl_down_sync(0xffffffff, cs_constraint_score, 16);
            cs_constraint_score += __shfl_down_sync(0xffffffff, cs_constraint_score, 8);
            cs_constraint_score += __shfl_down_sync(0xffffffff, cs_constraint_score, 4);
            cs_constraint_score += __shfl_down_sync(0xffffffff, cs_constraint_score, 2);
            cs_constraint_score += __shfl_down_sync(0xffffffff, cs_constraint_score, 1);
            
            if (threadIdx.x == 0) {
                score[blockIdx.x] = cs_score;
                
                if(sol_score != nullptr){
                    sol_score[0] = cs_score;
                    sol_score[1] = cs_obj_score;
                    sol_score[2] = cs_constraint_score;
                    sol_score[3] = last_dist_to_target;
                }
                // printf("")
                // printf("block:%d, thread:%d, score:%f, obj_score:%f, constraint:%f\n",
                //     blockIdx.x, threadIdx.x, cs_score, cs_obj_score, cs_constraint_score);
            }
        }
    }
}

#endif