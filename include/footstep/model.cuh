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

    using Ex_GEMM_col = decltype(cublasdx::Size< row_bigE, 1, col_bigE>()
                        + cublasdx::Precision<float>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>()
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

    // the invert F * D
    using F_invD_col = decltype(cublasdx::Size< row_DiagF_inv, 1, col_DiagF_inv>()
                        + cublasdx::Precision<float>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>()
                        + cublasdx::SM<860>()
                        + cublasdx::Block());
    
    using D_GEMM = decltype(cublasdx::Size< row_DiagE, 1, col_DiagE>()
                        + cublasdx::Precision<float>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>()
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

        // if(blockIdx.x == 0 && threadIdx.x == 0){
        //     printf("update state param:");
        //     for(int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
        //         printf("%f ", cur_individual_param[i]);
        //     }
        //     printf("\n");
        // }

        float *N_states = cluster_state + blockIdx.x * N * state_dims;
        
        extern __shared__ __align__(16) char smem[];

        // matrix bigE times init_states, and record the result at N_states
        gemm_kernel<Ex_GEMM>(1.0f, bigE, init_state, 0.0f, N_states, smem);

        __syncthreads();

        // matrix bigE times init_states, and plus the result at F
        gemm_kernel<Fu_GEMM>(1.0f, bigF, cur_individual_param, 1.0f, N_states, smem);

        // __syncthreads();
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

        // if(blockIdx.x == 0 && threadIdx.x == 0 && sol_score!=nullptr){
        //     printf("evaluate model param:");
        //     for(int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
        //         printf("%f ", cluster_data->all_param[i]);
        //     }
        //     printf("\n");
        // }

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
            float *current_state = nullptr;

            float *cur_individual_param = cluster_data->all_param + blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x * control_dims;;

            if(threadIdx.x != 0){
                current_state = cluster_state + blockIdx.x * N * state_dims + (threadIdx.x - 1) * state_dims;
            }
            else{
                current_state = init_state;
            }

            // #################
            // region constraint
            // #################
            // if robot doesn't stay at any region will get a huge penalty
            // calculate the region
            int current_region = -1;
            for(int i = 0; i < num_regions; ++i){
                // if(current_state[0] + cur_individual_param[0] >= all_region2[i].x && current_state[0] + cur_individual_param[0] <= all_region2[i].y && current_state[1] + cur_individual_param[1] >= all_region2[i].z && current_state[1] + cur_individual_param[1] <= all_region2[i].w){
                //     current_region = i;
                //     break;
                // }
                if(current_state[0] + cur_individual_param[0] >= all_region[i].x && current_state[0] + cur_individual_param[0] <= all_region[i].y && current_state[1] + cur_individual_param[1] >= all_region[i].z && current_state[1] + cur_individual_param[1] <= all_region[i].w){
                    current_region = i;
                    break;
                }
            }

            if(current_region == -1)    cs_constraint_score += pos_penalty;

            if(sol_score != nullptr && cs_constraint_score != 0){
                printf("current step: %d constraint from region:%f\n", threadIdx.x, cs_constraint_score);
            }

            // ##########################
            // speed and theta constraint
            // ##########################
            // if robot doesn't stay at any region will get a huge penalty

            // state[0]: position x
            // state[1]: position y
            // state[2]: dot x
            // state[3]: dot y
            // state[4]: theta
            cs_constraint_score += fabsf(current_state[2]) > speed_x_ub ? state_penalty * fabsf(current_state[2]) : 0.0f;
            cs_constraint_score += fabsf(current_state[3]) > speed_y_ub ? state_penalty * fabsf(current_state[3]): 0.0f;
            cs_constraint_score += fabsf(current_state[4]) > theta_ub ? state_penalty * fabsf(current_state[4]): 0.0f;

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
            // velocity constraint
            // ##########################
            // float before_vel_score = cs_constraint_score;
            // float backward_vel_constraint = (current_state[2] - (__cosf(current_state[4]) * vel_circle[0].x - __sinf(current_state[4]) * vel_circle[0].y)) * (current_state[2] - (__cosf(current_state[4]) * vel_circle[0].x - __sinf(current_state[4]) * vel_circle[0].y))
            //                             + (current_state[3] - (__sinf(current_state[4]) * vel_circle[0].x + __cosf(current_state[4]) * vel_circle[0].y)) * (current_state[3] - (__sinf(current_state[4]) * vel_circle[0].x + __cosf(current_state[4]) * vel_circle[0].y))
            //                             - (vel_circle_radii[0] * vel_circle_radii[0]);

            // float forward_vel_constraint = (current_state[2] - (__cosf(current_state[4]) * vel_circle[1].x - __sinf(current_state[4]) * vel_circle[1].y)) * (current_state[2] - (__cosf(current_state[4]) * vel_circle[1].x - __sinf(current_state[4]) * vel_circle[1].y))
            //                      + (current_state[3] - (__sinf(current_state[4]) * vel_circle[1].x + __cosf(current_state[4]) * vel_circle[1].y)) * (current_state[3] - (__sinf(current_state[4]) * vel_circle[1].x + __cosf(current_state[4]) * vel_circle[1].y))
            //                      - (vel_circle_radii[1] * vel_circle_radii[1]);

            // cs_constraint_score += (forward_vel_constraint > 0.0f) ? velocity_penalty * forward_vel_constraint : 0.0f;

            // cs_constraint_score += (backward_vel_constraint > 0.0f) ? velocity_penalty * backward_vel_constraint : 0.0f;

            // if(sol_score != nullptr && (cs_constraint_score - before_vel_score) != 0){
            //     printf("current step: %d constraint from velocity constraint:%f\n", threadIdx.x, cs_constraint_score - before_vel_score);
            // }

            // ##########################
            // foothold constraint
            // ##########################
            // foothold bound

            float2 cur_obj_circle = obj_circle;
            float2 *cur_foothold_circles = foothold_circles;
            if ((threadIdx.x & 1) != first_step_num){
                cur_obj_circle = obj_circle2;
                cur_foothold_circles = foothold_circles2;
            }
            float before_foothold = cs_constraint_score;
            float foothold_ub_constraint = (cur_individual_param[0] - (__cosf(current_state[4]) * cur_foothold_circles[0].x - __sinf(current_state[4]) * cur_foothold_circles[0].y)) * (cur_individual_param[0] - (__cosf(current_state[4]) * cur_foothold_circles[0].x - __sinf(current_state[4]) * cur_foothold_circles[0].y))
                                         + (cur_individual_param[1] - (__sinf(current_state[4]) * cur_foothold_circles[0].x + __cosf(current_state[4]) * cur_foothold_circles[0].y)) * (cur_individual_param[1] - (__sinf(current_state[4]) * cur_foothold_circles[0].x + __cosf(current_state[4]) * cur_foothold_circles[0].y))
                                         - (foothold_radii[0] * foothold_radii[0]);

            float foothold_lb_constraint = (cur_individual_param[0] - (__cosf(current_state[4]) * cur_foothold_circles[1].x - __sinf(current_state[4]) * cur_foothold_circles[1].y)) * (cur_individual_param[0] - (__cosf(current_state[4]) * cur_foothold_circles[1].x - __sinf(current_state[4]) * cur_foothold_circles[1].y))
                                         + (cur_individual_param[1] - (__sinf(current_state[4]) * cur_foothold_circles[1].x + __cosf(current_state[4]) * cur_foothold_circles[1].y)) * (cur_individual_param[1] - (__sinf(current_state[4]) * cur_foothold_circles[1].x + __cosf(current_state[4]) * cur_foothold_circles[1].y))
                                         - (foothold_radii[1] * foothold_radii[1]);

            cs_constraint_score += (foothold_ub_constraint > 0.0f) ? foothold_penalty * foothold_ub_constraint : 0.0f;

            cs_constraint_score += (foothold_lb_constraint > 0.0f) ? foothold_penalty * foothold_lb_constraint : 0.0f;

            if(sol_score != nullptr && (cs_constraint_score - before_foothold) != 0){
                printf("current step: %d constraint from foothold:%f\n", threadIdx.x, cs_constraint_score - before_foothold);
            }

            // ##########################
            // objective function
            // ##########################

            // float dist_to_target_x = fabsf(current_state[0] - (__cosf(current_state[4]) * cur_target_circle.x - __sinf(current_state[4]) * cur_target_circle.y));
            // float dist_to_target_y = fabsf(current_state[1] - (__sinf(current_state[4]) * cur_target_circle.x - __cosf(current_state[4]) * cur_target_circle.y));

            // equal to uRu^T
            cs_u_theta_cost = 5.0f * cur_individual_param[2] * cur_individual_param[2];

            // equal to xQx^T
            cs_u_theta_cost += (0.5f * current_state[2] * current_state[2] + 0.5f * current_state[3] * current_state[3]);

            float forceTrackingPenalty_x = fabsf(cur_individual_param[0] - (__cosf(current_state[4]) * cur_obj_circle.x - __sinf(current_state[4]) * cur_obj_circle.y));

            float forceTrackingPenalty_y = fabsf(cur_individual_param[1] - (__sinf(current_state[4]) * cur_obj_circle.x - __cosf(current_state[4]) * cur_obj_circle.y));

            cs_obj_score = cs_u_theta_cost + 50.0f * (forceTrackingPenalty_x * forceTrackingPenalty_x + forceTrackingPenalty_y * forceTrackingPenalty_y);

            cs_dist_to_target = target_weight * sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));

            if(threadIdx.x == N-1){
                last_dist_to_target = sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));    
                // printf("blockIdx.x:%d dist_to_target:%f\n", blockIdx.x, dist_to_target);
            }
            // cs_score = cs_obj_score + cs_constraint_score + cs_dist_to_target;
            cs_score = cs_obj_score + cs_constraint_score;
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

            if (sol_score != nullptr){
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
            }
            
            // __syncthreads();

            if (threadIdx.x == 0) {
                score[blockIdx.x] = cs_score + target_weight * last_dist_to_target;
                
                if(sol_score != nullptr){
                    // printf("sol_score:%f, sol_obj_score:%f, sol_constraint_score:%f dist_to_target:%f\n",cs_score + target_weight * last_dist_to_target, cs_obj_score, cs_constraint_score, last_dist_to_target);
                    sol_score[0] = cs_score + target_weight * last_dist_to_target;
                    sol_score[1] = cs_obj_score;
                    sol_score[2] = cs_constraint_score;
                    sol_score[3] = last_dist_to_target;
                }
            }
        }
    }


    /************************Version 2*********************** */
    // finish E*x[k]
    __device__ void matrixE_vectorX_multiply_colmajor(const float* E_col, const float* x, float* result) {
        // // 初始化结果为0
        // #pragma unroll
        // for (int i = 0; i < row_E; i++) {
        //     result[i] = 0.0f;
        // }
        
        // // 按列进行计算，适合column-major存储
        // #pragma unroll
        // for (int j = 0; j < col_E; j++) {
        //     float xj = x[j];  // 缓存x[j]
            
        //     #pragma unroll
        //     for (int i = 0; i < row_E; i++) {
        //         // column-major索引: j*row_A+i
        //         result[i] += E_col[j * row_E + i] * xj;
        //     }
        // }

        if constexpr (row_E == 5 && col_E == 5) {
            // 列0
            result[0] += E_col[0] * x[0];
            result[1] += E_col[1] * x[0];
            result[2] += E_col[2] * x[0];
            result[3] += E_col[3] * x[0];
            result[4] += E_col[4] * x[0];
            
            // 列1
            result[0] += E_col[5] * x[1];
            result[1] += E_col[6] * x[1];
            result[2] += E_col[7] * x[1];
            result[3] += E_col[8] * x[1];
            result[4] += E_col[9] * x[1];
            
            // 列2
            result[0] += E_col[10] * x[2];
            result[1] += E_col[11] * x[2];
            result[2] += E_col[12] * x[2];
            result[3] += E_col[13] * x[2];
            result[4] += E_col[14] * x[2];
            
            // 列3
            result[0] += E_col[15] * x[3];
            result[1] += E_col[16] * x[3];
            result[2] += E_col[17] * x[3];
            result[3] += E_col[18] * x[3];
            result[4] += E_col[19] * x[3];
            
            // 列4
            result[0] += E_col[20] * x[4];
            result[1] += E_col[21] * x[4];
            result[2] += E_col[22] * x[4];
            result[3] += E_col[23] * x[4];
            result[4] += E_col[24] * x[4];
        }
    }

    // Construct matrix D (D = C - HugeE X_0)
    __global__ void ConstructMatrixD(const float *E_col, float *d_D, float *cluster_N_state, float *score=nullptr){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
        if(threadIdx.x >= footstep::N)   return;
        if(score != nullptr && score[blockIdx.x] != 0.0f)   return;
        // x[k]
        float *current_state = cluster_N_state + (footstep::N + 1) * blockIdx.x * footstep::state_dims + threadIdx.x * footstep::state_dims;
        // x[k+1]
        float *next_states = cluster_N_state + (footstep::N + 1) * blockIdx.x * footstep::state_dims + (threadIdx.x + 1) * footstep::state_dims ;
        float *current_D = d_D + footstep::N * footstep::state_dims * blockIdx.x + threadIdx.x * footstep::state_dims;
        // extern __shared__ __align__(16) char smem[];

        // gemm_kernel<D_GEMM>(-1.0f, DiagE_col, current_state, 0.0f, d_D, smem);

        matrixE_vectorX_multiply_colmajor(E_col, current_state, current_D);

        for(int i = 0; i < state_dims; ++i){
            current_D[i] = next_states[i] - current_D[i];
        }
    }

    // calculate control input
    __global__ void SolveControlInputAndVelocity(float *cluster_N_state, float *cluster_u){
        if(blockIdx.x >= 1) return;
        if(threadIdx.x >= CUDA_SOLVER_POP_SIZE) return;

        float *kplus_state = nullptr;
        float *k_state = init_state;
        cluster_N_state[threadIdx.x * (footstep::N + 1) * footstep::state_dims + 2] = init_state[2];
        cluster_N_state[threadIdx.x * (footstep::N + 1) * footstep::state_dims + 3] = init_state[3];
        float *k_u = nullptr;
        const float omega = sqrtf(g / legLength);
        const float tmp1 =  sinhf(omega * T)/omega;
        const float tmp2 = coshf(omega * T);
        const float tmp3 = 1.0f - coshf(omega * T);
        const float tmp4 = -omega * sinhf(omega * T);
        // solve u_k, and v_k+1
        for(int i = 0; i < footstep::N; ++i){
            // k+1 state
            kplus_state = cluster_N_state + threadIdx.x * (footstep::N + 1) * footstep::state_dims + (i + 1) * state_dims;
            // k state
            if (i != 0) k_state = cluster_N_state + threadIdx.x * (footstep::N + 1) * footstep::state_dims + i * state_dims;

            // k u
            k_u = cluster_u + threadIdx.x * N * control_dims + i * control_dims;

            float kplus_x = kplus_state[0], kplus_y = kplus_state[1], kplus_theta = kplus_state[4];
            float k_x = k_state[0], k_y = k_state[1], k_theta = k_state[4], k_vx = k_state[2], k_vy = k_state[3];
            
            k_u[0] = (kplus_x - k_x - tmp1 * k_vx) / tmp3;
            k_u[1] = (kplus_y - k_y - tmp1 * k_vy) / tmp3;
            k_u[2] = (kplus_theta - k_theta);

            kplus_state[2] = tmp2 * k_vx + tmp4 * k_u[0];
            kplus_state[3] = tmp2 * k_vy + tmp4 * k_u[1];
        }
    }

    // Generative N step state
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void CalculateControlInput(const float *DiagF_inv_col, float *D, float *cluster_u, float *score=nullptr){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;

        if(score != nullptr && score[blockIdx.x] != 0.0f)   return;

        float *current_D = D + N * state_dims * blockIdx.x;

        // if(blockIdx.x == 0 && threadIdx.x == 0){
        //     printf("block id=0 D:\n");
        //     for(int i = 0; i < N * state_dims; ++i){
        //         printf("%f ", current_D[i]);
        //     }

        //     // printf("block id=0 D:\n");
        //     // for(int i = 0; i < N * state_dims; ++i){
        //     //     printf("%f ", current_D[i]);
        //     // }
        // }

        float *current_u = cluster_u + blockIdx.x * N * control_dims; 
        
        extern __shared__ __align__(16) char smem[];

        // matrix bigE times init_states, and record the result at N_states
        gemm_kernel<F_invD_col>(1.0f, DiagF_inv_col, current_D, 0.0f, current_u, smem);

        __syncthreads();

        // matrix bigE times init_states, and plus the result at F
        // gemm_kernel<Fu_GEMM>(1.0f, bigF, cur_individual_param, 1.0f, N_states, smem);

        // __syncthreads();
    }

    // // Construct matrix D (D = C - HugeE X_0)
    // template<int T = CUDA_SOLVER_POP_SIZE>
    // __global__ void ConstructMatrixD(const float *bigE_column, void **d_batch_D){
    //     if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
    //     // ########### 
    //     // Update State
    //     // ###########
    //     float *N_states = reinterpret_cast<float*>(d_batch_D[blockIdx.x]);

    //     // if(blockIdx.x == 0 && threadIdx.x == 0){
    //     //     for(int i = 0; i < footstep::N * footstep::state_dims; ++i){
    //     //         printf("%f ", N_states[i]);
    //     //     }
    //     //     printf("\n");
    //     // }
        
    //     extern __shared__ __align__(16) char smem[];

    //     // matrix bigE times init_states, and record the result at N_states
    //     gemm_kernel<Ex_GEMM_col>(-1.0f, bigE_column, init_state, 1.0f, N_states, smem);
    // }

    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void EvaluatePosition(float *cluster_state, float *score){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
        if(threadIdx.x >= 32)    return;

        if(threadIdx.x < N){
            float *current_state = cluster_state + blockIdx.x * (N + 1) * state_dims + threadIdx.x * state_dims;

            int current_region = -1;
            for(int i = 0; i < num_regions; ++i){
                // if(current_state[0] + current_u[0] >= all_region2[i].x && current_state[0] + current_u[0] <= all_region2[i].y && current_state[1] + current_u[1] >= all_region2[i].z && current_state[1] + current_u[1] <= all_region2[i].w){
                //     current_region = i;
                //     break;
                // }
                if(current_state[0] >= all_region[i].x && current_state[0] <= all_region[i].y && current_state[1] >= all_region[i].z && current_state[1] <= all_region[i].w){
                    current_region = i;
                    break;
                }
            }
            if(current_region == -1)    score[blockIdx.x] = CUDA_MAX_FLOAT;
        }
    }

    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void EvaluateModel2(float *cluster_u, float *cluster_state, float *score, float *sol_score = nullptr){
        if(blockIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
        if(threadIdx.x >= 32)    return;
        if(score != nullptr && score[blockIdx.x] != 0.0f)   return;
        // 32 threads don't need share memory.
        // __shared__ ALIGN(64) float N_score_sum[32];
        // __shared__ ALIGN(64) float N_obj_score_sum[32];
        // N_score_sum[threadIdx.x] = 0.0f;
        // N_obj_score_sum[threadIdx.x] = 0.0f;

        // if(blockIdx.x == 0 && threadIdx.x == 0 && sol_score!=nullptr){
        //     printf("evaluate model param:");
        //     for(int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
        //         printf("%f ", cluster_data->all_param[i]);
        //     }
        //     printf("\n");
        // }

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
            // float *current_state = nullptr;

            float *current_u = cluster_u + blockIdx.x * N * control_dims + threadIdx.x * control_dims;

            float *current_state = cluster_state + blockIdx.x * (N + 1) * state_dims + threadIdx.x * state_dims;

            // float *current_u = cluster_data->all_param + blockIdx.x * CUDA_PARAM_MAX_SIZE + threadIdx.x * control_dims;;

            // if(threadIdx.x != 0){
            //     current_state = cluster_state + blockIdx.x * N * state_dims + (threadIdx.x - 1) * state_dims;
            // }
            // else{
            //     current_state = init_state;
            // }

            // #################
            // region constraint
            // #################
            // if robot doesn't stay at any region will get a huge penalty
            // calculate the region
            // int current_region = -1;
            // for(int i = 0; i < num_regions; ++i){
            //     // if(current_state[0] + current_u[0] >= all_region2[i].x && current_state[0] + current_u[0] <= all_region2[i].y && current_state[1] + current_u[1] >= all_region2[i].z && current_state[1] + current_u[1] <= all_region2[i].w){
            //     //     current_region = i;
            //     //     break;
            //     // }
            //     if(current_state[0] + current_u[0] >= all_region[i].x && current_state[0] + current_u[0] <= all_region[i].y && current_state[1] + current_u[1] >= all_region[i].z && current_state[1] + current_u[1] <= all_region[i].w){
            //         current_region = i;
            //         break;
            //     }
            // }

            // if(current_region == -1)    cs_constraint_score += pos_penalty;

            // if(sol_score != nullptr && cs_constraint_score != 0){
            //     printf("current step: %d constraint from region:%f\n", threadIdx.x, cs_constraint_score);
            // }

            // ##########################
            // speed and theta constraint
            // ##########################
            // if robot doesn't stay at any region will get a huge penalty

            // state[0]: position x
            // state[1]: position y
            // state[2]: dot x
            // state[3]: dot y
            // state[4]: theta
            cs_constraint_score += fabsf(current_state[2]) > speed_x_ub ? state_penalty * fabsf(current_state[2]) : 0.0f;
            cs_constraint_score += fabsf(current_state[3]) > speed_y_ub ? state_penalty * fabsf(current_state[3]): 0.0f;
            cs_constraint_score += fabsf(current_state[4]) > theta_ub ? state_penalty * fabsf(current_state[4]): 0.0f;

            // ##########################
            // control constraint
            // ##########################
            // current_u[0]: ux
            // current_u[1]: uy
            // current_u[2]: u_theta
            cs_constraint_score += fabsf(current_u[0]) > ux_ub ? control_penalty : 0.0f;
            cs_constraint_score += fabsf(current_u[1]) > uy_ub ? control_penalty : 0.0f;
            cs_constraint_score += fabsf(current_u[2]) > utheta_ub ? control_penalty : 0.0f;

            // if(fabsf(current_u[0]) > ux_ub || fabsf(current_u[1]) > uy_ub || fabsf(current_u[2]) > utheta_ub){
            //     printf("current param exceed the range:%f %f %f\n", current_u[0], current_u[1], current_u[2]);
            // }
            
            // ##########################
            // velocity constraint
            // ##########################
            // float before_vel_score = cs_constraint_score;
            // float backward_vel_constraint = (current_state[2] - (__cosf(current_state[4]) * vel_circle[0].x - __sinf(current_state[4]) * vel_circle[0].y)) * (current_state[2] - (__cosf(current_state[4]) * vel_circle[0].x - __sinf(current_state[4]) * vel_circle[0].y))
            //                             + (current_state[3] - (__sinf(current_state[4]) * vel_circle[0].x + __cosf(current_state[4]) * vel_circle[0].y)) * (current_state[3] - (__sinf(current_state[4]) * vel_circle[0].x + __cosf(current_state[4]) * vel_circle[0].y))
            //                             - (vel_circle_radii[0] * vel_circle_radii[0]);

            // float forward_vel_constraint = (current_state[2] - (__cosf(current_state[4]) * vel_circle[1].x - __sinf(current_state[4]) * vel_circle[1].y)) * (current_state[2] - (__cosf(current_state[4]) * vel_circle[1].x - __sinf(current_state[4]) * vel_circle[1].y))
            //                      + (current_state[3] - (__sinf(current_state[4]) * vel_circle[1].x + __cosf(current_state[4]) * vel_circle[1].y)) * (current_state[3] - (__sinf(current_state[4]) * vel_circle[1].x + __cosf(current_state[4]) * vel_circle[1].y))
            //                      - (vel_circle_radii[1] * vel_circle_radii[1]);

            // cs_constraint_score += (forward_vel_constraint > 0.0f) ? velocity_penalty * forward_vel_constraint : 0.0f;

            // cs_constraint_score += (backward_vel_constraint > 0.0f) ? velocity_penalty * backward_vel_constraint : 0.0f;

            // if(sol_score != nullptr && (cs_constraint_score - before_vel_score) != 0){
            //     printf("current step: %d constraint from velocity constraint:%f\n", threadIdx.x, cs_constraint_score - before_vel_score);
            // }

            // ##########################
            // foothold constraint
            // ##########################
            // foothold bound

            float2 cur_obj_circle = obj_circle;
            float2 *cur_foothold_circles = foothold_circles;
            if ((threadIdx.x & 1) != first_step_num){
                cur_obj_circle = obj_circle2;
                cur_foothold_circles = foothold_circles2;
            }
            float before_foothold = cs_constraint_score;
            float foothold_ub_constraint = (current_u[0] - (__cosf(current_state[4]) * cur_foothold_circles[0].x - __sinf(current_state[4]) * cur_foothold_circles[0].y)) * (current_u[0] - (__cosf(current_state[4]) * cur_foothold_circles[0].x - __sinf(current_state[4]) * cur_foothold_circles[0].y))
                                         + (current_u[1] - (__sinf(current_state[4]) * cur_foothold_circles[0].x + __cosf(current_state[4]) * cur_foothold_circles[0].y)) * (current_u[1] - (__sinf(current_state[4]) * cur_foothold_circles[0].x + __cosf(current_state[4]) * cur_foothold_circles[0].y))
                                         - (foothold_radii[0] * foothold_radii[0]);

            float foothold_lb_constraint = (current_u[0] - (__cosf(current_state[4]) * cur_foothold_circles[1].x - __sinf(current_state[4]) * cur_foothold_circles[1].y)) * (current_u[0] - (__cosf(current_state[4]) * cur_foothold_circles[1].x - __sinf(current_state[4]) * cur_foothold_circles[1].y))
                                         + (current_u[1] - (__sinf(current_state[4]) * cur_foothold_circles[1].x + __cosf(current_state[4]) * cur_foothold_circles[1].y)) * (current_u[1] - (__sinf(current_state[4]) * cur_foothold_circles[1].x + __cosf(current_state[4]) * cur_foothold_circles[1].y))
                                         - (foothold_radii[1] * foothold_radii[1]);

            cs_constraint_score += (foothold_ub_constraint > 0.0f) ? foothold_penalty * foothold_ub_constraint : 0.0f;

            cs_constraint_score += (foothold_lb_constraint > 0.0f) ? foothold_penalty * foothold_lb_constraint : 0.0f;

            // if(sol_score != nullptr && (cs_constraint_score - before_foothold) != 0){
            //     printf("current step: %d constraint from foothold:%f\n", threadIdx.x, cs_constraint_score - before_foothold);
            // }

            // ##########################
            // objective function
            // ##########################

            // float dist_to_target_x = fabsf(current_state[0] - (__cosf(current_state[4]) * cur_target_circle.x - __sinf(current_state[4]) * cur_target_circle.y));
            // float dist_to_target_y = fabsf(current_state[1] - (__sinf(current_state[4]) * cur_target_circle.x - __cosf(current_state[4]) * cur_target_circle.y));

            // equal to uRu^T
            cs_u_theta_cost = 5.0f * current_u[2] * current_u[2];

            // equal to xQx^T
            cs_u_theta_cost += (0.5f * current_state[2] * current_state[2] + 0.5f * current_state[3] * current_state[3]);

            float forceTrackingPenalty_x = fabsf(current_u[0] - (__cosf(current_state[4]) * cur_obj_circle.x - __sinf(current_state[4]) * cur_obj_circle.y));

            float forceTrackingPenalty_y = fabsf(current_u[1] - (__sinf(current_state[4]) * cur_obj_circle.x - __cosf(current_state[4]) * cur_obj_circle.y));

            cs_obj_score = cs_u_theta_cost + 50.0f * (forceTrackingPenalty_x * forceTrackingPenalty_x + forceTrackingPenalty_y * forceTrackingPenalty_y);

            cs_dist_to_target = target_weight * sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));

            if(threadIdx.x == N-1){
                last_dist_to_target = sqrtf((current_state[0] - target_pos.x) * (current_state[0] - target_pos.x) + (current_state[1] - target_pos.y) * (current_state[1] - target_pos.y));    
                // printf("blockIdx.x:%d dist_to_target:%f\n", blockIdx.x, dist_to_target);
            }
            // cs_score = cs_obj_score + cs_constraint_score + cs_dist_to_target;
            cs_score = cs_obj_score + cs_constraint_score;
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

            if (sol_score != nullptr){
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
            }
            
            // __syncthreads();

            if (threadIdx.x == 0) {
                score[blockIdx.x] = cs_score + target_weight * last_dist_to_target;
                
                if(sol_score != nullptr){
                    // printf("sol_score:%f, sol_obj_score:%f, sol_constraint_score:%f dist_to_target:%f\n",cs_score + target_weight * last_dist_to_target, cs_obj_score, cs_constraint_score, last_dist_to_target);
                    sol_score[0] = cs_score + target_weight * last_dist_to_target;
                    sol_score[1] = cs_obj_score;
                    sol_score[2] = cs_constraint_score;
                    sol_score[3] = last_dist_to_target;
                }
            }
        }
    }
}

#endif