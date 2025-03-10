#ifndef CUDAPROCESS_DIFF_EVOLUTION_DECODER_H
#define CUDAPROCESS_DIFF_EVOLUTION_DECODER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include "data_type.h"
#include "curve/bezier_curve.cuh"

namespace cudaprocess{
    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void InitParameter(CudaEvolveData* evolve_data, int size, CudaParamClusterData<T>* new_cluster_data, CudaParamClusterData<T*3>* old_cluster_data, float *uniform_data){
        int idx = threadIdx.x;
        if (idx >= size)    return;

        // initial evolve data
        // evolve_data->new_cluster_vec->data[idx].con_var_dims = evolve_data->problem_param.con_var_dims;
        // evolve_data->new_cluster_vec->data[idx].int_var_dims = evolve_data->problem_param.int_var_dims;
        // evolve_data->new_cluster_vec->data[idx].dims = evolve_data->dims;
        // evolve_data->new_cluster_vec->data[idx].fitness = 0.f;
        // evolve_data->new_cluster_vec->data[idx].cur_scale_f1 = 0.5f;
        // evolve_data->new_cluster_vec->data[idx].cur_scale_f = 0.5f;
        // evolve_data->new_cluster_vec->data[idx].cur_Cr = 0.5f;

        // initial new_cluster_data
        for (int i = 0; i < evolve_data->problem_param.dims; ++i) {
            // evolve_data->new_cluster_vec->data[idx].param[i] = 0;
            if (i < evolve_data->problem_param.con_var_dims){
                // each parameters were decode as a vector with the length of CUDA_PARAM_MAX_SIZE
                // printf("%f\n", uniform_data[idx * CUDA_PARAM_MAX_SIZE + i]);
                new_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i] = evolve_data->lower_bound[i] + uniform_data[idx * CUDA_PARAM_MAX_SIZE + i] * (evolve_data->upper_bound[i] - evolve_data->lower_bound[i]);
            }
            else{
                int generate_int = evolve_data->lower_bound[i] + uniform_data[idx * CUDA_PARAM_MAX_SIZE + i] * (evolve_data->upper_bound[i] + 1 - evolve_data->lower_bound[i]);
                if (generate_int == evolve_data->upper_bound[i] + 1 )   generate_int = evolve_data->upper_bound[i];
                new_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i] = generate_int;
            }
            // printf("wdawd:%d, problem_param.con_var_dims:%d\n", evolve_data->problem_param.dims, evolve_data->problem_param.con_var_dims);
            // if (i == 0) printf("index:%d lower bound:%f, upper bound:%f, value:%f\n",i, evolve_data->lower_bound[i], evolve_data->upper_bound[i], new_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i]);
        }
        // printf("\n");
        if(idx == 0){
            old_cluster_data->con_var_dims = new_cluster_data->con_var_dims = evolve_data->problem_param.con_var_dims;
            old_cluster_data->int_var_dims = new_cluster_data->int_var_dims = evolve_data->problem_param.int_var_dims;
            old_cluster_data->dims = new_cluster_data->dims = evolve_data->problem_param.dims;
            // printf("Thread 0: first few params = [%f, %f, %f]\n",
            // new_cluster_data->all_param[0],
            // new_cluster_data->all_param[1],
            // new_cluster_data->all_param[2]);
        }

        new_cluster_data->fitness[idx] = CUDA_MAX_FLOAT;
        // new_cluster_data->objective_score[idx] = CUDA_MAX_FLOAT;
        // new_cluster_data->constraint_score[idx] = CUDA_MAX_FLOAT;
        new_cluster_data->lshade_param[idx * 3 + 0] = 0.8f;                        // scale_f
        new_cluster_data->lshade_param[idx * 3 + 1] = 0.8f;                        // scale_f1
        new_cluster_data->lshade_param[idx * 3 + 2] = 0.9f;                        // crossover

        // initial old_cluster_data
        old_cluster_data->fitness[idx] = CUDA_MAX_FLOAT;
        // old_cluster_data->objective_score[idx] = CUDA_MAX_FLOAT;
        // old_cluster_data->constraint_score[idx] = CUDA_MAX_FLOAT;

        // printf("Finish the initialization of thread id:%d\n", idx);
    }


    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void RestOldParameter(CudaEvolveData* evolve_data, int size, CudaParamClusterData<T*3>* old_cluster_data, float *uniform_data, float keep_rate){
        int idx = threadIdx.x;
        if (idx >= size || idx < keep_rate * size)    return;

        // reset old_cluster_data
        for (int i = 0; i < evolve_data->problem_param.dims; ++i) {
            // evolve_data->new_cluster_vec->data[idx].param[i] = 0;
            if (i < evolve_data->problem_param.con_var_dims){
                // each parameters were decode as a vector with the length of CUDA_PARAM_MAX_SIZE
                // printf("%f\n", uniform_data[idx * CUDA_PARAM_MAX_SIZE + i]);
                old_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i] = evolve_data->lower_bound[i] + uniform_data[idx * CUDA_PARAM_MAX_SIZE + i] * (evolve_data->upper_bound[i] - evolve_data->lower_bound[i]);
            }
            else{
                int generate_int = evolve_data->lower_bound[i] + uniform_data[idx * CUDA_PARAM_MAX_SIZE + i] * (evolve_data->upper_bound[i] + 1 - evolve_data->lower_bound[i]);
                if (generate_int == evolve_data->upper_bound[i] + 1 )   generate_int = evolve_data->upper_bound[i];
                old_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i] = generate_int;
            }
            // printf("wdawd:%d, problem_param.con_var_dims:%d\n", evolve_data->problem_param.dims, evolve_data->problem_param.con_var_dims);
            // if (i == 0) printf("index:%d lower bound:%f, upper bound:%f, value:%f\n",i, evolve_data->lower_bound[i], evolve_data->upper_bound[i], new_cluster_data->all_param[idx * CUDA_PARAM_MAX_SIZE + i]);
        }
        // printf("\n");
        // if(idx == 0){
        //     old_cluster_data->con_var_dims = old_cluster_data->con_var_dims = evolve_data->problem_param.con_var_dims;
        //     old_cluster_data->int_var_dims = new_cluster_data->int_var_dims = evolve_data->problem_param.int_var_dims;
        //     old_cluster_data->dims = new_cluster_data->dims = evolve_data->problem_param.dims;
        //     // printf("Thread 0: first few params = [%f, %f, %f]\n",
        //     // new_cluster_data->all_param[0],
        //     // new_cluster_data->all_param[1],
        //     // new_cluster_data->all_param[2]);
        // }

        old_cluster_data->fitness[idx] = CUDA_MAX_FLOAT;
        // new_cluster_data->objective_score[idx] = CUDA_MAX_FLOAT;
        // new_cluster_data->constraint_score[idx] = CUDA_MAX_FLOAT;
        // old_cluster_data->lshade_param[idx * 3 + 0] = 0.8f;                        // scale_f
        // old_cluster_data->lshade_param[idx * 3 + 1] = 0.8f;                        // scale_f1
        // old_cluster_data->lshade_param[idx * 3 + 2] = 0.9f;                        // crossover

        // initial old_cluster_data
        // old_cluster_data->fitness[idx] = CUDA_MAX_FLOAT;
        // old_cluster_data->objective_score[idx] = CUDA_MAX_FLOAT;
        // old_cluster_data->constraint_score[idx] = CUDA_MAX_FLOAT;

        // printf("Finish the initialization of thread id:%d\n", idx);
    }

    template<int T = CUDA_SOLVER_POP_SIZE>
    __global__ void DecodeParameters2State(CudaParamClusterData<T>* new_cluster_data, bezier_curve::BezierCurve* curve, float *cluster_state, float *all_curve_param, bool record_best=false){
        int step_id = blockIdx.x;
        int sol_id = threadIdx.x;
        if(record_best){
            sol_id = blockIdx.x;
            step_id = threadIdx.x;
        }

        if(step_id >= CURVE_NUM_STEPS)  return;
        if(sol_id >= CUDA_SOLVER_POP_SIZE) return;
        
        // construct the complete bezier curve param 
        // float curve_param[3 * BEZIER_SIZE];
        float *curve_param = all_curve_param + sol_id * 3 * BEZIER_SIZE;
        // float *curve_param_y;
        int construct_idx = 0;
        float *current_sol_param = new_cluster_data->all_param + sol_id * CUDA_PARAM_MAX_SIZE;

        // if (blockIdx.x == 0 && threadIdx.x == 0){
        //     printf("cluster param: ");
        //     for(int i = 0; i < new_cluster_data->dims; ++i){
        //         printf("%f ",current_sol_param[i]);
        //     }
        //     printf("\n");
        // }
        // __syncthreads();
        int xy_param_idx = 0;
        // int y_param_idx = 0;
        int theta_param_idx = 0;
        for(int i = 0; i < BEZIER_SIZE; ++i){
            xy_param_idx = (curve->is_point_xy_fixed[i]) ? xy_param_idx + 1 : xy_param_idx;
            theta_param_idx = (curve->is_theta_point_fixed[i]) ? theta_param_idx + 1: theta_param_idx;

            curve_param[i] = !curve->is_point_xy_fixed[i] * current_sol_param[i-xy_param_idx] + curve->is_point_xy_fixed[i] * curve->control_points[i].x;
            curve_param[i + BEZIER_SIZE] = !curve->is_point_xy_fixed[i] * current_sol_param[i-xy_param_idx + Y_START] + curve->is_point_xy_fixed[i] * curve->control_points[i].y;
            curve_param[i + 2*BEZIER_SIZE] = !curve->is_theta_point_fixed[i] * current_sol_param[i-theta_param_idx + THETA_START] + curve->control_points[i].z;
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     curve->control_points[i].x = curve_param[i];
            //     curve->control_points[i].y = curve_param[i+BEZIER_SIZE];
            //     curve->control_points[i].z = curve_param[i+2*BEZIER_SIZE];
            // }
            // if(i == curve->xy_fixed_point_idx[xy_param_idx]){
            //     curve_param[i] = curve->control_points[i].x;
            //     curve_param[i + BEZIER_SIZE] = curve->control_points[i].y;
            //     xy_param_idx++;
            //     // continue;
            // }
            // else{
            //     // curve_param[i] = current_sol_param[i-j];
            //     // curve_param[i + BEZIER_SIZE] = current_sol_param[i-j + bias];
            //     curve_param[i] = current_sol_param[i-xy_param_idx];
            //     curve->control_points[i].x = current_sol_param[i-xy_param_idx];
            //     curve_param[i + BEZIER_SIZE] =current_sol_param[i-xy_param_idx + Y_START];
            //     curve->control_points[i].y = current_sol_param[i-xy_param_idx + Y_START];
            //     // if(blockIdx.x == 0 && threadIdx.x == 0)     printf("Using param %f for x, %f for y\n", current_sol_param[i-xy_param_idx], current_sol_param[i-xy_param_idx + Y_START]);
            // }

            // if(i == curve->theta_fixed_point_idx[theta_param_idx]){
            //     curve_param[i + 2*BEZIER_SIZE] = curve->control_points[i].z;
            //     theta_param_idx++;
            // }
            // else{
            //     curve_param[i + 2*BEZIER_SIZE] = current_sol_param[i-theta_param_idx + THETA_START];
            //     curve->control_points[i].z = current_sol_param[i-theta_param_idx + THETA_START];
            // }
            // __syncthreads();
        }
        
        // if(blockIdx.x == 0 && threadIdx.x == 0){
            
        //     printf("%d %d\n", xy_param_idx, theta_param_idx);
        //     for(int i = 0; i < BEZIER_SIZE; ++i){
        //         printf("Point %d and its' value (%f, %f %f) pointer form:(%f, %f %f)\n",i, curve->control_points[i].x, curve->control_points[i].y, curve->control_points[i].z, curve_param[i], curve_param[i+BEZIER_SIZE], curve_param[i+2*BEZIER_SIZE]);
        //     }
        // }
        
        float *current_state = cluster_state + sol_id * footstep::state_dims * CURVE_NUM_STEPS + step_id * footstep::state_dims;
        // printf("block:%d, start idx:%d\n", blockIdx.x ,sol_id * footstep::state_dims * (footstep::N + 1) + step_id * footstep::state_dims);
        bezier_curve::GetTrajStateFromBezierBasedLookup(curve, curve_param, step_id, 0, BEZIER_SIZE-1, BEZIER_SIZE, 2*BEZIER_SIZE-1, 2*BEZIER_SIZE, 3*BEZIER_SIZE-1, current_state);
        // __syncthreads();

        // record all state except the initial state
        // if(step_id != 0){
        //     // int tmp_idx = sol_id * footstep::state_dims * footstep::N + (step_id - 1) * footstep::state_dims;
        //     // d_D[tmp_idx] = current_state[0];
        //     // d_D[tmp_idx+1] = current_state[1];
        //     // d_D[tmp_idx+2] = current_state[2];
        //     // d_D[tmp_idx+3] = current_state[3];
        //     // d_D[tmp_idx+4] = current_state[4];
            
        //     int row = (step_id - 1) * footstep::state_dims;
        //     float *sol_data = reinterpret_cast<float*>(d_batch_D[sol_id]);
        //     sol_data[row] = current_state[0];
        //     sol_data[row + 1] = current_state[1];
        //     sol_data[row + 2] = current_state[2];
        //     sol_data[row + 3] = current_state[3];
        //     sol_data[row + 4] = current_state[4];
        // }
        // printf("step %d and its' state (%f, %f, %f, %f, %f)\n",blockIdx.x, current_state[0], current_state[1], current_state[2], current_state[3], current_state[4]);

        // printf("step %d and its' state (%f, %f, %f, %f, %f)\n",blockIdx.x, current_state[0], current_state[1], current_state[2], current_state[3], current_state[4]);
    }

    // template<int T = CUDA_SOLVER_POP_SIZE>
    // __global__ void DecodeParameters2State(float *cluster_state){
    //     int step_id = blockIdx.x;
    //     int sol_id = threadIdx.x;
        
    //     float *current_state = cluster_state + (footstep::N + 1) * blockIdx.x * footstep::state_dims + threadIdx.x * footstep::state_dims;
    //     // x[k+1]
    //     float *next_states = cluster_state + (footstep::N + 1) * blockIdx.x * footstep::state_dims + (threadIdx.x + 1) * footstep::state_dims ;
        
    //     current_state[2] = next_states[0] - current_state[0])
    // }
}
#endif