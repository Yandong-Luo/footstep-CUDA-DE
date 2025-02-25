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
    __global__ void DecodeParameters2State(CudaParamClusterData<T>* new_cluster_data, bezier_curve::BezierCurve* curve, float *cluster_state){
        int step_id = blockIdx.x;
        int sol_id = threadIdx.x;

        if(step_id >= CURVE_NUM_STEPS)  return;
        if(sol_id >= CUDA_SOLVER_POP_SIZE) return;
        
        // construct the complete bezier curve param 
        float curve_param[2 * BEZIER_SIZE];
        // float *curve_param_y;
        int construct_idx = 0;
        float *current_sol_param = new_cluster_data->all_param + sol_id * CUDA_PARAM_MAX_SIZE;
        int bias = new_cluster_data->dims / 2;

        // if (blockIdx.x == 0){
        //     printf("cluster param: ");
        //     for(int i = 0; i < new_cluster_data->dims; ++i){
        //         printf("%f ",current_sol_param[i]);
        //     }
        //     printf("\n");
        // }
        for(int i = 0, j = 0; i < BEZIER_SIZE; ++i){
            if(i == curve->fixed_point_idx[j]){
                curve_param[i] = curve->control_points[i].x;
                curve_param[i + BEZIER_SIZE] = curve->control_points[i].y;
                j++;
                // continue;
            }
            else{
                // curve_param[i] = current_sol_param[i-j];
                // curve_param[i + BEZIER_SIZE] = current_sol_param[i-j + bias];
                curve_param[i] = curve->control_points[i].x = current_sol_param[i-j];
                curve_param[i + BEZIER_SIZE] = curve->control_points[i].y = current_sol_param[i-j + bias];
            }
        }
        __syncthreads();
        if(blockIdx.x == 0 && threadIdx.x == 0){
            for(int i = 0; i < BEZIER_SIZE; ++i){
                printf("Point %d and its' value (%f, %f) pointer form:(%f, %f)\n",i, curve->control_points[i].x, curve->control_points[i].y, curve_param[i], curve_param[i+BEZIER_SIZE]);
            }
        }
        
        float *current_state = cluster_state + sol_id * footstep::state_dims * CURVE_NUM_STEPS + step_id * footstep::state_dims;
        // printf("block:%d, start idx:%d\n", blockIdx.x ,sol_id * footstep::state_dims * (footstep::N + 1) + step_id * footstep::state_dims);
        bezier_curve::GetTrajStateFromBezierBasedLookup(curve, curve_param, step_id, 0, BEZIER_SIZE-1, BEZIER_SIZE, 2*BEZIER_SIZE-1, current_state);
        // __syncthreads();

        // printf("step %d and its' state (%f, %f, %f, %f, %f)\n",blockIdx.x, current_state[0], current_state[1], current_state[2], current_state[3], current_state[4]);

        // printf("step %d and its' state (%f, %f, %f, %f, %f)\n",blockIdx.x, current_state[0], current_state[1], current_state[2], current_state[3], current_state[4]);
    }
}
#endif