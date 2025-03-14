#ifndef CUDAPROCESS_DIFF_EVOLVE_H
#define CUDAPROCESS_DIFF_EVOLVE_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "diff_evolution_solver/data_type.h"

namespace cudaprocess{
    template <int T = CUDA_SOLVER_POP_SIZE>
    __global__ void DuplicateBestAndReorganize(int epoch, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, int copy_num){
        // if(epoch > 0)   return;
        int size = 2 * T;

        if (epoch == 0){
            size = T;
        }

        int param_id = blockIdx.x;
        int sol_id = threadIdx.x;

        float param;
        float best_fitness;
        
        if (sol_id < size + copy_num){
            if (sol_id <= copy_num){
                // For the first copy_num solutions, use the parameters of the best solution (sol_id=0)
                param = old_param->all_param[param_id];
                if (param_id == 0)  best_fitness = old_param->fitness[0];   // copy the best fitness
            }
            else{
                // for the remaining solution, use the original parameters
                param = old_param->all_param[(sol_id - copy_num) * CUDA_PARAM_MAX_SIZE + param_id];
                if(param_id == 0){
                    best_fitness = old_param->fitness[sol_id - copy_num];
                }
            }
        }
        // wait for all thread finish above all copy step
        __syncthreads();
        // reorganize
        
        if(sol_id < size + copy_num){
            old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = param;
            if (param_id == 0){
                old_param->fitness[sol_id] = best_fitness;
                if(sol_id == 0){
                    old_param->len = size + copy_num;
                }
            }
        }
    }

    template <int T = CUDA_SOLVER_POP_SIZE>
    __global__ void DuplicateBestAndReorganize2(int epoch, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, int copy_num){
        // if(epoch > 0)   return;
        int size = 2 * T;

        if (epoch == 0){
            size = T;
        }

        int sol_id = blockIdx.x;
        int param_id = threadIdx.x;  // 交换这两行，因为现在 blockIdx.x 对应solution，threadIdx.x 对应parameter

        float param;
        float best_fitness;
        
        if (sol_id < size + copy_num){
            if (sol_id <= copy_num){
                // For the first copy_num solutions, use the parameters of the best solution (sol_id=0)
                param = old_param->all_param[param_id];
                if (param_id == 0)  best_fitness = old_param->fitness[0];   // copy the best fitness
            }
            else{
                // for the remaining solution, use the original parameters
                param = old_param->all_param[(sol_id - copy_num) * CUDA_PARAM_MAX_SIZE + param_id];
                if(param_id == 0){
                    best_fitness = old_param->fitness[sol_id - copy_num];
                }
            }
        }
        // wait for all thread finish above all copy step
        __syncthreads();
        // reorganize
        
        if(sol_id < size + copy_num){
            old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = param;
            if (param_id == 0){
                old_param->fitness[sol_id] = best_fitness;
                if(sol_id == 0){
                    old_param->len = size + copy_num;
                }
            }
        }
    }

    /**
     * param_diversity
     */
    // __global__ void calculate_cluster_diversity(const CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, float *diversity){
    //     // __shared__ float diversity[T];
    //     int sol_id = threadIdx.x;
    //     int param_id = blockIdx.x;

    //     const int warp_size = 32;
    //     const int warp_num = (CUDA_SOLVER_POP_SIZE >> 5);
    //     const int block_warp_num = (CUDA_PARAM_MAX_SIZE >> 5);
    //     int warp_id = (threadIdx.x >> 5);
    //     int lane_id = (threadIdx.x & 31);

    //     // record the one parameter difference
    //     __shared__ float sm_diff_sum[warp_num];
    //     // record all parameter difference
    //     __shared__ float sm_param_diff[CUDA_PARAM_MAX_SIZE];
    //     // record the result of block reduction sum
    //     __shared__ float sm_block_sum[block_warp_num];

    //     // param_diff_sum[warp_id] = 0.0; 

    //     // the difference between current individual and best individual
    //     float diff = 0.0f;

    //     // skip the first one (the best) and last one (the worst)
    //     if(sol_id > 0 && sol_id < CUDA_SOLVER_POP_SIZE - 1){
    //         diff = fabsf(old_param->all_param[param_id] - old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id]);
    //     }
        
    //     float diff_sum = diff;
    //     diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 16);
    //     diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 8);
    //     diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 4);
    //     diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 2);
    //     diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 1);

    //     if(lane_id == 0)    sm_diff_sum[warp_id] = (diff_sum / static_cast<float>(warp_size));

    //     __syncthreads();
        
    //     if(threadIdx.x < warp_num){
    //         float current_param_diff = sm_diff_sum[threadIdx.x];
    //         if (warp_num == 32){                // also means POP_SIZE = 1024
    //             current_param_diff += __shfl_down_sync(0xffffffff, current_param_diff, 16);
    //             current_param_diff += __shfl_down_sync(0x0000ffff, current_param_diff, 8);
    //             current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
    //             current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
    //             current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
    //         }
    //         else if (warp_num == 16) {          // also means POP_SIZE = 512
    //             current_param_diff += __shfl_down_sync(0x0000ffff, current_param_diff, 8);
    //             current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
    //             current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
    //             current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
    //         }
    //         else if (warp_num == 8) {            // also means POP_SIZE = 256
    //             current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
    //             current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
    //             current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
    //         }
    //         else if (warp_num == 4) {            // also means POP_SIZE = 128
    //             current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
    //             current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
    //         }
    //         else if (warp_num == 2) {           // also means POP_SIZE = 64    
    //             current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
    //         }
    //         if(threadIdx.x == 0) sm_param_diff[param_id] = (current_param_diff / static_cast<float>(warp_num));
    //     }
    //     __syncthreads();

    //     // 在每个block中，计算完sm_param_diff[blockIdx.x]后
    //     if(threadIdx.x == 0) {  // 只需要一个线程
    //         // 每个block直接添加自己的贡献到最终结果
    //         atomicAdd(diversity, sm_param_diff[blockIdx.x] / static_cast<float>(CUDA_PARAM_MAX_SIZE));
    //     }
        
    //     // // block parallel reduction sum 
    //     // if(threadIdx.x < CUDA_PARAM_MAX_SIZE){
    //     //     float block_warp_sum = sm_param_diff[threadIdx.x];
    //     //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 16);
    //     //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 8);
    //     //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 4);
    //     //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 2);
    //     //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 1);

    //     //     sm_block_sum[warp_id] = (block_warp_sum / static_cast<float>(block_warp_num));
    //     // }
    //     // __syncthreads();

    //     // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     //     diversity[0] = 0.0f;
    //     //     for(int i = 0; i < block_warp_num; ++i){
    //     //         diversity[0] += sm_block_sum[i];
    //     //     }
    //     // }
    // }

    template <CudaEvolveType SearchType = CudaEvolveType::GLOBAL>
    __global__ void CudaEvolveProcess(int epoch, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, float *uniform_data,
                                      float *normal_data, CudaEvolveData *evolve_data, int pop_size, float eliteRatio){
        const int warp_num = blockDim.x >> 5;
        __shared__ float shared_scale_f;
        __shared__ float shared_scale_f1;
        __shared__ float shared_crossover;
        __shared__ int shared_elite_strategy;
        __shared__ int shared_guide_idx;
        __shared__ int shared_mutation_start;
        __shared__ int shared_parent_idx[3];
        __shared__ int s_first_mutation[8];

        int dims = evolve_data->problem_param.dims, con_dims = evolve_data->problem_param.con_var_dims;
        // int int_dims = evolve_data->problem_param.int_var_dims;

        int sol_idx = blockIdx.x;
        int warp_id = threadIdx.x >> 5;     // 线程所在的warp id  threadIdx.x >> 5 == threadIdx.x / 32
        int lane_id = threadIdx.x & 31;   // 线程在warp内的id     threadIdx.x & 0x1F == threadIdx.x & 31 == threadIdx.x % 32
        // int param_idx = threadIdx.x;
        int UsingEliteStrategy = 0;
        int guideEliteIdx = 0;

        float crossover, scale_f, scale_f1;

        int mutationStartDim;

        float origin_param, best_origin_param, result_param;

        // Avoid using the same random number in the same place as other functions
        int normal_rnd_evolve_pos = (epoch % REGENRATE_RANDOM_FREQUENCE) * CUDA_SOLVER_POP_SIZE * 3 + sol_idx * 3;
        int uniform_rnd_evolve_pos = (epoch % REGENRATE_RANDOM_FREQUENCE) * CUDA_SOLVER_POP_SIZE * (2 * CUDA_PARAM_MAX_SIZE + 5) + sol_idx * (2 * CUDA_PARAM_MAX_SIZE + 5);

        int selected_parent_idx;
        int parent_param_idx[3], sorted_parent_param_idx[3];
        if (threadIdx.x < 3){
            int total_len = old_param->len;
            // Initialize 3 index that increases with threadIdx.x to ensure that subsequent sorting does not require too frequent operations
            selected_parent_idx = min((int)floor(uniform_data[uniform_rnd_evolve_pos + threadIdx.x] * (total_len - threadIdx.x - 1)), total_len - threadIdx.x - 2);
        }
        // Due to the selected_parent_idx was calculated in thread 0-2, we need let thread 0 know all result of selected_parent_idx for future sorting part
        // Use warp shuffle to share indices between threads
        parent_param_idx[0] = selected_parent_idx;
        parent_param_idx[1] = __shfl_sync(0x0000000f, selected_parent_idx, 1);
        parent_param_idx[2] = __shfl_sync(0x0000000f, selected_parent_idx, 2);

        if(threadIdx.x == 0){
            // sorting part
            // Ensure that the parent individuals selected in the differential evolution algorithm are different and non-repetitive
            // printf("====wft: %d, %d, %d, total len: %d\n", parent_param_idx[0], parent_param_idx[1], parent_param_idx[2], old_param->len);
            for (int i = 0; i < 3; ++i) {
                sorted_parent_param_idx[i] = parent_param_idx[i];
                for (int j = 0; j < i; ++j) {
                    if (parent_param_idx[j] <= sorted_parent_param_idx[i]) sorted_parent_param_idx[i]++;
                }
                parent_param_idx[i] = sorted_parent_param_idx[i];
                // 插入排序
                for (int j = i; j > 0; j--) {
                    if (parent_param_idx[j] < parent_param_idx[j - 1]) {
                        int tmp = parent_param_idx[j];
                        parent_param_idx[j] = parent_param_idx[j - 1];
                        parent_param_idx[j - 1] = tmp;
                    }
                }
                if (sorted_parent_param_idx[i] >= blockIdx.x)   sorted_parent_param_idx[i]++;
            }
            // printf("wft####: %d, %d, %d, %d, %d, %d, total len: %d\n", parent_param_idx[0], parent_param_idx[1], parent_param_idx[2], sorted_parent_param_idx[0], sorted_parent_param_idx[1], sorted_parent_param_idx[2], old_param->len);

            scale_f = min(1.f, max(normal_data[normal_rnd_evolve_pos] * 0.01f + evolve_data->hist_lshade_param.scale_f, 0.1));
            scale_f1 = min(1.f, max(normal_data[normal_rnd_evolve_pos + 1] * 0.01f + evolve_data->hist_lshade_param.scale_f, 0.1));
            crossover = min(1.f, max(normal_data[normal_rnd_evolve_pos + 2] * 0.01f + evolve_data->hist_lshade_param.Cr, 0.1));

            mutationStartDim = min((int)floor(uniform_data[uniform_rnd_evolve_pos + 3] * dims), dims - 1);

            int num_top = int(pop_size * evolve_data->problem_param.top_ratio) + 1;
            // The index of an individual randomly selected from the top proportion of high-quality individuals in the population
            // Due to uniform_rnd_evolve_pos to uniform_rnd_evolve_pos + 3 have been used for parent_param_idx and mutationStartDim. So, starting from 4
            guideEliteIdx = max(0, min((num_top - 1), (int)floor(uniform_data[uniform_rnd_evolve_pos + 4] * num_top)));

            if (sol_idx * 1.f < pop_size * eliteRatio)  UsingEliteStrategy = 1;

            shared_scale_f = scale_f;
            shared_scale_f1 = scale_f1;
            shared_crossover = crossover;
            shared_elite_strategy = UsingEliteStrategy;
            shared_guide_idx = guideEliteIdx;
            shared_mutation_start = mutationStartDim;
            shared_parent_idx[0] = sorted_parent_param_idx[0];
            shared_parent_idx[1] = sorted_parent_param_idx[1];
            shared_parent_idx[2] = sorted_parent_param_idx[2];
        }
        // make sure all parameter have been calculated
        // __syncwarp();

        __syncthreads();

        // // 使用 warp shuffle 广播这些值给其他线程
        // scale_f = __shfl_sync(0x0000ffff, scale_f, 0);
        // scale_f1 = __shfl_sync(0x0000ffff, scale_f1, 0);
        // crossover = __shfl_sync(0x0000ffff, crossover, 0);
        // UsingEliteStrategy = __shfl_sync(0x0000ffff, UsingEliteStrategy, 0);
        // guideEliteIdx = __shfl_sync(0x0000ffff, guideEliteIdx, 0);

        // int parent1_idx = sorted_parent_param_idx[0], parent2_idx = sorted_parent_param_idx[1];
        // int mutant_idx = sorted_parent_param_idx[2];

        // // printf("previous wft: %d, %d, %d, total len: %d\n", parent1_idx, parent2_idx, mutant_idx, old_param->len);

        // parent1_idx = __shfl_sync(0x0000ffff, parent1_idx, 0);
        // parent2_idx = __shfl_sync(0x0000ffff, parent2_idx, 0);
        // mutant_idx = __shfl_sync(0x0000ffff, mutant_idx, 0);

        // // Other threads need to obtain the broadcasted data
        // sorted_parent_param_idx[0] = parent1_idx;
        // sorted_parent_param_idx[1] = parent2_idx;
        // sorted_parent_param_idx[2] = mutant_idx;

        scale_f = shared_scale_f;
        scale_f1 = shared_scale_f1;
        crossover = shared_crossover;
        UsingEliteStrategy = shared_elite_strategy;
        guideEliteIdx = shared_guide_idx;
        mutationStartDim = shared_mutation_start;
        
        // 更新父代索引
        sorted_parent_param_idx[0] = shared_parent_idx[0];
        sorted_parent_param_idx[1] = shared_parent_idx[1];
        sorted_parent_param_idx[2] = shared_parent_idx[2];
        
        // int totalsize = old_param->len;
        // check the random_idx valid or not
        if (sorted_parent_param_idx[0] >= old_param->len || sorted_parent_param_idx[1] >= old_param->len || sorted_parent_param_idx[2] >= old_param->len || old_param->len >= pop_size * 2 + 10) {
            printf("wft: %d, %d, %d, total len: %d\n", sorted_parent_param_idx[0], sorted_parent_param_idx[1], sorted_parent_param_idx[2], old_param->len);
        }

        // record the parameter for mutant
        float mutant_param[3];
        mutant_param[0] = old_param->all_param[sorted_parent_param_idx[0] * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        mutant_param[1] = old_param->all_param[sorted_parent_param_idx[1] * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        mutant_param[2] = old_param->all_param[sorted_parent_param_idx[2] * CUDA_PARAM_MAX_SIZE + threadIdx.x];

        if(UsingEliteStrategy){
            // Use the parameters corresponding to the current individual idx as the basis
            origin_param = old_param->all_param[sol_idx * CUDA_PARAM_MAX_SIZE + threadIdx.x];
        } else{
            // Use the best parameter as the basis
            origin_param = old_param->all_param[threadIdx.x];
        }

        // initial result param
        result_param = origin_param;

        // load the random param from top level as best
        best_origin_param = old_param->all_param[guideEliteIdx * CUDA_PARAM_MAX_SIZE + threadIdx.x];

        float f = (threadIdx.x >= con_dims) ? scale_f1 : scale_f;
        if (SearchType == CudaEvolveType::GLOBAL){
            float mutant_prob = uniform_data[uniform_rnd_evolve_pos + 5 + threadIdx.x];

            // initial the firstMutationDimIdx by last one
            int firstMutationDimIdx = CUDA_PARAM_MAX_SIZE;

            // crossover select
            if(mutant_prob > (UsingEliteStrategy ? crossover : 0.9f) && threadIdx.x < dims){
                firstMutationDimIdx = threadIdx.x;
            }

            // parallel reduce
            int tmp_idx = __shfl_down_sync(0xffffffff, firstMutationDimIdx, 16);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0xffffffff, firstMutationDimIdx, 8);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0xffffffff, firstMutationDimIdx, 4);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0xffffffff, firstMutationDimIdx, 2);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            tmp_idx = __shfl_down_sync(0xffffffff, firstMutationDimIdx, 1);
            firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            // let all thread know the firstMutationDimIdx
            // int warp_min = __shfl_sync(0xffffffff, firstMutationDimIdx, 0);

            // 每个warp的第一个线程将结果写入shared memory
            if (lane_id == 0) {
                s_first_mutation[warp_id] = firstMutationDimIdx;
            }

            __syncthreads();
            // 只在第一个warp中进行warp间规约
            if (threadIdx.x == 0) {
                int min_val = s_first_mutation[0];
                for (int i = 1; i < warp_num; ++i) {
                    min_val = min(min_val, s_first_mutation[i]);
                }
                s_first_mutation[0] = min_val;

                // printf("num_warp:%d\n",warp_num);
            }
            
            __syncthreads();

            firstMutationDimIdx = s_first_mutation[0];

            // // parallel reduce
            // int tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 8);
            // firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            // tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 4);
            // firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            // tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 2);
            // firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            // tmp_idx = __shfl_down_sync(0x0000ffff, firstMutationDimIdx, 1);
            // firstMutationDimIdx = min(tmp_idx, firstMutationDimIdx);
            // // let all thread know the firstMutationDimIdx
            // int warp_min = __shfl_sync(0x0000ffff, firstMutationDimIdx, 0);

            // // 只在第一个线程中收集所有warp的最小值
            // if (threadIdx.x == 0) {
            //     const int num_warps = CUDA_PARAM_MAX_SIZE / 32;  // 计算总的warp数
            //     for(int i = 1; i < num_warps; ++i) {
            //         int other_min = __shfl_sync(0xffffffff, warp_min, 0, i * 32);
            //         warp_min = min(warp_min, other_min);
            //     }
            // }

            // firstMutationDimIdx = __shfl_sync(0xffffffff, warp_min, 0);

            if (threadIdx.x < dims){
                bool isInMutationWindow = true;
                if (firstMutationDimIdx < dims){
                    int step = threadIdx.x - mutationStartDim;
                    if(step < 0)    step += dims;
                    if(step > firstMutationDimIdx){
                        isInMutationWindow = false;
                    }
                }

                if(isInMutationWindow){
                    if(UsingEliteStrategy){
                        result_param = origin_param + f * (best_origin_param + mutant_param[0] - origin_param - mutant_param[1]);
                    }
                    else{
                        result_param = origin_param + 0.8f * (mutant_param[0] - mutant_param[2]);
                    }
                }
            }
        }

        if (threadIdx.x < dims){
            float lower_bound = evolve_data->lower_bound[threadIdx.x];
            float upper_bound = evolve_data->upper_bound[threadIdx.x];

            if(result_param < lower_bound || result_param > upper_bound){
                result_param = uniform_data[uniform_rnd_evolve_pos + (CUDA_PARAM_MAX_SIZE + 5) + threadIdx.x] * (upper_bound - lower_bound) + lower_bound;
            }
        }
        // printf("sol_idx:%d, thread:%d, result_param:%f\n", sol_idx, threadIdx.x, result_param);
        new_param->all_param[sol_idx * CUDA_PARAM_MAX_SIZE + threadIdx.x] = result_param;

        if (threadIdx.x == 0) {
            reinterpret_cast<float3 *>(new_param->lshade_param)[sol_idx] = float3{scale_f, scale_f1, crossover};
        }
    }

    __device__ __forceinline__ void BitonicWarpCompare(float &param, float &fitness, int lane_mask){
        float mapping_param = __shfl_xor_sync(0xffffffff, param, lane_mask);
        float mapping_fitness = __shfl_xor_sync(0xffffffff, fitness, lane_mask);
        // determine current sort order is increase (1.0) or decrease (-1.0)
        float sortOrder = (threadIdx.x > (threadIdx.x ^ lane_mask)) ? -1.0 : 1.0;

        if(sortOrder * (mapping_fitness - fitness) < 0.f){
            param = mapping_param;
            fitness = mapping_fitness;
        }
    }

    /**
     * Sort the first 64 of old param (0 - 63)
     */
    template <int T=CUDA_SOLVER_POP_SIZE>
    __device__ __forceinline__ void SortCurrentParamBasedBitonic(float *all_param, float *all_fitness){
        // each block have a share memory
        __shared__ float sm_sorted_fitness[T];
        __shared__ float sm_sorted_param[T];
        int param_id = blockIdx.x;
        int sol_id = threadIdx.x;
        float current_param = all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];
        float current_fitness = all_fitness[sol_id];

        int compare_idx;
        float mapping_param, mapping_fitness, sortOrder;

        if (threadIdx.x < T){
            // Sort the contents of 32 threads in a warp based on Bitonic merge sort. Implement detail is the alternative representation of https://en.wikipedia.org/wiki/Bitonic_sorter
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 3);
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 7);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 15);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finish the sorting 16 threads in Warp, continue to finish 2 group of 16 threads
            BitonicWarpCompare(current_param, current_fitness, 31);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finsh the sort for each warp, continue to finish the sort between different warp by share memory.
            // record the warp sorting result to share memory
            sm_sorted_param[sol_id] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
        }
        
        // Wait for all thread finish above computation
        __syncthreads();

        if(threadIdx.x < T){
            // if T == 64 (we have 2 warp), we just need to compare these 2 warp by share memory.
            // Otherwise, we need to modify the following code
            compare_idx = sol_id ^ 63;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];

            sortOrder = (threadIdx.x > (threadIdx.x ^ 63)) ? -1.0 : 1.0;

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
        }
        
        // Wait for the sort between two warp finish
        __syncthreads();

        if(threadIdx.x < CUDA_SOLVER_POP_SIZE){
            // Now, we can come back to the sorting in the warp
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finish all sorting for fitness and param
            if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
                all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = current_param;
                // printf("======================== Update sorted param for solution id:%d\n", threadIdx.x);
            }
            if (blockIdx.x == 0)    all_fitness[threadIdx.x] = current_fitness;
        } 
    }

    /**
     * Sort the delete part of old param (64 - 127)
     */
    template <int T=CUDA_SOLVER_POP_SIZE> // T is the pop size
    __global__ void SortDeleteParamBasedBitonic(float *all_param, float *all_fitness){
        if (all_param == nullptr || all_fitness == nullptr) return;
        // each block have a share memory
        __shared__ float sm_sorted_fitness[T * 2];
        __shared__ float sm_sorted_param[T * 2];
        int param_id = blockIdx.x;
        int sol_id = threadIdx.x;
        float current_param = all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id + T];
        float current_fitness = all_fitness[sol_id + T];

        int compare_idx;
        float mapping_param, mapping_fitness, sortOrder;

        if (threadIdx.x >= T){
            // Sort the contents of 32 threads in a warp based on Bitonic merge sort. Implement detail is the alternative representation of https://en.wikipedia.org/wiki/Bitonic_sorter
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 3);
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 7);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            BitonicWarpCompare(current_param, current_fitness, 15);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finish the sorting 16 threads in Warp, continue to finish 2 group of 16 threads
            BitonicWarpCompare(current_param, current_fitness, 31);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finsh the sort for each warp, continue to finish the sort between different warp by share memory.
            // record the warp sorting result to share memory
            sm_sorted_param[sol_id ] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
        }
        
        // Wait for all thread finish above computation
        __syncthreads();

        if (threadIdx.x >= T)
        {
            compare_idx = sol_id ^ 63;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];

            sortOrder = (threadIdx.x > (threadIdx.x ^ 63)) ? -1.0 : 1.0;

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
        }
        
        
        // Wait for the sort between two warp finish
        __syncthreads();
        if(threadIdx.x >= T){
            // Now, we can come back to the sorting in the warp
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            sm_sorted_param[sol_id ] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
        }
        
        __syncthreads();
        if(threadIdx.x >= T){
            compare_idx = threadIdx.x ^ 127;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;

            if (sortOrder * (mapping_fitness - current_fitness) < 0.f) {
                current_fitness = mapping_fitness;
                current_param = mapping_param;
                sm_sorted_fitness[threadIdx.x] = current_fitness;
                sm_sorted_param[threadIdx.x] = current_param;
            }
        }
        
        __syncthreads();
        if(threadIdx.x >= T){
            compare_idx = threadIdx.x ^ 32;
            mapping_fitness = sm_sorted_fitness[compare_idx];
            mapping_param = sm_sorted_param[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            if (sortOrder * (mapping_fitness - current_fitness) < 0.f) {
                current_fitness = mapping_fitness;
                current_param = mapping_param;
            }
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);

            // above all finish all sorting for fitness and param
            if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
                all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id + T] = current_param;
                // printf("======================== Update sorted param for solution id:%d\n", threadIdx.x);
            }
            if (blockIdx.x == 0)    all_fitness[threadIdx.x + T] = current_fitness;
        }
    }

    /**
     * Based on thrust sort the current parameter and delete part of old param (pop_size:64 0 - 256) or (pop_size: 256 - 512)
     */
    __global__ void RecordParamBasedSortIndices(float *params, int *indices, const float *origin_params){
        int sol_id = blockIdx.x;
        int param_id = threadIdx.x;
        
        int old_idx = indices[sol_id];
        
        params[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = origin_params[old_idx * CUDA_PARAM_MAX_SIZE + param_id];
    }
    
    /**
     * Sort the current parameter and delete part of old param (pop_size:64 0 - 256) or (pop_size: 256 - 512)
     */
    template<int T = CUDA_SOLVER_POP_SIZE>// T is the pop size
    __device__ __forceinline__ void SortOldParamBasedBitonic(float *all_param, float *all_fitness, int bias = 0){
        if (all_param == nullptr || all_fitness == nullptr) return;
        if (threadIdx.x >= T)   return;
        // each block have a share memory
        __shared__ float sm_sorted_fitness[T];
        __shared__ float sm_sorted_param[T];
        int param_id = blockIdx.x;
        int sol_id = threadIdx.x;
        float current_param;
        float current_fitness;

        current_param = all_param[(sol_id +bias) * CUDA_PARAM_MAX_SIZE + param_id];
        current_fitness = all_fitness[sol_id + bias];
        
        int compare_idx;
        float mapping_param, mapping_fitness, sortOrder;

        // Sort the contents of 32 threads in a warp based on Bitonic merge sort. Implement detail is the alternative representation of https://en.wikipedia.org/wiki/Bitonic_sorter
        BitonicWarpCompare(current_param, current_fitness, 1);

        BitonicWarpCompare(current_param, current_fitness, 3);
        BitonicWarpCompare(current_param, current_fitness, 1);

        BitonicWarpCompare(current_param, current_fitness, 7);
        BitonicWarpCompare(current_param, current_fitness, 2);
        BitonicWarpCompare(current_param, current_fitness, 1);

        BitonicWarpCompare(current_param, current_fitness, 15);
        BitonicWarpCompare(current_param, current_fitness, 4);
        BitonicWarpCompare(current_param, current_fitness, 2);
        BitonicWarpCompare(current_param, current_fitness, 1);

        // above all finish the sorting 16 threads in Warp, continue to finish 2 group of 16 threads
        BitonicWarpCompare(current_param, current_fitness, 31);
        BitonicWarpCompare(current_param, current_fitness, 8);
        BitonicWarpCompare(current_param, current_fitness, 4);
        BitonicWarpCompare(current_param, current_fitness, 2);
        BitonicWarpCompare(current_param, current_fitness, 1);

        // above all finsh the sort for each warp, continue to finish the sort between different warp by share memory.
        // record the warp sorting result to share memory
        sm_sorted_param[sol_id ] = current_param;
        sm_sorted_fitness[sol_id] = current_fitness;
        
        // Wait for all thread finish above computation
        __syncthreads();

        if (T >= 64){
            compare_idx = sol_id ^ 63;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > (threadIdx.x ^ 63)) ? -1.0 : 1.0;

            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
            
            // Wait for the sort between two warp finish
            __syncthreads();
            // Now, we can come back to the sorting in the warp
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);
        }
        if(T >= 128){
            // 1. 先存储当前值到共享内存
            sm_sorted_param[sol_id] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
            __syncthreads();

            // 2. 进行128元素的比较
            compare_idx = sol_id ^ 127;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            // 3. 更新结果到共享内存
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_fitness = mapping_fitness;
                current_param = mapping_param;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 4. 进行32元素比较
            compare_idx = sol_id ^ 32;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
            __syncthreads();

            // 5. 最后的warp内部清理
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);
        }
        if(T >= 256){
            // 1. 先存储当前值到共享内存
            sm_sorted_param[sol_id] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
            __syncthreads();

            // 2. 进行256元素的比较
            compare_idx = sol_id ^ 255;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            // 3. 更新结果到共享内存
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_fitness = mapping_fitness;
                current_param = mapping_param;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 5. 进行64元素比较
            compare_idx = sol_id ^ 64;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 6. 进行32元素比较
            compare_idx = sol_id ^ 32;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
            __syncthreads();

            // 7. 最后的warp内部清理
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);
        }
        if(T >= 512){
            // 1. 先存储当前值到共享内存
            sm_sorted_param[sol_id] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
            __syncthreads();

            // 2. 进行512元素的比较
            compare_idx = sol_id ^ 511;  // 511 = 111111111
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            // 3. 更新结果到共享内存
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_fitness = mapping_fitness;
                current_param = mapping_param;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 5. 进行128元素比较
            compare_idx = sol_id ^ 128;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 6. 进行64元素比较
            compare_idx = sol_id ^ 64;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();

            // 7. 进行32元素比较
            compare_idx = sol_id ^ 32;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();

            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
            }
            __syncthreads();

            // 8. 最后的warp内部清理
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);
        }
        if(T >= 1024){
            // 1. 先存储当前值到共享内存
            sm_sorted_param[sol_id] = current_param;
            sm_sorted_fitness[sol_id] = current_fitness;
            __syncthreads();
    
            // 2. 进行512元素的比较
            compare_idx = sol_id ^ 1023;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();
    
            // 3. 更新结果到共享内存
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_fitness = mapping_fitness;
                current_param = mapping_param;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();
    
            // 4. 进行256元素比较
            compare_idx = sol_id ^ 256;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();
    
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();
    
            // 5. 进行128元素比较
            compare_idx = sol_id ^ 128;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();
    
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();
    
            // 6. 进行64元素比较
            compare_idx = sol_id ^ 64;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();
    
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();
    
            // 7. 进行32元素比较
            compare_idx = sol_id ^ 32;
            mapping_param = sm_sorted_param[compare_idx];
            mapping_fitness = sm_sorted_fitness[compare_idx];
            sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
            __syncthreads();
    
            if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
                current_param = mapping_param;
                current_fitness = mapping_fitness;
                sm_sorted_param[sol_id] = current_param;
                sm_sorted_fitness[sol_id] = current_fitness;
            }
            __syncthreads();
    
            // 8. 最后的warp内部清理
            BitonicWarpCompare(current_param, current_fitness, 16);
            BitonicWarpCompare(current_param, current_fitness, 8);
            BitonicWarpCompare(current_param, current_fitness, 4);
            BitonicWarpCompare(current_param, current_fitness, 2);
            BitonicWarpCompare(current_param, current_fitness, 1);
        }
        
        if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
            all_param[(sol_id + bias) * CUDA_PARAM_MAX_SIZE + param_id] = current_param;
        }
        if (blockIdx.x == 0)    all_fitness[threadIdx.x + bias] = current_fitness;
    }
    
    /**
     * I feel something wrong for the following implement of SortOldParamBasedBitonic
     */
    // template <int T=64> // T is the pop size
    // __device__ __forceinline__ void SortOldParamBasedBitonic(float *all_param, float *all_fitness){
    //     if (all_param == nullptr || all_fitness == nullptr) return;
    //     // each block have a share memory
    //     __shared__ float sm_sorted_fitness[T * 2];
    //     __shared__ float sm_sorted_param[T * 2];
    //     int param_id = blockIdx.x;
    //     int sol_id = threadIdx.x;
    //     float current_param;
    //     float current_fitness;

    //     if(threadIdx.x < T){
    //         current_param = all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];
    //         current_fitness = all_fitness[sol_id];
    //     }
    //     else{
    //         current_param = all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id + T];
    //         current_fitness = all_fitness[sol_id + T];
    //     }
        

    //     int compare_idx;
    //     float mapping_param, mapping_fitness, sortOrder;

    //     // if (threadIdx.x >= 64){
    //     // Sort the contents of 32 threads in a warp based on Bitonic merge sort. Implement detail is the alternative representation of https://en.wikipedia.org/wiki/Bitonic_sorter
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     BitonicWarpCompare(current_param, current_fitness, 3);
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     BitonicWarpCompare(current_param, current_fitness, 7);
    //     BitonicWarpCompare(current_param, current_fitness, 2);
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     BitonicWarpCompare(current_param, current_fitness, 15);
    //     BitonicWarpCompare(current_param, current_fitness, 4);
    //     BitonicWarpCompare(current_param, current_fitness, 2);
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     // above all finish the sorting 16 threads in Warp, continue to finish 2 group of 16 threads
    //     BitonicWarpCompare(current_param, current_fitness, 31);
    //     BitonicWarpCompare(current_param, current_fitness, 8);
    //     BitonicWarpCompare(current_param, current_fitness, 4);
    //     BitonicWarpCompare(current_param, current_fitness, 2);
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     // above all finsh the sort for each warp, continue to finish the sort between different warp by share memory.
    //     // record the warp sorting result to share memory
    //     sm_sorted_param[sol_id ] = current_param;
    //     sm_sorted_fitness[sol_id] = current_fitness;
    //     // }
        
    //     // Wait for all thread finish above computation
    //     __syncthreads();

    //     // if (threadIdx.x >= 64)
    //     // {
    //     compare_idx = sol_id ^ 63;
    //     mapping_param = sm_sorted_param[compare_idx];
    //     mapping_fitness = sm_sorted_fitness[compare_idx];

    //     sortOrder = (threadIdx.x > (threadIdx.x ^ 63)) ? -1.0 : 1.0;

    //     if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
    //         current_param = mapping_param;
    //         current_fitness = mapping_fitness;
    //     }        
    //     // Wait for the sort between two warp finish
    //     __syncthreads();
    //     // Now, we can come back to the sorting in the warp
    //     BitonicWarpCompare(current_param, current_fitness, 16);
    //     BitonicWarpCompare(current_param, current_fitness, 8);
    //     BitonicWarpCompare(current_param, current_fitness, 4);
    //     BitonicWarpCompare(current_param, current_fitness, 2);
    //     BitonicWarpCompare(current_param, current_fitness, 1);

    //     sm_sorted_param[sol_id ] = current_param;
    //     sm_sorted_fitness[sol_id] = current_fitness;
        
    //     __syncthreads();
    //     if(threadIdx.x >= T){
    //         compare_idx = threadIdx.x ^ 127;
    //         mapping_param = sm_sorted_param[compare_idx];
    //         mapping_fitness = sm_sorted_fitness[compare_idx];
    //         sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;

    //         if (sortOrder * (mapping_fitness - current_fitness) < 0.f) {
    //             current_fitness = mapping_fitness;
    //             current_param = mapping_param;
    //             sm_sorted_fitness[threadIdx.x] = current_fitness;
    //             sm_sorted_param[threadIdx.x] = current_param;
    //         }
    //     }
        
    //     __syncthreads();
    //     if(threadIdx.x >= T){
    //         compare_idx = threadIdx.x ^ 32;
    //         mapping_fitness = sm_sorted_fitness[compare_idx];
    //         mapping_param = sm_sorted_param[compare_idx];
    //         sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
    //         if (sortOrder * (mapping_fitness - current_fitness) < 0.f) {
    //             current_fitness = mapping_fitness;
    //             current_param = mapping_param;
    //         }
    //         BitonicWarpCompare(current_param, current_fitness, 16);
    //         BitonicWarpCompare(current_param, current_fitness, 8);
    //         BitonicWarpCompare(current_param, current_fitness, 4);
    //         BitonicWarpCompare(current_param, current_fitness, 2);
    //         BitonicWarpCompare(current_param, current_fitness, 1);
    //     }

    //     if(threadIdx.x >= T){
    //         // above all finish all sorting for fitness and param
    //         if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
    //             all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id + T] = current_param;
    //             // printf("======================== Update sorted param for solution id:%d\n", threadIdx.x);
    //         }
    //         if (blockIdx.x == 0)    all_fitness[threadIdx.x + T] = current_fitness;
    //     }
    //     else{
    //         // above all finish all sorting for fitness and param
    //         if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
    //             all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = current_param;
    //             // printf("======================== Update sorted param for solution id:%d\n", threadIdx.x);
    //         }
    //         if (blockIdx.x == 0)    all_fitness[threadIdx.x] = current_fitness;
    //     }
    // }

    // __device__ __forceinline__ void EvolveTerminate(CudaEvolveData *evolve, float *last_best_fitness, float *all_fitness, int *terminate_flag){
    //     if(threadIdx.x > 0)    return;
    //     if(*terminate_flag != 0)    return;
    //     if(blockIdx.x > evolve->problem_param.elite_eval_count)   return;
        
    //     float elite_eval_sum = all_fitness[blockIdx.x];

    //     if (evolve->problem_param.elite_eval_count == 8){
    //         elite_eval_sum += __shfl_down_sync(0x000000ff, elite_eval_sum, 8);
    //         elite_eval_sum += __shfl_down_sync(0x000000ff, elite_eval_sum, 4);
    //         elite_eval_sum += __shfl_down_sync(0x000000ff, elite_eval_sum, 2);
    //         elite_eval_sum += __shfl_down_sync(0x000000ff, elite_eval_sum, 1);
    //     }
    //     printf("elite_eval_sum:%f\n",elite_eval_sum);
    //     if(abs((elite_eval_sum - evolve->problem_param.elite_eval_count * *last_best_fitness)/evolve->problem_param.elite_eval_count) < evolve->problem_param.accuracy_rng){
    //         *terminate_flag = 1;
    //     }
    // }

    __device__ __forceinline__ void EvolveTerminate(CudaEvolveData *evolve, float *last_best_fitness, float *all_fitness, int *terminate_flag){
        if(threadIdx.x > 0)    return;
        if(blockIdx.x > 0)  return;
        if(*terminate_flag != 0)    return;
                
        // 第0个block负责收集和累加所有block的结果
        if(blockIdx.x == 0) {
            float sum = 0.0;
            // 累加其他block的值
            for(int i = 1; i < evolve->problem_param.elite_eval_count; ++i) {
                sum += all_fitness[i];
                // printf("individual %d fitness:%f\n",i, sum );
            }
            
            // printf("elite_eval_sum:%f bias:%f\n", sum, abs((sum - evolve->problem_param.elite_eval_count * *last_best_fitness)/evolve->problem_param.elite_eval_count));
            if(abs((sum - evolve->problem_param.elite_eval_count * *last_best_fitness)/evolve->problem_param.elite_eval_count) < evolve->problem_param.accuracy_rng){
                *terminate_flag = 1;
            }
        }
    }

    template <int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateParameter2(int epoch, CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, 
                                CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, int* terminate_flag=nullptr, float *last_f=nullptr) {
        
        const int sol_id = threadIdx.x;
        const int param_id = blockIdx.x;
        
        // Each thread handles one solution's parameters
        if (sol_id < T) {
            float old_fitness = old_param->fitness[sol_id];
            float new_fitness = new_param->fitness[sol_id];

            // Update parameter if within valid parameter range
            if (param_id < CUDA_PARAM_MAX_SIZE) {
                float old_param_value = old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];
                float new_param_value = new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];

                // Compare fitness and update parameters accordingly
                if (new_fitness < old_fitness) {
                    // Select better solution as current sol, move previous to replaced part
                    old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = new_param_value;
                    old_param->fitness[sol_id] = new_fitness;
                    old_param->all_param[(sol_id + 2 * T) * CUDA_PARAM_MAX_SIZE + param_id] = old_param_value;
                    old_param->fitness[sol_id + 2 * T] = old_fitness;
                } else {
                    // Keep old solution, move new solution to replaced part
                    old_param->all_param[(sol_id + 2 * T) * CUDA_PARAM_MAX_SIZE + param_id] = new_param_value;
                    old_param->fitness[sol_id + 2 * T] = new_fitness;
                }
            }

            // Synchronize before L-SHADE parameter updates
            __syncthreads();

            /**
             * L-SHADE hyperparameter update section
             */
            if (blockIdx.x == 0) {
                // Calculate 8 float data for a warp
                float ALIGN(64) adaptiveParamSums[8];
                const int num_warps = T / 32;
                __shared__ ALIGN(64) float share_sum[num_warps * 8];

                // Load and calculate L-SHADE parameters
                float3 lshade_param = reinterpret_cast<float3 *>(new_param->lshade_param)[sol_id];
                float scale_f = lshade_param.x;
                float scale_f1 = lshade_param.y;
                float cr = lshade_param.z;
                
                // Calculate weight based on fitness improvement
                float w = (new_fitness - old_fitness) / max(1e-4f, new_fitness);

                // Calculate weighted parameters for L-SHADE update equations
                adaptiveParamSums[0] = w;
                adaptiveParamSums[1] = w * scale_f;
                adaptiveParamSums[2] = w * scale_f * scale_f;
                adaptiveParamSums[3] = w * scale_f1;
                adaptiveParamSums[4] = w * scale_f1 * scale_f1;
                adaptiveParamSums[5] = w * cr;
                adaptiveParamSums[6] = w * cr * cr;
                adaptiveParamSums[7] = 0;

                // Warp-level parallel reduction
                for (int i = 0; i < 7; ++i) {
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 16);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 8);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 4);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 2);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 1);
                }

                // Store warp results in shared memory
                if ((threadIdx.x & 31) == 0) {
                    reinterpret_cast<float4 *>(share_sum)[(threadIdx.x >> 5) * 2] = reinterpret_cast<float4 *>(adaptiveParamSums)[0];
                    reinterpret_cast<float4 *>(share_sum)[(threadIdx.x >> 5) * 2 + 1] = reinterpret_cast<float4 *>(adaptiveParamSums)[1];
                }
                
                __syncthreads();

                // Final reduction across warps
                if (threadIdx.x < (T >> 5)) {
                    // Load parameters from shared memory
                    reinterpret_cast<float4 *>(adaptiveParamSums)[0] = reinterpret_cast<float4 *>(share_sum)[threadIdx.x * 2];
                    reinterpret_cast<float4 *>(adaptiveParamSums)[1] = reinterpret_cast<float4 *>(share_sum)[threadIdx.x * 2 + 1];
                    
                    // Parallel reduction based on population size
                    for(int i = 0; i < 7; ++i) {
                        if (T == 1024){
                            adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 16);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000ffff, adaptiveParamSums[i], 8);
                            adaptiveParamSums[i] += __shfl_down_sync(0x000000ff, adaptiveParamSums[i], 4);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        if (T == 512) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000ffff, adaptiveParamSums[i], 8);
                            adaptiveParamSums[i] += __shfl_down_sync(0x000000ff, adaptiveParamSums[i], 4);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        else if (T == 256) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x000000ff, adaptiveParamSums[i], 4);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        else if (T == 128) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        else if (T == 64) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                    }

                    // Update evolve parameters
                    if (threadIdx.x == 0) {
                        if (adaptiveParamSums[2] > 1e-4f && adaptiveParamSums[4] > 1e-4f && adaptiveParamSums[6] > 1e-4f) {
                            evolve->hist_lshade_param.scale_f = adaptiveParamSums[2] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[1] * adaptiveParamSums[0]);
                            evolve->hist_lshade_param.scale_f1 = adaptiveParamSums[4] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[3] * adaptiveParamSums[0]);
                            evolve->hist_lshade_param.Cr = adaptiveParamSums[6] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[5] * adaptiveParamSums[0]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        // If using thrust, don't need to use Bitonic
        SortOldParamBasedBitonic(old_param->all_param, old_param->fitness);
        // __syncthreads();
        SortOldParamBasedBitonic(old_param->all_param, old_param->fitness, CUDA_SOLVER_POP_SIZE);

    }

    template <int T = CUDA_SOLVER_POP_SIZE>
    __global__ void UpdateParameter(int epoch, CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, int* terminate_flag=nullptr, float *last_f=nullptr){
        // if ((*terminate_flag & 1) > 0) return;
        // for old_param (current sol, delete sol, replaced sol), we select the fitness of current sol for all old param
        // so threadIdx.x & (T-1) equal to threadIdx.x % (T-1) which can help us to mapping all old_param to current sol
        float old_fitness = old_param->fitness[threadIdx.x & (T-1)], new_fitness = new_param->fitness[threadIdx.x & (T-1)];
        // printf("epoch:%d, old_fitness:%f, new_fitness:%f\n",epoch, old_fitness, new_fitness);
        // printf("updating parameter????????????????\n");
        int sol_id = threadIdx.x;
        int param_id = blockIdx.x;
        // float current_fitness = CUDA_MAX_FLOAT;

        // Update parameter
        if (sol_id < T){
            if (param_id < CUDA_PARAM_MAX_SIZE){
                float old_param_value = old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];
                float new_param_value = new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id];

                // compare old_fitness and new_fitness to determine which solution should be replaced.
                if (new_fitness < old_fitness){
                    // printf("epoch:%d, old_fitness:%f, new_fitness:%f\n",epoch, old_fitness, new_fitness);
                    // current_fitness = new_fitness;
                    // select better solution as current sol, and move previous solution to replaced part
                    old_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = new_param_value;
                    old_param->fitness[sol_id] = new_fitness;
                    old_param->all_param[(sol_id + 2 * T) * CUDA_PARAM_MAX_SIZE + param_id] = old_param_value;
                    // old_param->fitness[sol_id + 2*T] = old_fitness;
                }
                else{
                    // current_fitness = old_fitness;
                    old_param->all_param[(sol_id + 2 * T) * CUDA_PARAM_MAX_SIZE + param_id] = new_param_value;
                }
            }
        }
        else{
            old_param->fitness[sol_id ] = (new_fitness < old_fitness) ? old_fitness : new_fitness;
        }

        // wait for all thread finish above all computation
        __syncthreads();

        /**
         * Based on the rule of L shade to update hyperparameter
         */
        if (blockIdx.x == 0){
            // calculate 8 float data for a warp.
            float ALIGN(64) adaptiveParamSums[8];
            // each warp will runing parallel reduction for sum. for 64 pop_size cluster, we need 2 warp. And then, all result are storaged in a share memory array.
            const int num_warps = T / 32;  // 计算总的warp数
            __shared__ ALIGN(64) float share_sum[num_warps * 8];

            float3 lshade_param;
            float scale_f, scale_f1, cr, w;
            if (threadIdx.x < T){
                // float3 is a built-in vector type of CUDA. And their memory addresses are continuous.
                // It is more efficient to read parameters by this conversion method.
                lshade_param = reinterpret_cast<float3 *>(new_param->lshade_param)[sol_id];
                scale_f = lshade_param.x;
                scale_f1 = lshade_param.y;
                cr = lshade_param.z;
                // formula (8) and (9) in https://ieeexplore.ieee.org/document/6900380
                w  = (new_fitness - old_fitness) / max(1e-4f, new_fitness);

                // calculate w*cr, w*cr*cr, w*scale_f, w*scale_f*scale_f for equation (7) in https://ieeexplore.ieee.org/document/6900380
                adaptiveParamSums[0] = w;
                adaptiveParamSums[1] = w * scale_f;
                adaptiveParamSums[2] = w * scale_f * scale_f;
                adaptiveParamSums[3] = w * scale_f1;
                adaptiveParamSums[4] = w * scale_f1 * scale_f1;
                adaptiveParamSums[5] = w * cr;
                adaptiveParamSums[6] = w * cr * cr;
                adaptiveParamSums[7] = 0;

                // Warp parallel reduction sum (finish the sum part of equal (7) (8) (9))
                for (int i = 0; i < 7; ++i) {
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 16);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 8);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 4);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 2);
                    adaptiveParamSums[i] += __shfl_down_sync(0xffffffff, adaptiveParamSums[i], 1);
                }
                // The recursive results for each warp are recorded at the shared memoery
                if ((threadIdx.x & 31) == 0){
                    reinterpret_cast<float4 *>(share_sum)[(threadIdx.x >> 5) * 2] = reinterpret_cast<float4 *>(adaptiveParamSums)[0];
                    reinterpret_cast<float4 *>(share_sum)[(threadIdx.x >> 5) * 2 + 1] = reinterpret_cast<float4 *>(adaptiveParamSums)[1];
                }
            }
            __syncthreads();
            // continue to use parallel reduction for different results of above data that have been storaged at share memory
            if (threadIdx.x < T){
                if (threadIdx.x < (T >> 5)){
                    // loading parameter from share memory to adaptiveParamSums so that each thread responsible for the from different warp
                    reinterpret_cast<float4 *>(adaptiveParamSums)[0] = reinterpret_cast<float4 *>(share_sum)[threadIdx.x * 2];
                    reinterpret_cast<float4 *>(adaptiveParamSums)[1] = reinterpret_cast<float4 *>(share_sum)[threadIdx.x * 2 + 1];
                    
                    // parallel reduction for different warp result in share memory
                    for(int i = 0; i < 7; ++i){
                        // !!!!!!!!!!!!!!! If the T or pop_size is not 64. This part should be modified. !!!!!!!!!!!!!!!!!!
                        // if (T == 128) {
                        //     adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                        // }
                        // adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);

                        if (T == 512) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000ffff, adaptiveParamSums[i], 8);
                            adaptiveParamSums[i] += __shfl_down_sync(0x000000ff, adaptiveParamSums[i], 4);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        // For T=256: need 8->4->2->1 reduction pattern
                        else if (T == 256) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x000000ff, adaptiveParamSums[i], 4);
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                            // adaptiveParamSums[i] += __shfl_down_sync(0x00000001, adaptiveParamSums[i], 1);
                        }
                        // For T=128: need 4->2->1 reduction pattern
                        else if (T == 128) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x0000000f, adaptiveParamSums[i], 2);
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                        // For T=64: need 2->1 reduction pattern
                        else if (T == 64) {
                            adaptiveParamSums[i] += __shfl_down_sync(0x00000003, adaptiveParamSums[i], 1);
                        }
                    }

                    // update the evolve data
                    if(threadIdx.x == 0){
                        if (adaptiveParamSums[2] > 1e-4f && adaptiveParamSums[4] > 1e-4f && adaptiveParamSums[6] > 1e-4f){
                            evolve->hist_lshade_param.scale_f = adaptiveParamSums[2] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[1] * adaptiveParamSums[0]);
                            evolve->hist_lshade_param.scale_f1 = adaptiveParamSums[4] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[3] * adaptiveParamSums[0]);
                            evolve->hist_lshade_param.Cr = adaptiveParamSums[6] * adaptiveParamSums[0] / max(1e-4f, adaptiveParamSums[5] * adaptiveParamSums[0]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        // If using thrust, don't need to use Bitonic
        // SortOldParamBasedBitonic(old_param->all_param, old_param->fitness);
        // __syncthreads();
        // SortOldParamBasedBitonic(old_param->all_param, old_param->fitness, CUDA_SOLVER_POP_SIZE);

        

        // The following this method is only compare current fitness and last fitness. When the bias of these two fitness lower than accuracy_rng then stop evaluation
        // __syncthreads();
        // if(threadIdx.x == 0 && blockIdx.x == 0 && *terminate_flag == 0){
        //     // printf("evolve->problem_param.accuracy_rng:%f\n",evolve->problem_param.accuracy_rng);
        //     // if(abs(old_param->fitness[0] - *last_f) < evolve->problem_param.accuracy_rng){
        //     //     *terminate_flag = 1;
        //     //     // printf("stop stop stop");
        //     // }
        //     if(abs(old_param->fitness[0]) < 10){
        //         *terminate_flag = 1;
        //         // printf("stop stop stop");
        //     }
        //     *last_f = old_param->fitness[threadIdx.x];
        // }

        // This method consider the average fitness of top 8 individual. when the avg lower than accuracy_rng then stop evaluation
        // EvolveTerminate(evolve, last_f, old_param->fitness, terminate_flag);
        // *last_f = old_param->fitness[threadIdx.x];
    }
}


#endif