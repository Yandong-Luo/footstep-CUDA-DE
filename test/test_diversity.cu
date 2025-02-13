#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_PARAM_MAX_SIZE 96
#define CUDA_SOLVER_POP_SIZE 1024

// CPU版本的计算函数
float calculate_diversity_cpu(const float* params, int pop_size, int param_size) {
    float total_diversity = 0.0f;
    
    // 对每个参数维度
    for (int param_id = 0; param_id < param_size; param_id++) {
        float param_diff_sum = 0.0f;
        int valid_count = 0;
        
        // 计算每个个体与最优个体的差异
        for (int sol_id = 1; sol_id < pop_size - 1; sol_id++) {
            float diff = fabs(params[param_id] - params[sol_id * param_size + param_id]);
            param_diff_sum += diff;
            valid_count++;
        }
        
        // 计算这个参数维度的平均差异
        total_diversity += (param_diff_sum / valid_count);
    }
    
    // 返回所有参数维度的平均差异
    return total_diversity / param_size;
}

// CUDA kernel
__global__ void calculate_cluster_diversity(const float* params, float* diversity) {
    int sol_id = threadIdx.x;
    int param_id = blockIdx.x;

    const int warp_size = 32;
    const int warp_num = (CUDA_SOLVER_POP_SIZE >> 5);
    const int block_warp_num = (CUDA_PARAM_MAX_SIZE >> 5);
    // printf("%d\n",block_warp_num);
    int warp_id = (threadIdx.x >> 5);
    int lane_id = (threadIdx.x & 31);

    __shared__ float sm_diff_sum[warp_num];
    __shared__ float sm_param_diff[CUDA_PARAM_MAX_SIZE];
    __shared__ float sm_block_sum[block_warp_num];

    float diff = 0.0f;
    if(sol_id > 0 && sol_id < CUDA_SOLVER_POP_SIZE - 1) {
        diff = fabsf(params[param_id] - params[sol_id * CUDA_PARAM_MAX_SIZE + param_id]);
    }
    
    float diff_sum = diff;
    diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 16);
    diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 8);
    diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 4);
    diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 2);
    diff_sum += __shfl_down_sync(0xffffffff, diff_sum, 1);

    if(lane_id == 0) {
        sm_diff_sum[warp_id] = (diff_sum / static_cast<float>(warp_size));
        // printf("sm_diff_sum:%f\n", sm_diff_sum[warp_id]);
    }
    __syncthreads();
    
    if(threadIdx.x < warp_num){
        float current_param_diff = sm_diff_sum[threadIdx.x];
        if (warp_num == 32){                // also means POP_SIZE = 1024
            current_param_diff += __shfl_down_sync(0xffffffff, current_param_diff, 16);
            current_param_diff += __shfl_down_sync(0x0000ffff, current_param_diff, 8);
            current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
            current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
            current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
        }
        else if (warp_num == 16) {          // also means POP_SIZE = 512
            current_param_diff += __shfl_down_sync(0x0000ffff, current_param_diff, 8);
            current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
            current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
            current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
        }
        else if (warp_num == 8) {            // also means POP_SIZE = 256
            current_param_diff += __shfl_down_sync(0x000000ff, current_param_diff, 4);
            current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
            current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
        }
        else if (warp_num == 4) {            // also means POP_SIZE = 128
            current_param_diff += __shfl_down_sync(0x0000000f, current_param_diff, 2);
            current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
        }
        else if (warp_num == 2) {           // also means POP_SIZE = 64    
            current_param_diff += __shfl_down_sync(0x00000003, current_param_diff, 1);
        }
        if(threadIdx.x == 0){
            sm_param_diff[param_id] = (current_param_diff / static_cast<float>(warp_num));
            // printf("sm_param_diff:%f\n", sm_param_diff[param_id]);
        } 
    }
    __syncthreads();

    // 在每个block中，计算完sm_param_diff[blockIdx.x]后
    if(threadIdx.x == 0) {  // 只需要一个线程
        // 每个block直接添加自己的贡献到最终结果
        atomicAdd(diversity, sm_param_diff[blockIdx.x] / static_cast<float>(CUDA_PARAM_MAX_SIZE));
    }
    
    // block parallel reduction sum 
    // if(threadIdx.x < CUDA_PARAM_MAX_SIZE && (blockIdx.x & 31) == 0){
    //     float block_warp_sum = sm_param_diff[threadIdx.x];
    //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 16);
    //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 8);
    //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 4);
    //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 2);
    //     block_warp_sum += __shfl_down_sync(0xffffffff, block_warp_sum, 1);

    //     if(lane_id == 0){
    //         sm_block_sum[warp_id] = (block_warp_sum / static_cast<float>(block_warp_num));
    //         printf("sm_block_sum:%f\n", sm_block_sum[warp_id]);
    //     } 
    // }
    // __syncthreads();

    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     diversity[0] = 0.0f;
    //     for(int i = 0; i < block_warp_num; ++i){
    //         diversity[0] += sm_block_sum[i];
    //         printf("sm_block:%f\n", sm_block_sum[i]);
    //     }
    //     // atomicAdd(&diversity[0], sm_block_sum[warp_id]);
    // }
}

int main() {
    // 分配和初始化数据
    const int total_size = CUDA_SOLVER_POP_SIZE * CUDA_PARAM_MAX_SIZE;
    float* h_params = new float[total_size];
    
    // 用随机数填充参数数组
    for (int i = 0; i < total_size; i++) {
        h_params[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配GPU内存
    float *d_params, *d_diversity;
    cudaMalloc(&d_params, total_size * sizeof(float));
    cudaMalloc(&d_diversity, sizeof(float));
    cudaMemset(d_diversity, 0, sizeof(float));

    // 复制数据到GPU
    cudaMemcpy(d_params, h_params, total_size * sizeof(float), cudaMemcpyHostToDevice);

    // 在GPU上计算
    calculate_cluster_diversity<<<CUDA_PARAM_MAX_SIZE, CUDA_SOLVER_POP_SIZE>>>(d_params, d_diversity);

    // 获取GPU结果
    float gpu_result;
    cudaMemcpy(&gpu_result, d_diversity, sizeof(float), cudaMemcpyDeviceToHost);

    // 在CPU上计算
    float cpu_result = calculate_diversity_cpu(h_params, CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE);

    // 比较结果
    printf("GPU Result: %f\n", gpu_result);
    printf("CPU Result: %f\n", cpu_result);
    printf("Difference: %f\n", fabs(gpu_result - cpu_result));

    // 清理内存
    delete[] h_params;
    cudaFree(d_params);
    cudaFree(d_diversity);

    return 0;
}