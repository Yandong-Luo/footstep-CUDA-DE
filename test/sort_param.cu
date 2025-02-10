#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>

#define CUDA_PARAM_MAX_SIZE 64
#define T 256  // template parameter for SortParamBasedBitonic

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Helper function to print arrays
// void printArrays(float* fitness, float* params, int size) {
//     printf("\nFitness values:\n");
//     for(int i = 0; i < size; i++) {
//         printf("%.2f ", fitness[i]);
//     }
//     // printf("\n\nParam values (first parameter only):\n");
//     // for(int i = 0; i < size; i++) {
//     //     printf("%.2f ", params[i * CUDA_PARAM_MAX_SIZE]);
//     // }
//     printf("\n\n");
// }

void printArrays(float* fitness, float* params, int size) {
    printf("\nFitness values and corresponding parameters:\n");
    printf("Index\tFitness\tParameters (all)\n");
    printf("----------------------------------------\n");
    for(int i = 0; i < size; i++) {
        printf("%d\t%.2f\t", i, fitness[i]);
        // 打印该索引位置的所有参数
        for(int j = 0; j < CUDA_PARAM_MAX_SIZE; j++) {
            printf("%.2f ", params[i * CUDA_PARAM_MAX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void verifySort(const std::unordered_map<float, std::vector<float>> fitnessToParams, float* fitness, float* params, int size) {
    printf("\nVerifying sorting results...\n");

    bool mappingCorrect = true;
    printf("\nChecking fitness-param mapping consistency...\n");

    for (int i = 0; i < size; i++) {
        float key = fitness[i];

        // 使用 find() 避免 `key` 不存在时 `operator[]` 访问出错
        auto it = fitnessToParams.find(key);
        if (it == fitnessToParams.end()) {
            mappingCorrect = false;
            printf("Error: Fitness value %.2f is missing in the original mapping!\n", key);
            continue;
        }

        const std::vector<float>& originalParams = it->second;
        bool paramMismatch = false;

        for (int j = 0; j < CUDA_PARAM_MAX_SIZE; j++) {
            float sortedParam = params[i * CUDA_PARAM_MAX_SIZE + j];
            if (std::fabs(originalParams[j] - sortedParam) > 0.001f) { // 允许小数点误差
                paramMismatch = true;
                mappingCorrect = false;
            }
        }

        // 仅在发生不匹配时打印详细信息
        if (paramMismatch) {
            printf("Mismatch at index %d for fitness %.2f:\n", i, key);
            printf("  Original Params: ");
            for (float p : originalParams) {
                printf("%.2f ", p);
            }
            printf("\n  Sorted Params  : ");
            for (int j = 0; j < CUDA_PARAM_MAX_SIZE; j++) {
                printf("%.2f ", params[i * CUDA_PARAM_MAX_SIZE + j]);
            }
            printf("\n");
        }
    }

    if (mappingCorrect) {
        printf("\nMapping verification PASSED!\n");
    } else {
        printf("\nMapping verification FAILED! Check above mismatches.\n");
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

__global__ void SortParamBasedBitonic3(float *all_param, float *all_fitness, int bias = 0){
    if (all_param == nullptr || all_fitness == nullptr) return;
    // if (threadIdx.x >= T)   return;
    // each block have a share memory
    __shared__ float sm_sorted_fitness[2*T];
    __shared__ float sm_sorted_param[2*T];
    int param_id = blockIdx.x;
    int sol_id = threadIdx.x;
    float current_param;
    float current_fitness;

    current_param = all_param[(sol_id ) * CUDA_PARAM_MAX_SIZE + param_id];
    current_fitness = all_fitness[(sol_id)];

    if (blockIdx.x == 13) {
        printf("Block 13: sol_id=%d, param_value=%.2f\n", sol_id, current_param);
    }
     

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
        if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
            current_param = mapping_param;
            current_fitness = mapping_fitness;
        }

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

        // // 4. 进行128元素比较
        // compare_idx = sol_id ^ 64;
        // mapping_param = sm_sorted_param[compare_idx];
        // mapping_fitness = sm_sorted_fitness[compare_idx];
        // sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
        // if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
        //     current_param = mapping_param;
        //     current_fitness = mapping_fitness;
        //     sm_sorted_param[sol_id] = current_param;
        //     sm_sorted_fitness[sol_id] = current_fitness;
        // }
        // __syncthreads();

        // 5. 进行64元素比较
        compare_idx = sol_id ^ 64;
        mapping_param = sm_sorted_param[compare_idx];
        mapping_fitness = sm_sorted_fitness[compare_idx];
        sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
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
        if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
            current_param = mapping_param;
            current_fitness = mapping_fitness;
        }

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

        // // 4. 进行256元素比较
        // compare_idx = sol_id ^ 256;
        // mapping_param = sm_sorted_param[compare_idx];
        // mapping_fitness = sm_sorted_fitness[compare_idx];
        // sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
        // if(sortOrder * (mapping_fitness - current_fitness) < 0.f){
        //     current_param = mapping_param;
        //     current_fitness = mapping_fitness;
        //     sm_sorted_param[sol_id] = current_param;
        //     sm_sorted_fitness[sol_id] = current_fitness;
        // }
        // __syncthreads();

        // 5. 进行128元素比较
        compare_idx = sol_id ^ 128;
        mapping_param = sm_sorted_param[compare_idx];
        mapping_fitness = sm_sorted_fitness[compare_idx];
        sortOrder = (threadIdx.x > compare_idx) ? -1.f : 1.f;
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
    __syncthreads();
    if (blockIdx.x < CUDA_PARAM_MAX_SIZE){
        all_param[(sol_id )* CUDA_PARAM_MAX_SIZE + param_id] = current_param;
    }
    if (blockIdx.x == 0)    all_fitness[(threadIdx.x)] = current_fitness;
}

int main() {
    // Host arrays
    float *h_fitness, *h_params;
    // Device arrays
    float *d_fitness, *d_params;

    std::unordered_map<float, std::vector<float>> fitnessToParams;

    CHECK_CUDA(cudaSetDevice(0));
    
    // Allocate host memory
    cudaHostAlloc(&h_fitness, 2*T * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_params, 2*T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaHostAllocDefault);
    // Initialize random seed
    srand(time(NULL));
    printf("T:%d\n", T);
    // Initialize fitness with decreasing values
    for(int i = 0; i < 2*T; i++) {
        // h_fitness[i] = (float)(2*T - i);  // Creates values from T down to 1
        std::vector<float> paramValues;
        for(int j = 0; j < CUDA_PARAM_MAX_SIZE; ++j){
            h_params[i * CUDA_PARAM_MAX_SIZE + j] = (float)rand() / RAND_MAX * 100.0f;  // Random values between 0 and 100
            paramValues.push_back(h_params[i * CUDA_PARAM_MAX_SIZE + j]);
        }
        h_fitness[i] = (float)rand() / RAND_MAX * 100.0f;
        fitnessToParams[h_fitness[i]] = paramValues;
    }
    
    // // Initialize params with random values
    // for(int i = 0; i < 2*T * CUDA_PARAM_MAX_SIZE; i++) {
    //     h_params[i] = (float)rand() / RAND_MAX * 100.0f;  // Random values between 0 and 100
    // }
    
    printf("Initial arrays:");
    printArrays(h_fitness, h_params, 2*T);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_fitness, 2*T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_params, 2*T * CUDA_PARAM_MAX_SIZE * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_fitness, h_fitness, 2*T * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_params, h_params, 2*T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    // We need CUDA_PARAM_MAX_SIZE blocks because we're sorting each parameter independently
    // SortParamBasedBitonic<<<CUDA_PARAM_MAX_SIZE, T>>>(d_params, d_fitness);
    
    // SortParamBasedBitonic2<<<CUDA_PARAM_MAX_SIZE, 2*T>>>(d_params, d_fitness);

    SortParamBasedBitonic3<<<CUDA_PARAM_MAX_SIZE, 2*T>>>(d_params, d_fitness, 0);
    // SortParamBasedBitonic3<<<CUDA_PARAM_MAX_SIZE, T>>>(d_params, d_fitness, T);
    
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_fitness, d_fitness, 2*T * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_params, d_params, 2*T * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Sorted arrays:");
    printArrays(h_fitness, h_params, 2*T);

    verifySort(fitnessToParams, h_fitness, h_params, 2*T);
    
    // // Verify sorting
    // bool sorted = true;
    // for(int i = 1; i < 2*T; i++) {
    //     if(h_fitness[i-1] < h_fitness[i]) {
    //         sorted = false;
    //         printf("Error: Array not properly sorted at index %d\n", i);
    //         break;
    //     }
    // }
    // if(sorted) {
    //     printf("Verification: Arrays successfully sorted in descending order!\n");
    // }
    
    // Cleanup
    CHECK_CUDA(cudaFreeHost(h_fitness));
    CHECK_CUDA(cudaFreeHost(h_params));
    CHECK_CUDA(cudaFree(d_fitness));
    CHECK_CUDA(cudaFree(d_params));
    
    return 0;
}