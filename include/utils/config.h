#ifndef CUDA_PROCESS_CONSTANTS_H
#define CUDA_PROCESS_CONSTANTS_H

#define DEBUG_PRINT_FLAG false
#define DEBUG_PRINT_EVALUATE_FLAG false
#define DEBUG_PRINT_SOLVER_FLAG false
#define DEBUG_PRINT_INIT_SOLVER_FLAG false
#define DEBUG_PRINT_WARM_START_FLAG false
#define DEBUG_ENABLE_NVTX false
#define DEBUG_FOOTSTEP false

#define HOST_DEVICE __device__ __forceinline__ __host__
#define CUDA_SOLVER_POP_SIZE 1024
#define REGENRATE_RANDOM_FREQUENCE 1000

#define CUDA_PARAM_MAX_SIZE 64
// #define CUDA_SOLVER_POP_SIZE 128
#define CUDA_MAX_FLOAT 1e30
#define CUDA_MAX_TASKS 1
#define CUDA_MAX_POTENTIAL_SOLUTION 4
#define CUDA_MAX_ROUND_NUM 300000

#define BEZIER_SIZE 7

#define INT_VARIABLE 0
#define CONTINUOUS_VARIABLE 1

#endif