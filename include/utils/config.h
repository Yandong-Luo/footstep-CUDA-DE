#ifndef CUDA_PROCESS_CONSTANTS_H
#define CUDA_PROCESS_CONSTANTS_H

#define DEBUG_PRINT_FLAG false
#define DEBUG_PRINT_EVALUATE_FLAG false
#define DEBUG_PRINT_SOLVER_FLAG false
#define DEBUG_PRINT_INIT_SOLVER_FLAG false
#define DEBUG_PRINT_WARM_START_FLAG false
#define DEBUG_ENABLE_NVTX false
#define DEBUG_FOOTSTEP true

#define HOST_DEVICE __device__ __forceinline__ __host__
#define CUDA_SOLVER_POP_SIZE 64
#define REGENRATE_RANDOM_FREQUENCE 1000

#define CUDA_PARAM_MAX_SIZE 32
// #define CUDA_SOLVER_POP_SIZE 128
#define CUDA_MAX_FLOAT 1e30
#define CUDA_MAX_TASKS 1
#define CUDA_MAX_POTENTIAL_SOLUTION 4
#define CUDA_MAX_ROUND_NUM 300000

#define INT_VARIABLE 0
#define CONTINUOUS_VARIABLE 1


// **************Bezier Curve***************
#define BEZIER_SIZE 7   // 6 + 1
#define CURVE_NUM_STEPS (footstep::N + 1)
#define NUM_XYFIXED_CP 4	// the number of fix control point, P_0, P_1, P_{n-1}, P_n
#define NUM_THETA_FIXED_CP 2

#define X_START 0
#define Y_START (BEZIER_SIZE - NUM_XYFIXED_CP)
#define THETA_START (2*(BEZIER_SIZE - NUM_XYFIXED_CP))

// **************ARCH LENGTH PARAMETER**************
#define ARC_LENGTH_SAMPLES 100  // Number of samples for arc-length calculation

#endif