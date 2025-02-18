#ifndef CUDAPROCESS_FOOTSTEP_UTILS_H
#define CUDAPROCESS_FOOTSTEP_UTILS_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
// #include "utils/config.h"
#include "utils/utils.cuh"

// #ifndef CUDA_SOLVER_POP_SIZE
// #define CUDA_SOLVER_POP_SIZE 512
// #endif

// #ifndef CUDA_PARAM_MAX_SIZE
// #define CUDA_PARAM_MAX_SIZE 32
// #endif

namespace footstep{

// CONSTANT
constexpr int N = 20;                           // prediction step
constexpr float T = 0.4f;           // Delta t

constexpr float legLength = 1.0f;
constexpr float g = 9.81f;
constexpr float PI = 3.14159265358979323846f;
const float omega = std::sqrt(g / legLength);

// const float penalty = 1000.0f;

// u boundary
constexpr float ux_lb = -0.25;
constexpr float ux_ub = 0.25;
constexpr float uy_lb = -0.25;
constexpr float uy_ub = 0.25;
const float utheta_lb = -PI / 12.0f;
const float utheta_ub = PI / 12.0f;

// state boundary
constexpr float speed_x_lb = -0.5;
constexpr float speed_x_ub = 0.5;
constexpr float speed_y_lb = -0.5;
constexpr float speed_y_ub = 0.5;
const float theta_lb = -5.0f * PI / 4.0f;
const float theta_ub = 5.0f * PI / 4.0f;

// const int num_constraints = 50;     // n_c in paper
constexpr int num_constraints = 12;     // n_c in paper
constexpr int state_dims = 5;           // n_x in paper: x, y, dot x, dot y, theta, 
constexpr int control_dims = 3;         // n_u in paper: u_x, u_y, u_theta

constexpr float scale = 1.0f;   // for environment2

constexpr int num_regions = 7;          // n_delta in paper: 7 differents regions (if environment2 we have 14 regions)
// x: x lower boundary, y: x upper boundary, z: y lower boundary, w: y upper boundary
extern __constant__ float4 all_region[num_regions];

// constexpr int num_regions = 14;          // n_delta in paper: 7 differents regions (if environment2 we have 14 regions)
// // Environment2 x: x lower boundary, y: x upper boundary, z: y lower boundary, w: y upper boundary
// extern __constant__ float4 all_region2[num_regions];

extern __constant__ float4 region1;
extern __constant__ float4 region2;
extern __constant__ float4 region3;
extern __constant__ float4 region4;
extern __constant__ float4 region5;
extern __constant__ float4 region6;
extern __constant__ float4 region7;

constexpr float Mx = 10.0f;
constexpr float My = 10.0f;
constexpr float Mu = 5.0f;
constexpr float Mt = 7.0f;

constexpr bool left_stand_first = false;

constexpr int first_step_num = left_stand_first? 0 : 1;

constexpr int row_init_state = 5, col_init_state = 1;
extern __constant__ float init_state[5];

// circle center
constexpr int foothold_circle_num = 2;
extern __constant__ float2 foothold_circles[foothold_circle_num];
extern __constant__ float2 foothold_circles2[foothold_circle_num];
extern __constant__ float foothold_radii[foothold_circle_num];

// velocity circle center
constexpr int vel_circle_num = 2;
extern __constant__ float2 vel_circle[vel_circle_num];
extern __constant__ float vel_circle_radii[vel_circle_num];


extern __constant__ float2 obj_circle;
extern __constant__ float2 obj_circle2;

// target
extern __constant__ float2 target_pos;

// __constant__ float2 center1 = {0, 1};
// __constant__ float2 center2 = {0, -0.44};


// E Matrix (5x5), Row priority
const int row_E = state_dims, col_E = state_dims;
extern float *d_E;
extern float h_E[25];

// Matrix F (5x3), Row priority
const int row_F = state_dims, col_F = control_dims;
extern float *d_F;
extern float h_F[15];

void InitMatrixEAndF();

// // Matrix R (3x3), Row priority
// const float row_R = control_dims, col_R = control_dims;
// extern __constant__ float R[9];

// // Matrix G (5x7), Row priority
// const int row_G = state_dims, col_G = num_regions;
// extern __constant__ float G[35];


// // Q M (5x5), Row priority
// const int row_Q = state_dims, col_Q = state_dims;
// extern __constant__ float Q[25];

// // R Matrix (3x3), Row priority
// const int row_R = control_dims, col_R = control_dims;
// extern __constant__ float R[9];

// // H1 Matrix (12x5), Row priority
// const int row_H1 = num_constraints, col_H1 = state_dims;
// extern __constant__ float H1[60];

// // H2 Matrix (12x3), Row priority
// const int row_H2 = num_constraints, col_H2 = control_dims;
// extern __constant__ float H2[36];

// const int row_h = num_constraints, col_h = 1;
// extern __constant__ float h[12];

// const int row_state = N

// thrust::device_vector<float> Ek_record(row_E * col_E); 
// __device__ float Ek_record[N][row_E * col_E];

// bigE Matrix (150x5)
const int row_bigE = N * state_dims, col_bigE = state_dims;
// extern float h_bigE[750];
extern float *bigE;
extern float *h_bigE;

void ConstructEandF(cudaStream_t stream);

void ConstructBigEAndF(float *bigE, float *bigF, cublasHandle_t handle, cudaStream_t stream);

// bigF Matrix ()
const int row_bigF = N * state_dims, col_bigF = N * control_dims;
// extern float h_bigF[13500];

extern float *bigF;
extern float *h_bigF;

extern float *d_sol_state;
extern float *h_sol_state;

extern float *d_sol_score;
extern float *h_sol_score;

// ################################
// ########## Penalty #############
// ################################
constexpr float pos_penalty = 50000.0f;
constexpr float state_penalty = 500.0f;
constexpr float control_penalty = 200.0f;
constexpr float velocity_penalty = 10.0f;
constexpr float foothold_penalty = 200.0f;

// the weight of the distance between N position and target position
constexpr float target_weight = 200.0f;

// ##############################
// ########## DEBUG #############
// ##############################

extern float *d_cluster_N_state;

extern float h_cluster_N_state[N * CUDA_SOLVER_POP_SIZE * state_dims];

extern float h_cluster_param[CUDA_SOLVER_POP_SIZE * CUDA_PARAM_MAX_SIZE];
}

#endif