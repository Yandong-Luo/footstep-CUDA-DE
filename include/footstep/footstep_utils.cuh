#ifndef CUDAPROCESS_FOOTSTEP_UTILS_H
#define CUDAPROCESS_FOOTSTEP_UTILS_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include "utils/utils.cuh"

#ifndef CUDA_SOLVER_POP_SIZE
#define CUDA_SOLVER_POP_SIZE 512
#endif

namespace footstep{

// CONSTANT
const int N = 30;                           // prediction step
const float T = 0.4f;           // Delta t

const float legLength = 1.0f;
const float g = 9.81f;
constexpr float PI = 3.14159265358979323846f;
const float omega = std::sqrt(g / legLength);

const float penalty = 1000.0f;

// u boundary
const float ux_lb = -0.25;
const float ux_ub = 0.25;
const float uy_lb = -0.25;
const float uy_ub = 0.25;
const float utheta_lb = -PI / 12.0f;
const float utheta_ub = PI / 12.0f;

// const int num_constraints = 50;     // n_c in paper
const int num_constraints = 12;     // n_c in paper
const int state_dims = 5;           // n_x in paper: x, y, dot x, dot y, theta, 
const int control_dims = 3;         // n_u in paper: u_x, u_y, u_theta
const int num_regions = 7;          // n_delta in paper: 7 differents regions

// x: x lower boundary, y: x upper boundary, z: y lower boundary, w: y upper boundary
extern __constant__ float4 region1;
extern __constant__ float4 region2;
extern __constant__ float4 region3;
extern __constant__ float4 region4;
extern __constant__ float4 region5;
extern __constant__ float4 region6;
extern __constant__ float4 region7;

const int row_init_state = 5, col_init_state = 1;
extern __constant__ float init_state[5];
extern float *d_init_state;

extern float *N_state;
extern float *h_N_state;

// E Matrix (5x5), Row priority
const int row_E = state_dims, col_E = state_dims;
extern float *d_E;
extern float h_E[25];

// Matrix F (5x3), Row priority
const int row_F = state_dims, col_F = control_dims;
extern float *d_F;
extern float h_F[15];

void InitMatrixEAndF();

// Matrix F (5x7), Row priority
const int row_G = state_dims, col_G = num_regions;
extern __constant__ float G[35];


// Q M (5x5), Row priority
const int row_Q = state_dims, col_Q = state_dims;
extern __constant__ float Q[25];

// R Matrix (3x3), Row priority
const int row_R = control_dims, col_R = control_dims;
extern __constant__ float R[9];

// H1 Matrix (12x5), Row priority
const int row_H1 = num_constraints, col_H1 = state_dims;
extern __constant__ float H1[60];

// H2 Matrix (12x3), Row priority
const int row_H2 = num_constraints, col_H2 = control_dims;
extern __constant__ float H2[36];

const int row_h = num_constraints, col_h = 1;
extern __constant__ float h[12];

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

// ##############################
// ########## DEBUG #############
// ##############################

extern __managed__ float cluster_N_state[N * CUDA_SOLVER_POP_SIZE * state_dims];

extern float h_cluster_param[N * CUDA_SOLVER_POP_SIZE * control_dims];
}

#endif