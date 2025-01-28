#ifndef CUDAPROCESS_FOOTSTEP_UTILS_H
#define CUDAPROCESS_FOOTSTEP_UTILS_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

// the boundary of position
const float pos_lb = -1.0f;
const float pos_ub = 1.f;

// the boundary of theta
const float theta_lb = -PI/2;
const float theta_ub = PI/2;

// the boundary of speed
const float speed_lb = -2 * (pos_ub - pos_lb) / runtime_step;
const float speed_ub = 2 * (pos_ub - pos_lb) / runtime_step;

// the boundary of \dot theta
const float dtheta_lb = -PI / runtime_step;
const float dtheta_ub = PI / runtime_step;

// const float dtheta_lb = -1;
// const float dtheta_ub = 1;

// the boundary of control input (u)
const float u_lb = -20.0f;
const float u_ub = 20.0f;

const float d_left = 0.40f;
const float d_right = 0.35f;
const float d_max = 0.6f;
const float lam_max = 30.0f;


// const int num_constraints = 50;     // n_c in paper
const int num_constraints = 12;     // n_c in paper
const int state_dims = 5;           // n_x in paper: x, y, dot x, dot y, theta, 
const int control_dims = 3;         // n_u in paper: u_x, u_y, u_theta
const int num_regions = 7;          // n_delta in paper: 7 differents regions
const int u_dims = 1;

// x: x lower boundary, y: x upper boundary, z: y lower boundary, w: y upper boundary
const float4 region1 = {0, 1, 0, 3};
const float4 region2 = {1, 2, 0, 1};
const float4 region3 = {2, 3, 0, 3};
const float4 region4 = {1, 2, 2, 3};
const float4 region5 = {0, 0.2, 1.4, 1.6};
const float4 region6 = {1.4, 1.6, 2.8, 3};
const float4 region7 = {2.8, 3, 1.4, 1.6};

const int row_C = N * num_constraints;
const int col_C = N * (state_dims + control_input_dims) + state_dims;       // 

const int row_A = (N + 1) * state_dims;                                     // (N + 1) * nx in paper
const int col_A = N * (state_dims + control_input_dims) + state_dims;       // N * n_{xu} + nx in paper

// E Matrix (4x4), Row priority
const int row_E = state_dims, col_E = state_dims;
extern __constant__ float E[25];
// __constant__ float E[16] = {
//     1.0f + 0.0f*runtime_step, 0.0f*runtime_step,        runtime_step,      0.0f*runtime_step,
//     0.0f*runtime_step,        1.0f + 0.0f*runtime_step, 0.0f,    runtime_step,
//     0.0f*runtime_step,        g*mp/mc*runtime_step,      1.0f,    0.0f*runtime_step,
//     0.0f*runtime_step,        g*(mc+mp)/(ll*mc)*runtime_step, 0.0f, 1.0f
// };

// // Matrix F (4x3), Row priority
// const int row_F = 4, col_F = 3;
// __constant__ float F[12] = {
//     0.0f,         0.0f,         0.0f,
//     0.0f,         0.0f,         0.0f,
//     runtime_step/mc,        0.0f,         0.0f,
//     runtime_step/(ll*mc),   runtime_step/(ll*mp),  runtime_step/(ll*mp)
// };

// Matrix F (5x3), Row priority
const int row_F = state_dims, col_F = control_dims;
extern __constant__ float F[15];
// __constant__ float F[4] = {
//     0.0f,
//     0.0f,
//     runtime_step/mc,
//     runtime_step/(ll*mc)
// };

// Matrix F (5x7), Row priority
const int row_G = state_dims, col_G = num_regions;
extern __constant__ float G[35];
// __constant__ float G[8] = {
//     0.0f,                   0.0f,
//     0.0f,                   0.0f,
//     0.0f,                   0.0f,
//     runtime_step/(ll*mp),  runtime_step/(ll*mp)
// };

// // Matrix G (4x2), Row priority
// const int row_G = 4, col_G = 2;
// __constant__ float G[8] = {
//     0.0f, 0.0f,
//     0.0f, 0.0f,
//     0.0f, 0.0f,
//     0.0f, 0.0f
// };

// Q M (5x5), Row priority
const int row_Q = state_dims, col_Q = state_dims;
extern __constant__ float Q[25]; 
// __constant__ float Q[16] = {
//     1.0f,  0.0f,  0.0f,  0.0f,
//     0.0f, 50.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  1.0f,  0.0f,
//     0.0f,  0.0f,  0.0f, 50.0f
// };

// R Matrix (3x3), Row priority
const int row_R = control_dims, col_R = control_dims;
extern __constant__ float R[9];
// __constant__ float R[9] = {
//     0.1f, 0.0f, 0.0f,
//     0.0f, 0.1f, 0.0f,
//     0.0f, 0.0f, 0.1f
// };

// const int row_E = state_dims, col_E = state_dims;
// extern thrust::device_vector<float> E[row_E * col_E]

// H1 Matrix (12x5), Row priority
const int row_H1 = num_constraints, col_H1 = state_dims;
extern __constant__ float H1[60];
// __constant__ float H1[80] = {
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     -1.0f,  ll,   0.0f,  0.0f,
//     1.0f,  -ll,   0.0f,  0.0f,
//     1.0f,  -ll,   0.0f,  0.0f,
//     -1.0f,  ll,   0.0f,  0.0f,
//     1.0f,  0.0f,  0.0f,  0.0f,
//     -1.0f, 0.0f,  0.0f,  0.0f,
//     0.0f,  1.0f,  0.0f,  0.0f,
//     0.0f, -1.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  1.0f,  0.0f,
//     0.0f,  0.0f, -1.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  1.0f,
//     0.0f,  0.0f,  0.0f, -1.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f,
//     0.0f,  0.0f,  0.0f,  0.0f
// };

// H2 Matrix (12x3), Row priority
const int row_H2 = num_constraints, col_H2 = control_dims;
extern __constant__ float H2[36];

// H3 Matrix (50x7), Row priority
// const int row_H3 = num_constraints, col_H3 = num_regions;
// extern __constant__ float H3[350];

const int row_h = num_constraints, col_h = 1;
extern __constant__ float h[12];

// const int row_Inx = state_dims, col_Inx = state_dims;
// extern __constant__ float Inx[16];

/**
 * Evaluation
 */
const int row_state_weight = N * state_dims, col_state_weight = N * state_dims;                       // 40 x 40

const float control_weight = 0.1;

const int row_state_matrix = CUDA_SOLVER_POP_SIZE, col_state_matrix = state_dims * N;       // CUDA_SOLVER_POP_SIZE x 40

// record the result of weight(Q) x state^T (Qx in formula)
const int row_temp_state_score = state_dims * N, col_temp_state_score = CUDA_SOLVER_POP_SIZE;   // 40 x CUDA_SOLVER_POP_SIZE

// record the result of state x (weight(Q) x state^T)
const int row_quadratic_state = CUDA_SOLVER_POP_SIZE, col_quadratic_state = CUDA_SOLVER_POP_SIZE;   // CUDA_SOLVER_POP_SIZE x CUDA_SOLVER_POP_SIZE

// record the diag of quadratic_state as the state score for all individual
const int row_state_score = CUDA_SOLVER_POP_SIZE, col_state_score = 1;

// control
const int row_control_matrix = CUDA_SOLVER_POP_SIZE, col_control_matrix = N;                // CUDA_SOLVER_POP_SIZE x 10

const int row_control_weight = N, col_control_weight = N;                                   // 10 x 10

// record the result of weight(R) x control^T (Ru in formula)
const int row_temp_control_score = N, col_temp_control_score = CUDA_SOLVER_POP_SIZE;        // 10 x CUDA_SOLVER_POP_SIZE

// record the result of state x (weight(Q) x state^T)
const int row_quadratic_control = CUDA_SOLVER_POP_SIZE, col_quadratic_control = CUDA_SOLVER_POP_SIZE;   // CUDA_SOLVER_POP_SIZE x CUDA_SOLVER_POP_SIZE

// record the diag of quadratic_state as the state score for all individual
const int row_control_score = CUDA_SOLVER_POP_SIZE, col_control_score = 1;                      // CUDA_SOLVER_POP_SIZE x 1


const int row_score = CUDA_SOLVER_POP_SIZE, col_score = 1;

extern __constant__ float4 current_state;
extern __constant__ float2 current_wall_pos;
}

#endif