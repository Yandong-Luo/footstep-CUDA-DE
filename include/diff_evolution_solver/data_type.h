#ifndef CUDA_DIFF_EVOLUTION_DATA_TYPE_H
#define CUDA_DIFF_EVOLUTION_DATA_TYPE_H
#include <unordered_map>
// #include "utils/utils.cuh"
#include "utils/config.h"
#include "footstep/footstep_utils.cuh"

namespace cudaprocess{

#define ALIGN(n) __align__(n)

enum CudaEvolveType { CON = 0, INT = 1, GLOBAL = 2 };


template <typename T, int size>
struct CudaVector {
  T data[size];
  int len{size};
};

struct Problem{
    int num_continous = 2;
    int num_int = 1;

    float *int_upper_bound = nullptr;
    float *int_lower_bound = nullptr;
    float *con_upper_bound = nullptr;
    float *con_lower_bound = nullptr;

    // float int_upper_bound[1] = {20};
    // float int_lower_bound[1] = {0};
    // float con_upper_bound[2] = {10., 10.};
    // float con_lower_bound[2] = {0, 0};

    // matrix dims
    int row_objective_mat, col_objective_mat;
    int row_constraint_mat, col_constraint_mat;   // row x col should equal to num_constraint x constraint variable + 1 (constant).
    int row_lambda, col_lambda;

    int row_objective_Q, col_objective_Q;

    float *objective_mat = nullptr;
    float *constraint_mat = nullptr;
    float *lambda_mat = nullptr;

    float *objective_Q_mat = nullptr;
    
    // float objective_mat[4] = {-4, -3, -5, 0};
    // float constraint_mat[4][2] = {{2, 2}, {3, 1}, {1, 3}, {-12, -12}};

    // float lambda[2] = {100, 100};

    int max_lambda;
    int init_lambda;
    int max_evolve_round;

    float accuracy_rng = 0.5;
    int elite_eval_count = 8;

    // ~Problem() {
    //     delete[] int_upper_bound;
    //     delete[] int_lower_bound;
    //     delete[] con_upper_bound;
    //     delete[] con_lower_bound;
    //     delete[] objective_mat;
    //     delete[] constraint_mat;
    //     delete[] lambda_mat;
    //     delete[] objective_Q_mat;
    // }

    // // (Abandoned) Use for loop to evaluate 
    // int num_constraint = 2;
    // int num_constraint_variable = 4;
    // // objective function
    // float objective_param[3] = {-4., -3, -5};
    // int num_objective_param = 3;
    // float constraint_param[2][4] = {
    //     {2, 3, 1, -12},
    //     {2, 1, 3, -12}
    // };
};

/*
 * Param:
 * prob_f:  scaling factor for L-Shape 
 * Cr:      Crossover Probability
 * weight:  weight factor
 */
struct ALIGN(16) CudaLShadePair {
    // float3 will better than this struct
    float scale_f, scale_f1, Cr, weight;
};

// static constexpr int N_states_dims = footstep::N * footstep::state_dims;

struct CudaParamIndividual{
    float param[CUDA_PARAM_MAX_SIZE];
    // float state[footstep::N * footstep::state_dims];
    int con_var_dims, int_var_dims, dims;
    float cur_scale_f{0.8F}, cur_scale_f1{0.8F}, cur_Cr{0.9F};
    float fitness;
    // float objective_score;
    // float constraint_score;
    float *N_control;
    float *N_states;
    // float 
};

/*
 * Store individual variables in a cluster in a more compact way
 */
template <int T>
struct ALIGN(64) CudaParamClusterData{
    float all_param[CUDA_PARAM_MAX_SIZE * T];
    float lshade_param[T * 3];  // current scale_f, current scale_f1, current crossover probability
    // float all_state[footstep::N * footstep::state_dims * T];
    int con_var_dims, int_var_dims, dims;
    float fitness[T];
    // float objective_score[T];
    // float constraint_score[T];
    int len{0};
};

struct ALIGN(64) CudaProblemParam{
    int con_var_dims, int_var_dims, dims;
    int max_lambda;
    int init_lambda;
    int max_round;

    float accuracy_rng;
    int elite_eval_count;

    float top_ratio;
};

struct CudaEvolveData{
    
    CudaLShadePair hist_lshade_param;
    ALIGN(64) float upper_bound[CUDA_PARAM_MAX_SIZE];
    ALIGN(64) float lower_bound[CUDA_PARAM_MAX_SIZE];
    CudaVector<CudaParamIndividual, CUDA_MAX_POTENTIAL_SOLUTION> last_potential_sol;
    CudaParamIndividual warm_start;

    CudaProblemParam problem_param;
};

}


#endif