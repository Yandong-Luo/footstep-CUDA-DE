#include "diff_evolution_solver/solver.cuh"
#include "diff_evolution_solver/decoder.cuh"
#include "diff_evolution_solver/debug.cuh"
#include "diff_evolution_solver/evolve.cuh"
#include "diff_evolution_solver/evaluate.cuh"
// #include "cart_pole/cart_pole_utils.cuh"
// #include "cart_pole/model.cuh"
// #include "cart_pole/evaluate.cuh"
#include "footstep/footstep_utils.cuh"
#include "footstep/model.cuh"
#include "utils/utils_fun.cuh"
#include <math.h>

namespace cudaprocess{

void CudaDiffEvolveSolver::MallocSetup(){
    CHECK_CUDA(cudaSetDevice(gpu_device_));

    // GPU Device
    // CHECK_CUDA(cudaMalloc(&decoder_, sizeof(CudaProblemDecoder)));
    CHECK_CUDA(cudaMalloc(&evolve_data_, sizeof(CudaEvolveData)));
    CHECK_CUDA(cudaMalloc(&new_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE>)));
    CHECK_CUDA(cudaMalloc(&old_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE * 3>)));
    // CHECK_CUDA(cudaMalloc(&new_cluster_vec_, sizeof(CudaVector<CudaParamIndividual, CUDA_SOLVER_POP_SIZE>)));
    // CHECK_CUDA(cudaMalloc(&problem_, sizeof(Problem)));
    CHECK_CUDA(cudaMalloc(&evaluate_score_, CUDA_SOLVER_POP_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&last_fitness, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&terminate_flag, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&result, sizeof(CudaParamIndividual)));

    // objective, constraint, constraint_score, lambda, parameter matrix
    // CHECK_CUDA(cudaMalloc(&constraint_matrix, row_constraint * col_constraint * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&objective_matrix, row_obj * col_obj * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&constraint_score, CUDA_SOLVER_POP_SIZE * row_constraint * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&lambda_matrix, row_lambda * col_lambda * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&param_matrix, (dims_ + 1) * CUDA_SOLVER_POP_SIZE * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&objective_Q_matrix, row_obj_Q * col_obj_Q * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&quad_matrix, CUDA_SOLVER_POP_SIZE * CUDA_SOLVER_POP_SIZE * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&quad_transform, row_obj_Q * CUDA_SOLVER_POP_SIZE * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&quadratic_score, 1 * CUDA_SOLVER_POP_SIZE * sizeof(float)));
    

    // CPU Host
    // CHECK_CUDA(cudaHostAlloc(&h_terminate_flag, sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_result, sizeof(CudaParamIndividual), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&host_evolve_data_, sizeof(CudaEvolveData), cudaHostAllocDefault));

    // CHECK_CUDA(cudaHostAlloc(&host_evolve_data_, sizeof(CudaEvolveData), cudaHostAllocDefault));

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG || DEBUG_PRINT_WARM_START_FLAG){
        CHECK_CUDA(cudaHostAlloc(&host_new_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE>), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&host_old_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3>), cudaHostAllocDefault));
    }
    
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_EVALUATE_FLAG){
        // objective, constraint, constraint_score, lambda, parameter, score matrix
        // CHECK_CUDA(cudaHostAlloc(&h_constraint_matrix, row_constraint * col_constraint * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&h_objective_matrix, row_obj * col_obj * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&host_constraint_score, CUDA_SOLVER_POP_SIZE * row_constraint * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&h_lambda_matrix, row_lambda * col_lambda * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&host_param_matrix, (dims_ + 1) * CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&h_objective_Q_matrix, row_obj_Q * col_obj_Q * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&host_evaluate_score_, CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault));

        // CHECK_CUDA(cudaHostAlloc(&host_quad_matrix, CUDA_SOLVER_POP_SIZE * CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&h_quad_transform, row_obj_Q * CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault));
        // CHECK_CUDA(cudaHostAlloc(&h_quadratic_score, 1 * CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault));
    }

    // !--------------- Footstep ---------------!

    CHECK_CUDA(cudaMalloc(&footstep::d_E, footstep::row_E * footstep::col_E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&footstep::d_F, footstep::row_F * footstep::col_F * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&footstep::bigE, footstep::row_bigE * footstep::col_bigE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&footstep::bigF, footstep::row_bigF * footstep::col_bigF * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&footstep::d_cluster_N_state, footstep::N * CUDA_SOLVER_POP_SIZE * footstep::state_dims * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&footstep::d_sol_state, footstep::state_dims * footstep::N * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&footstep::d_sol_score, 3 * sizeof(float)));  // all socre, objective score, constraint score

    CHECK_CUDA(cudaHostAlloc(&footstep::h_sol_state, footstep::state_dims * footstep::N * sizeof(float), cudaHostAllocDefault)); 

    CHECK_CUDA(cudaHostAlloc(&footstep::h_sol_score, 3 * sizeof(float), cudaHostAllocDefault)); 

    if (DEBUG_PRINT_FLAG || DEBUG_FOOTSTEP){
        // printf("Debug flags enabled, allocating host memory\n");
        printf("bigE size:%d\n", footstep::row_bigE * footstep::col_bigE);
        printf("bigF size:%d\n", footstep::row_bigF * footstep::col_bigF);
        
        CHECK_CUDA(cudaHostAlloc(&footstep::h_bigE, footstep::row_bigE * footstep::col_bigE * sizeof(float), cudaHostAllocDefault));  
        CHECK_CUDA(cudaHostAlloc(&footstep::h_bigF, footstep::row_bigF * footstep::col_bigF * sizeof(float), cudaHostAllocDefault));    

        // CHECK_CUDA(cudaHostAlloc(&footstep::h_N_state, footstep::state_dims * footstep::N * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&host_evaluate_score_, CUDA_SOLVER_POP_SIZE * sizeof(float), cudaHostAllocDefault)); 
    }
    cuda_utils_ = std::make_shared<CudaUtil>();

    cudamalloc_flag = true;
}

void CudaDiffEvolveSolver::InitDiffEvolveParam(float top, float d_top, float min_top, float diff, float d_diff, float min_diff, float scale_f, float prob_crossover){
    top_ = top;
    d_top_ = d_top;
    min_top_ = min_top;
    diff_ = diff;
    d_diff_ = d_diff;
    min_diff_ = min_diff;
    
    lshade_param_.scale_f = lshade_param_.scale_f1 = scale_f;
    lshade_param_.Cr = prob_crossover;

}

__global__ void InitCudaEvolveData(CudaEvolveData* evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3>* old_cluster_data, int pop_size){
    int idx = threadIdx.x;
    if (idx == 0) {
        evolve->problem_param.top_ratio = 0.;
        evolve->hist_lshade_param.scale_f = evolve->hist_lshade_param.scale_f1 = 0.6;
        evolve->hist_lshade_param.Cr = 0.9;
        // evolve->new_cluster_vec->len = pop_size;
        old_cluster_data->len = pop_size;
    }
    if (idx < pop_size){
        // initial the each parameter in old_cluster 
        for (int i = 0; i < CUDA_PARAM_MAX_SIZE; ++i){
            old_cluster_data->all_param[(idx + pop_size) * CUDA_PARAM_MAX_SIZE + i] = 0.f;
        }
        old_cluster_data->fitness[idx + pop_size] = CUDA_MAX_FLOAT;
        // old_cluster_data->objective_score[idx + pop_size] = CUDA_MAX_FLOAT;
        // old_cluster_data->constraint_score[idx + pop_size] = CUDA_MAX_FLOAT;
    }
}

void CudaDiffEvolveSolver::SetBoundary(){
    for(int i = 0; i < dims_; ++i){
        if (i % footstep::control_dims == 0){
            host_evolve_data_->lower_bound[i] = footstep::ux_lb;
            host_evolve_data_->upper_bound[i] = footstep::ux_ub;
        }
        else if(i % footstep::control_dims == 1){
            host_evolve_data_->lower_bound[i] = footstep::uy_lb;
            host_evolve_data_->upper_bound[i] = footstep::uy_ub;
        }
        else{
            host_evolve_data_->lower_bound[i] = footstep::utheta_lb;
            host_evolve_data_->upper_bound[i] = footstep::utheta_ub;
        }
        // printf("current %d low bound:%f\n", i, host_evolve_data_->lower_bound[i]);
    }
}

/**
 * CudaEvolveData* ----> CudaParamClusterData<T> *
 */
__global__ void UpdateClusterDataBasedEvolve(CudaEvolveData* evolve_data, CudaParamClusterData<CUDA_SOLVER_POP_SIZE>* new_cluster_data, int num_last_potential_sol){
    int idx = blockIdx.x;
    if (idx >= num_last_potential_sol)   return;
    ConvertCudaParam<CUDA_SOLVER_POP_SIZE>(new_cluster_data, &evolve_data->last_potential_sol.data[idx], idx, threadIdx.x);
}

/**
 * CudaParamClusterData ----> CudaParamIndividual * as output
 */
template <int T>
__global__ void UpdateVecParamBasedClusterData(CudaParamIndividual *output, CudaParamClusterData<T> *cluster_data){
    ConvertCudaParamRevert<T>(cluster_data, &output[blockIdx.x], blockIdx.x, threadIdx.x);
}

/**
 * CudaParamClusterData<T> * ---->  CudaEvolveData* 
 */
__global__ void UpdateEvolveWarmStartBasedClusterData(CudaEvolveData *evolve_data, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param){
    ConvertCudaParamRevert<CUDA_SOLVER_POP_SIZE*3>(old_param, &evolve_data->warm_start, 0, threadIdx.x);
}

__global__ void SaveNewParamAsOldParam(CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, int left_bound, int right_bound, int bias){
    int sol_id = blockIdx.x;
    if (sol_id < left_bound || sol_id >= right_bound)   return;
    ConvertCudaParamBetweenClusters<CUDA_SOLVER_POP_SIZE, CUDA_SOLVER_POP_SIZE*3>(new_param, old_param, sol_id, sol_id + bias, threadIdx.x);
}

__global__ void GenerativeRandSolNearBest(CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, float *uniform_data, int rand_idx, float delta_con, float delta_int, int size){
    int sol_id = blockIdx.x;
    int param_id = threadIdx.x;

    if (sol_id == 0 || sol_id >= size)  return;
    float upper_bound = evolve->upper_bound[param_id];
    float lower_bound = evolve->lower_bound[param_id];

    if (param_id < evolve->problem_param.con_var_dims){
        float rand_range = (upper_bound - lower_bound) * delta_con;

        // based on rand_range update the boundary
        upper_bound = min(upper_bound, new_param->all_param[param_id] + rand_range);
        lower_bound = max(lower_bound, new_param->all_param[param_id] - rand_range);
        
        // based on new boundary near parameter to generate the new parameter
        new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
    }
    else if(param_id < evolve->problem_param.int_var_dims){
        float rand_range = (upper_bound - lower_bound) * delta_int;

        // based on rand_range update the boundary
        upper_bound = min(upper_bound, new_param->all_param[param_id] + rand_range);
        lower_bound = max(lower_bound, new_param->all_param[param_id] - rand_range);
        
        // based on new boundary near parameter to generate the new parameter
        new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
    }
}

// __global__ void GenerativeHeuristicsParam(CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, float *uniform_data, int rand_idx, float delta_con, float delta_int, int heuristics_size){
//     __shared__ float2 expected_force;
//     int sol_id = blockIdx.x;
//     int param_id = threadIdx.x;

//     if(sol_id >= heuristics_size)  return;

//     if (threadIdx.x == 0 && blockIdx.x == 0){
//         float pos = cart_pole::current_state.x, speed = cart_pole::current_state.y, theta = cart_pole::current_state.z, dtheta = cart_pole::current_state.w;
//         float right_wall_pos = cart_pole::current_wall_pos.x, left_wall_pos = cart_pole::current_wall_pos.y;

//         float pole_pos1 = -(cart_pole::ll * __sinf(theta) - pos), pole_pos2 = (cart_pole::ll * __sinf(theta) - pos);
//         float lam1 = 0.0f, lam2 = 0.0f;
//         if(pole_pos1 >= cart_pole::d_right){
//             lam1 = cart_pole::k1 * (right_wall_pos - cart_pole::d_right);
//             // printf("contact right wall, force:%f\n",lam1);
//         }
//         else if(pole_pos2 <= -cart_pole::d_left){
//             lam2 = cart_pole::k2 * (left_wall_pos - cart_pole::d_left);
//             printf("contact left wall, force:%f\n",lam2);
//         }
//         expected_force.x = -(-dtheta*dtheta*cart_pole::ll*cart_pole::mp*__sinf(theta) + cart_pole::g*cart_pole::mp*__sinf(2*theta)/2 + lam1*__cosf(theta)*__cosf(theta) - lam1 - lam2*__cosf(theta)*__cosf(theta) + lam2);
//         expected_force.y = -(-dtheta*dtheta*cart_pole::ll*cart_pole::mp*cart_pole::mp*__sinf(2*theta)/2 + cart_pole::g*cart_pole::mc*cart_pole::mp*__sinf(theta) + cart_pole::g*cart_pole::mp*cart_pole::mp*__sinf(theta) + lam1*cart_pole::mc*__cosf(theta) - lam2*cart_pole::mc*__cosf(theta))/(cart_pole::mp *__cosf(theta));
//         // printf("expected force1:%f, expected force2:%f\n",expected_force.x, expected_force.y);
//     }

//     float upper_bound = evolve->upper_bound[param_id];
//     float lower_bound = evolve->lower_bound[param_id];
//     int half_heuristics_size = heuristics_size >> 1;
//     // float expect_u = (sol_id >= half_heuristics_size)? expected_force.x : expected_force.y;
//     float expect_u = (sol_id >= half_heuristics_size)? expected_force.x : expected_force.y;

//     if (param_id < evolve->problem_param.con_var_dims){
//         float rand_range = (upper_bound - lower_bound) * delta_con;

//         // based on rand_range update the boundary
//         upper_bound = min(upper_bound, expect_u + rand_range);
//         lower_bound = max(lower_bound, expect_u - rand_range);
        
//         // based on new boundary near parameter to generate the new parameter
//         new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
//     }
//     else if(param_id < evolve->problem_param.int_var_dims){
//         float rand_range = (upper_bound - lower_bound) * delta_int;

//         // based on rand_range update the boundary
//         upper_bound = min(upper_bound, expect_u + rand_range);
//         lower_bound = max(lower_bound, expect_u - rand_range);
        
//         // based on new boundary near parameter to generate the new parameter
//         new_param->all_param[sol_id * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 100 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
//     }
// }

// __global__ void GenerativeRandomParamFromLastSol(CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param, float *uniform_data, int rand_idx, float delta_con, float delta_int, int size, int bias){
//     CudaParamIndividual last_sol = evolve->warm_start;

//     int sol_id = blockIdx.x;
//     int param_id = threadIdx.x;

//     if(sol_id >= size)  return;

//     float upper_bound = evolve->upper_bound[param_id];
//     float lower_bound = evolve->lower_bound[param_id];

//     if (param_id < evolve->problem_param.con_var_dims){
//         float rand_range = (upper_bound - lower_bound) * delta_con;

//         // based on rand_range update the boundary
//         upper_bound = min(upper_bound, last_sol.param[param_id] + rand_range);
//         lower_bound = max(lower_bound, last_sol.param[param_id] - rand_range);
        
//         // based on new boundary near parameter to generate the new parameter
//         new_param->all_param[(sol_id + bias) * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 110 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
//     }
//     else if(param_id < evolve->problem_param.int_var_dims){
//         float rand_range = (upper_bound - lower_bound) * delta_int;

//         // based on rand_range update the boundary
//         upper_bound = min(upper_bound, last_sol.param[param_id] + rand_range);
//         lower_bound = max(lower_bound, last_sol.param[param_id] - rand_range);
        
//         // based on new boundary near parameter to generate the new parameter
//         new_param->all_param[(sol_id + bias) * CUDA_PARAM_MAX_SIZE + param_id] = lower_bound + uniform_data[CUDA_SOLVER_POP_SIZE * 110 * CUDA_PARAM_MAX_SIZE + sol_id * CUDA_SOLVER_POP_SIZE + rand_idx + param_id] * (upper_bound - lower_bound);
//     }
// }

__global__ void LoadWarmStartResultForSolver(CudaEvolveData *evolve, CudaParamClusterData<CUDA_SOLVER_POP_SIZE> *new_param){
    ConvertCudaParam<CUDA_SOLVER_POP_SIZE>(new_param, &evolve->warm_start, blockIdx.x, threadIdx.x);
}

void CudaDiffEvolveSolver::WarmStart(){
    InitParameter<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, CUDA_SOLVER_POP_SIZE, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);
    printf("warm start\n");
    Evaluation(CUDA_SOLVER_POP_SIZE, 0);

//     SaveNewParamAsOldParam<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, CUDA_SOLVER_POP_SIZE, 0);
//     // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
//     int half_pop_size = CUDA_SOLVER_POP_SIZE >> 1;
//     int quad_pop_size = CUDA_SOLVER_POP_SIZE >> 2;
//     // setting half of population based on expected force and add some noise
//     GenerativeHeuristicsParam<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 32, 0.01, 0.01, quad_pop_size);

    

//     if(last_sol_fitness < 100){
//         // GenerativeRandSolNearBest<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, CUDA_SOLVER_POP_SIZE);
//         GenerativeRandomParamFromLastSol<<<half_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 48, 0.01, 0.01, half_pop_size, quad_pop_size);
//         // SaveNewParamAsOldParam<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, half_pop_size, half_pop_size+quad_pop_size, half_pop_size);
//         // if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("USING LAST POTENTIAL SOL\n");
//         // int half_pop_size = CUDA_SOLVER_POP_SIZE >> 1;
//         // int quad_pop_size = CUDA_SOLVER_POP_SIZE >> 2;
//         // // one cluster generate one solution, each cluster works on one block. 
//         // // We need to generate quad_pop_size new solutions based on last potential solution, so init the new cluster in quad_pop_size grid.
//         // UpdateClusterDataBasedEvolve<<<quad_pop_size, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, last_potential_sol_.len);
//     }
//     // UpdateVecParamBasedClusterData<CUDA_SOLVER_POP_SIZE><<<CUDA_SOLVER_POP_SIZE, 16, 0, cuda_utils_->streams_[0]>>>(new_cluster_vec_->data, new_cluster_data_);

//     // // int cet = 10;
//     // // Update the output param based on warm start.
//     // // CHECK_CUDA(cudaMemcpyAsync(output_sol, &new_cluster_vec_->data[cet], sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));

//     // // Evaluate random solutions or potential solutions in warmstart
//     Evaluation(CUDA_SOLVER_POP_SIZE, 0);

//     // // SortParamBasedBitonic<64><<<16, 64, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_->all_param, new_cluster_data_->fitness);

//     // Find the best solution among the random solutions or potential solutions in warmstart and put it in the first place
//     // ParaFindMax2<CUDA_SOLVER_POP_SIZE, CUDA_SOLVER_POP_SIZE><<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

//     // put heuristics individual in para_old[0, half_pop_size)
//     SaveNewParamAsOldParam<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, CUDA_SOLVER_POP_SIZE, 0);

//     // // based on warm start result to generate random solution. Further improve the quality of the initial population
//     // GenerativeRandSolNearBest<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.1, 0.1, CUDA_SOLVER_POP_SIZE);

//     // // convert the parameter from warm start to old parameter
//     // SaveNewParamAsOldParam<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, CUDA_SOLVER_POP_SIZE, 0);

//     // // Based on all old parameter to update the warm start of evolve data
//     // // 将 old_cluster_data_<CUDA_SOLVER_POP_SIZE*3> 中索引为0的数据提取出来,填充到evolve data单个CudaParamIndividual结构中,记为warm start。索引为0的解是warm start过程中最优的
//     // UpdateEvolveWarmStartBasedClusterData<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_);

//     if(DEBUG_PRINT_WARM_START_FLAG){
//         // CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
//         CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
//         CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
//         // PrintClusterData<CUDA_SOLVER_POP_SIZE*3>(host_old_cluster_data_);
//         PrintClusterData<CUDA_SOLVER_POP_SIZE>(host_new_cluster_data_);
//     }

//     // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
}

// (Abandoned) Use for loop to evaluate 
// template<int T>
// __global__ void MainEvaluation(CudaEvolveData *evolve, CudaParamClusterData<T> *cluster_data){
//     DynamicEvaluation2(evolve, cluster_data, evolve->lambda);
// }

auto max_dim3 = [](const dim3& a, const dim3& b) {
    return (a.x * a.y * a.z) > (b.x * b.y * b.z) ? a : b;
};

void CudaDiffEvolveSolver::Evaluation(int size, int epoch){
    printf("current epoch:%d\n",epoch);
    // CHECK_CUDA(cudaDeviceSynchronize());
    const size_t gemm_shared_mem_size = std::max(footstep::Ex_GEMM().shared_memory_size, footstep::Fu_GEMM().shared_memory_size);

    dim3 dim = max_dim3(footstep::Ex_GEMM().block_dim, footstep::Fu_GEMM().block_dim);
    printf("cublasDx need share memory size:%zu max block dim:%u\n",gemm_shared_mem_size, dim.x*dim.y*dim.z); 
    
    // // 验证是否超过设备限制
    cudaGetDevice(0);
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    
    if (gemm_shared_mem_size > prop.sharedMemPerBlock) {
        std::cout << "Required shared memory (" << gemm_shared_mem_size 
                  << " bytes) exceeds device limit (" << prop.sharedMemPerBlock 
                  << " bytes)" << std::endl;

        // Increase max dynamic shared memory for the kernel if needed
        CHECK_CUDA(cudaFuncSetAttribute(
            footstep::UpdateState<CUDA_SOLVER_POP_SIZE>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            gemm_shared_mem_size
        ));
    }
    
    // cudaGetDevice(0);  // 获取当前设备
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);  // 获取设备属性

    // std::cout << "==================== CUDA Device Info ====================" << std::endl;
    // std::cout << "Device Name                        : " << prop.name << std::endl;
    // std::cout << "Max Shared Memory per Block        : " << prop.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "Max Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    // std::cout << "Max Threads per Block              : " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Warp Size                           : " << prop.warpSize << std::endl;
    // std::cout << "=========================================================" << std::endl;
    CHECK_CUDA(cudaMemset(footstep::d_cluster_N_state, 0, sizeof(footstep::d_cluster_N_state)));

    footstep::UpdateState<CUDA_SOLVER_POP_SIZE><<<size, dim, gemm_shared_mem_size, cuda_utils_->streams_[0]>>>(new_cluster_data_, footstep::bigE, footstep::bigF, footstep::d_cluster_N_state);

    footstep::EvaluateModel<CUDA_SOLVER_POP_SIZE><<<size, 32, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, footstep::d_cluster_N_state, evaluate_score_);

    if(DEBUG_PRINT_FLAG || DEBUG_FOOTSTEP){
        CHECK_CUDA(cudaMemcpy(footstep::h_cluster_param, new_cluster_data_->all_param, CUDA_SOLVER_POP_SIZE * CUDA_PARAM_MAX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(footstep::h_cluster_N_state, footstep::d_cluster_N_state, footstep::N * CUDA_SOLVER_POP_SIZE * footstep::state_dims * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_evaluate_score_, evaluate_score_, CUDA_SOLVER_POP_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        PrintMatrixByRow(footstep::h_cluster_param, CUDA_SOLVER_POP_SIZE , CUDA_PARAM_MAX_SIZE, "h_cluster_param");
        PrintMatrixByRow(footstep::h_cluster_N_state, CUDA_SOLVER_POP_SIZE , footstep::N * footstep::state_dims, "cluster_N_state");
        PrintMatrixByRow(host_evaluate_score_, CUDA_SOLVER_POP_SIZE , 1, "evaluation score");
    }
    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    UpdateFitnessBasedMatrix<CUDA_SOLVER_POP_SIZE><<<1, size, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, evaluate_score_);
}

void CudaDiffEvolveSolver::Evolution(int epoch, CudaEvolveType search_type){
    DuplicateBestAndReorganize<<<CUDA_PARAM_MAX_SIZE, CUDA_SOLVER_POP_SIZE*3, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, 2);
    CudaEvolveProcess<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(epoch, old_cluster_data_, new_cluster_data_, random_center_->uniform_data_, random_center_->normal_data_, evolve_data_, CUDA_SOLVER_POP_SIZE, true);
    Evaluation(CUDA_SOLVER_POP_SIZE, epoch);

    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    UpdateParameter<CUDA_SOLVER_POP_SIZE><<<CUDA_PARAM_MAX_SIZE, CUDA_SOLVER_POP_SIZE*2, 0, cuda_utils_->streams_[0]>>>(epoch, evolve_data_, new_cluster_data_, old_cluster_data_);

    // CHECK_CUDA(cudaMemcpyAsync(h_terminate_flag, terminate_flag, sizeof(int), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
}

void CudaDiffEvolveSolver::InitSolver(int gpu_device){
    if(DEBUG_ENABLE_NVTX)   init_range = nvtxRangeStart("Init Different Evolution Solver");

    gpu_device_ = gpu_device;
    random_center_ =std::make_shared<CudaRandomManager>(gpu_device_);

    CHECK_CUDA(cudaSetDevice(gpu_device_));
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("CUDA SET DEVICE\n");

    dims_ = footstep::control_dims * footstep::N;

    con_var_dims_ = dims_;
    int_var_dims_ = 0;

    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle_);

    MallocSetup();

    footstep::ConstructEandF(cuda_utils_->streams_[0]);
    footstep::ConstructBigEAndF(footstep::bigE, footstep::bigF, cublas_handle_, cuda_utils_->streams_[0]);

    // CHECK_CUDA(cudaMemcpy(footstep::d_init_state, footstep::init_state, footstep::state_dims * sizeof(float), cudaMemcpyHostToDevice));

    // CHECK_CUDA(cudaMemset(footstep::d_cluster_N_state, 0, sizeof(footstep::d_cluster_N_state)));

    if (DEBUG_PRINT_FLAG || DEBUG_FOOTSTEP){
        printf("Debug flags enabled, copying memory\n");
        CHECK_CUDA(cudaMemcpyAsync(footstep::h_bigE, footstep::bigE, footstep::row_bigE * footstep::col_bigE * sizeof(float), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaMemcpyAsync(footstep::h_bigF, footstep::bigF, footstep::row_bigF * footstep::col_bigF * sizeof(float), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

        PrintMatrixByRow(footstep::h_bigE, footstep::row_bigE, footstep::col_bigE, "bigE:");
        PrintMatrixByRow(footstep::h_bigF, footstep::row_bigF, footstep::col_bigF, "bigF:");
    }

    InitDiffEvolveParam();
    
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("INIT PARAM FOR DE\n");
    
    // *h_terminate_flag = 0;
    // cudaMemset(terminate_flag, 0, sizeof(int));
    // float init_last_f = CUDA_MAX_FLOAT;
    // CHECK_CUDA(cudaMemcpy(last_fitness, &init_last_f, sizeof(float), cudaMemcpyHostToDevice));

    if(DEBUG_ENABLE_NVTX)   setting_boundary_range = nvtxRangeStart("Init_Solver Setting Boundary");

    host_evolve_data_->problem_param.con_var_dims = con_var_dims_;
    host_evolve_data_->problem_param.dims = dims_;
    host_evolve_data_->problem_param.int_var_dims = int_var_dims_;

    host_evolve_data_->problem_param.max_round = 60;

    host_evolve_data_->problem_param.accuracy_rng = 0.5;

    SetBoundary();
    
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("START MEMORY ASYNC\n");

    CHECK_CUDA(cudaMemcpyAsync(evolve_data_, host_evolve_data_, sizeof(CudaEvolveData), cudaMemcpyHostToDevice, cuda_utils_->streams_[0]));

    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_INIT_SOLVER_FLAG) printf("MEMORY ASYNC SUBMIT\n");

    InitCudaEvolveData<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, CUDA_SOLVER_POP_SIZE);

    // WarmStart();

    // CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));

    // size_t uniform_size = random_center_->uniform_size_;
    // std::vector<float> host_uniform(uniform_size);
    
    // // 拷贝所有数据到host
    // CHECK_CUDA(cudaMemcpy(host_uniform.data(), random_center_->uniform_data_, 
    //           uniform_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // printf("All uniform random numbers (total size: %zu):\n", uniform_size);
    // for(size_t i = 0; i < uniform_size; i++) {
    //     printf("uniform_data_[%zu] = %f\n", i, host_uniform[i]);
    //     // 验证数值是否合理
    //     if(host_uniform[i] < 0.0f || host_uniform[i] > 1.0f) {
    //         printf("WARNING: Invalid random number at %zu: %f\n", i, host_uniform[i]);
    //     }
    // }

    printf("FINISH INIT SOLVER\n");
}

// void CudaDiffEvolveSolver::UpdateCartPoleSystem(float sys_state[4], float wall_pos[2]){
//     float4 new_state = {sys_state[0], sys_state[1], sys_state[2], sys_state[3]};
//     float2 new_wall_pos = {wall_pos[0], wall_pos[1]};
    
//     CHECK_CUDA(cudaMemcpyToSymbol(cart_pole::current_state, &new_state, sizeof(float4)));
//     CHECK_CUDA(cudaMemcpyToSymbol(cart_pole::current_wall_pos, &new_wall_pos, sizeof(float2)));
//     cudaDeviceSynchronize();
// }


template <int T=CUDA_SOLVER_POP_SIZE*3>
__global__ void GetSolFromOldParam(CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3> *old_param, CudaParamIndividual *solution){
    ConvertCudaParamRevert<CUDA_SOLVER_POP_SIZE*3>(old_param, solution, blockIdx.x, threadIdx.x);
}

CudaParamIndividual CudaDiffEvolveSolver::Solver(){
    // nvtx3::mark("Different Evolvution Solver!");
    if(DEBUG_ENABLE_NVTX)   solver_range = nvtxRangeStart("Different Evolvution Solver");

    // init_pop_size_ = CUDA_SOLVER_POP_SIZE;
    // pop_size_ = CUDA_SOLVER_POP_SIZE;

    InitCudaEvolveData<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_, CUDA_SOLVER_POP_SIZE);

    InitParameter<<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, CUDA_SOLVER_POP_SIZE, new_cluster_data_, old_cluster_data_, random_center_->uniform_data_);
    
    // if(enable_warmstart)    LoadWarmStartResultForSolver<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_);

    // // based on warm start result to generate 
    // GenerativeRandSolNearBest<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, new_cluster_data_, random_center_->uniform_data_, 16, 0.001, 0.001, CUDA_SOLVER_POP_SIZE);
    
    Evaluation(CUDA_SOLVER_POP_SIZE, 0);

    ParaFindMax2<CUDA_SOLVER_POP_SIZE, CUDA_SOLVER_POP_SIZE><<<1, CUDA_SOLVER_POP_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_);

    SaveNewParamAsOldParam<<<CUDA_SOLVER_POP_SIZE, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(new_cluster_data_, old_cluster_data_, 0, CUDA_SOLVER_POP_SIZE, 0);
    
    // WarmStart();

    // float sol_obj_score = CUDA_MAX_FLOAT;
    // float sol_score = CUDA_MAX_FLOAT;
    // float *sol_state = nullptr;
    // bool satisify = false;
    for (int i = 0; i < host_evolve_data_->problem_param.max_round; ++i) {
        // printf("generation i:%d\n", i);
        Evolution(i, CudaEvolveType::GLOBAL);

        if(i == host_evolve_data_->problem_param.max_round - 1){
            printf("CHECK solution\n");
            const size_t gemm_shared_mem_size = std::max(footstep::Ex_GEMM().shared_memory_size, footstep::Fu_GEMM().shared_memory_size);

            dim3 dim = max_dim3(footstep::Ex_GEMM().block_dim, footstep::Fu_GEMM().block_dim);

            // // 验证是否超过设备限制
            cudaGetDevice(0);
            cudaDeviceProp prop;
            CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
            std::cout << "Device shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
            
            if (gemm_shared_mem_size > prop.sharedMemPerBlock) {
                std::cout << "Required shared memory (" << gemm_shared_mem_size 
                        << " bytes) exceeds device limit (" << prop.sharedMemPerBlock 
                        << " bytes)" << std::endl;

                // Increase max dynamic shared memory for the kernel if needed
                CHECK_CUDA(cudaFuncSetAttribute(
                    footstep::UpdateState<CUDA_SOLVER_POP_SIZE>, 
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    gemm_shared_mem_size
                ));
            }
            CHECK_CUDA(cudaMemset(footstep::d_sol_state, 0.0f, sizeof(footstep::d_sol_state)));
            footstep::UpdateState<CUDA_SOLVER_POP_SIZE*3><<<1, dim, gemm_shared_mem_size, cuda_utils_->streams_[0]>>>(old_cluster_data_, footstep::bigE, footstep::bigF, footstep::d_sol_state);
            footstep::EvaluateModel<CUDA_SOLVER_POP_SIZE*3><<<1, 32, 0, cuda_utils_->streams_[0]>>>(old_cluster_data_, footstep::d_sol_state, evaluate_score_, footstep::d_sol_score);

            CHECK_CUDA(cudaMemcpy(footstep::h_sol_state, footstep::d_sol_state, footstep::N * footstep::state_dims * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(footstep::h_sol_score, footstep::d_sol_score, 3 * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
            printf("sol_score:%f, sol_obj_score:%f, sol_constraint_score:%f\n",footstep::h_sol_score[0], footstep::h_sol_score[1], footstep::h_sol_score[2]);
        }
    }
    
    if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG){
        CHECK_CUDA(cudaMemcpyAsync(host_old_cluster_data_, old_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE*3>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaMemcpyAsync(host_new_cluster_data_, new_cluster_data_, sizeof(CudaParamClusterData<CUDA_SOLVER_POP_SIZE>), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
        PrintClusterData<CUDA_SOLVER_POP_SIZE*3>(host_old_cluster_data_);
        printf("new cluster data=============================================\n");
        PrintClusterData<CUDA_SOLVER_POP_SIZE>(host_new_cluster_data_);

        // CHECK_CUDA(cudaMemcpyAsync(host_evolve_data_, evolve_data_, sizeof(CudaEvolveData), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
        // printf("CUDA_MAX_FLOAT %f\n", CUDA_MAX_FLOAT);
    }
    // Get the first individual from old param (after sorting, the first one is the best one)
    GetSolFromOldParam<CUDA_SOLVER_POP_SIZE*3><<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(old_cluster_data_, result);
    // // 将 old_cluster_data_<CUDA_SOLVER_POP_SIZE*3> 中索引为0的数据提取出来,填充到evolve data单个CudaParamIndividual结构中,记为warm start。索引为0的解是warm start过程中最优的
    // UpdateEvolveWarmStartBasedClusterData<<<1, CUDA_PARAM_MAX_SIZE, 0, cuda_utils_->streams_[0]>>>(evolve_data_, old_cluster_data_);
    CHECK_CUDA(cudaMemcpyAsync(host_result, result, sizeof(CudaParamIndividual), cudaMemcpyDeviceToHost, cuda_utils_->streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(cuda_utils_->streams_[0]));
    // cudaDeviceSynchronize();

    host_result->objective_score = footstep::h_sol_score[1];
    host_result->constraint_score = footstep::h_sol_score[2];
    host_result->N_states = footstep::h_sol_state;
    

    // for(int i = con_var_dims_; i < dims_; ++i){
    //     host_result->param[i] = floor(host_result->param[i]);
    // }

    // if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG)   printFinalResult(host_result->fitness, host_result->param, dims_);
    // printFinalResult(host_result->fitness, host_result->param, dims_);

    if(DEBUG_ENABLE_NVTX)   nvtxRangeEnd(solver_range);

    return *host_result;
}

CudaDiffEvolveSolver::~CudaDiffEvolveSolver(){
    if (cudamalloc_flag){
        // GPU device
        CHECK_CUDA(cudaFree(evolve_data_));
        CHECK_CUDA(cudaFree(new_cluster_data_));
        CHECK_CUDA(cudaFree(old_cluster_data_));
        // CHECK_CUDA(cudaFree(new_cluster_vec_));
        CHECK_CUDA(cudaFree(constraint_matrix));
        CHECK_CUDA(cudaFree(objective_matrix));
        CHECK_CUDA(cudaFree(param_matrix));
        CHECK_CUDA(cudaFree(evaluate_score_));
        CHECK_CUDA(cudaFree(constraint_score));
        CHECK_CUDA(cudaFree(quad_matrix));
        CHECK_CUDA(cudaFree(quad_transform));
        CHECK_CUDA(cudaFree(quadratic_score));
        CHECK_CUDA(cudaFree(lambda_matrix));
        CHECK_CUDA(cudaFree(result));

        // CPU host
        if (DEBUG_PRINT_FLAG || DEBUG_PRINT_SOLVER_FLAG){
            CHECK_CUDA(cudaFreeHost(host_new_cluster_data_));
            CHECK_CUDA(cudaFreeHost(host_old_cluster_data_));
        }

        if (DEBUG_PRINT_FLAG || DEBUG_PRINT_EVALUATE_FLAG){
            CHECK_CUDA(cudaFreeHost(host_evaluate_score_));
            CHECK_CUDA(cudaFreeHost(host_param_matrix));
            CHECK_CUDA(cudaFreeHost(host_constraint_score));
            CHECK_CUDA(cudaFreeHost(h_lambda_matrix));
            CHECK_CUDA(cudaFreeHost(h_constraint_matrix));
            CHECK_CUDA(cudaFreeHost(h_objective_matrix));
            CHECK_CUDA(cudaFreeHost(host_quad_matrix));
            CHECK_CUDA(cudaFreeHost(h_quad_transform));
            CHECK_CUDA(cudaFreeHost(h_quadratic_score));
        }
        
        CHECK_CUDA(cudaFreeHost(host_evolve_data_));
        CHECK_CUDA(cudaFreeHost(host_result));
        
    }
}
}