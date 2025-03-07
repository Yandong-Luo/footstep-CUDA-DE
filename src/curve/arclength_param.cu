#include "curve/arclength_param.cuh"

namespace cudaprocess{

namespace arclength{

float *d_allTraj_arcLenTable = nullptr;
float *d_allTraj_totalLen = nullptr;

float *h_allTraj_arcLenTable = nullptr;
float *h_allTraj_totalLen = nullptr;

// Calculate the total arc-length of the curve
// 1 block, each thread for one trajectory total len
__global__ void calculateTotalLength(const bezier_curve::BezierCurve* bezier_curve, 
        float *all_curve_param, float *allTraj_totalLen,
        int curve_param_size, int num_traj){
    if(threadIdx.x >= num_traj || blockIdx.x >= 1)   return;
    
    const float *cur_param = all_curve_param + threadIdx.x * curve_param_size;
    
    // if(threadIdx.x == 0){
    //     printf("curve param\n");
    //     for(int i = 0; i < curve_param_size;++i){
    //         printf("%f ", cur_param[i]);
    //     }
    //     printf("\n");
    // }

    allTraj_totalLen[threadIdx.x] = calculateLength(0.0, 1.0, bezier_curve, cur_param);
}

// each thread for t, each block for one trajectory
__global__ void initAllTrajArcLengthMap(const bezier_curve::BezierCurve* bezier_curve, 
                                        float *all_curve_param, float *allTraj_arcLenTable, 
                                        float *allTraj_totalLen,
                                        int curve_param_size, int num_traj){
    if(threadIdx.x >= ARC_LENGTH_SAMPLES || blockIdx.x >= num_traj)   return;
    float *traj_arcLen = allTraj_arcLenTable + blockIdx.x * (ARC_LENGTH_SAMPLES + 1);
    const float *cur_param = all_curve_param + blockIdx.x * curve_param_size;
    int i = threadIdx.x + 1;
    float t = i / static_cast<float>(ARC_LENGTH_SAMPLES);
    traj_arcLen[i] = calculateLength(0.0, t, bezier_curve, cur_param);

    if(allTraj_totalLen != nullptr && i == ARC_LENGTH_SAMPLES)  allTraj_totalLen[blockIdx.x] = traj_arcLen[i];
}

// each thread for one trajectory, each block for one timestep
__global__ void DecodeStateBasedArcLength(const bezier_curve::BezierCurve* bezier_curve,
                                          float* all_curve_param,
                                          float* cluster_state,
                                          float* allTraj_arcLengthTable,
                                          float* allTraj_arcTotalLength,
                                          const int arcLengthTableSize,
                                          const int curve_param_size){
    if(blockIdx.x >= CURVE_NUM_STEPS || threadIdx.x >= CUDA_SOLVER_POP_SIZE)  return;
    int step_idx = blockIdx.x, traj_idx = threadIdx.x;
    
    // get current traj total length and arcLength Table
    const float arcTotalLength = allTraj_arcTotalLength[traj_idx];
    const float *traj_arcLengthTable = allTraj_arcLengthTable + traj_idx * arcLengthTableSize;
    const float *curve_param = all_curve_param + traj_idx * curve_param_size;

    float t = blockIdx.x / static_cast<float>(footstep::N);
    
    // Arc length parameter
    float s = step_idx * arcTotalLength / static_cast<float>(footstep::N);

    float t_arc = getParameterForArcLengthNewton(bezier_curve, curve_param, traj_arcLengthTable, s, arcTotalLength);

    // if(threadIdx.x == 0){
    //     printf("block:%d t_arc:%f\n", blockIdx.x, t_arc);
    // }
    float *current_state = cluster_state + traj_idx * footstep::state_dims * CURVE_NUM_STEPS + step_idx * footstep::state_dims;
    bezier_curve::GetTrajStateFromBezier(bezier_curve, curve_param, t, 0, BEZIER_SIZE-1, BEZIER_SIZE, 2*BEZIER_SIZE-1, 2*BEZIER_SIZE, 3*BEZIER_SIZE-1, current_state);
}

}
}
