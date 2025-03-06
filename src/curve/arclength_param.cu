#include "curve/arclength_param.cuh"

namespace cudaprocess{

namespace arclength{

float *d_AllTraj_ArcLen = nullptr;
float *d_AllTraj_TotalLen = nullptr;

float *h_AllTraj_ArcLen = nullptr;
float *h_AllTraj_TotalLen = nullptr;

// each thread for t, each block for one trajectory
__global__ void initAllTrajArcLengthMap(const bezier_curve::BezierCurve* bezier_curve, 
                                        float *all_curve_param, float *AllTraj_ArcLen, 
                                        float *AllTraj_TotalLen,
                                        int curve_param_size, int num_traj){

    if(threadIdx.x >= ARC_LENGTH_SAMPLES || blockIdx.x >= num_traj)   return;
    float *traj_arcLen = AllTraj_ArcLen + blockIdx.x * (ARC_LENGTH_SAMPLES + 1);
    const float *cur_param = all_curve_param + blockIdx.x * curve_param_size;
    int i = threadIdx.x + 1;
    float t = i / static_cast<float>(ARC_LENGTH_SAMPLES);
    traj_arcLen[i] = calculateLength(0.0, t, bezier_curve, cur_param);

    if(AllTraj_TotalLen != nullptr && i == ARC_LENGTH_SAMPLES)  AllTraj_TotalLen[blockIdx.x] = traj_arcLen[i];
}

// Calculate the total arc-length of the curve
// 1 block, each thread for one trajectory total len
__global__ void calculateTotalLength(const bezier_curve::BezierCurve* bezier_curve, 
                                    float *all_curve_param, float *AllTraj_TotalLen,
                                    int curve_param_size, int num_traj){
    if(threadIdx.x >= num_traj || blockIdx.x >= 1)   return;
    const float *cur_param = all_curve_param + threadIdx.x * curve_param_size;
    if(threadIdx.x == 0){
        printf("curve param\n");
        for(int i = 0; i < curve_param_size;++i){
            printf("%f ", cur_param[i]);
        }
        printf("\n");
    }
    AllTraj_TotalLen[threadIdx.x] = calculateLength(0.0, 1.0, bezier_curve, cur_param);
}

}
}
