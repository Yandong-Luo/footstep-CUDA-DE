#ifndef CUDA_ARCLENGTH_PARAM_H
#define CUDA_ARCLENGTH_PARAM_H

#include "curve/bezier_curve.cuh"
#include "curve/numerical_integration.cuh"
#include "utils/config.h"

namespace cudaprocess {
namespace arclength {

extern float *d_AllTraj_ArcLen;
extern float *d_AllTraj_TotalLen;

extern float *h_AllTraj_ArcLen;
extern float *h_AllTraj_TotalLen;

// // Structure to be passed to the integration function
// struct BezierData {
//     bezier_curve::BezierCurve* bezier_curve;
//     float* curve_param;
// };

// // Class for arc-length parameterization of Bezier curves
// template <int N>
// class BezierArcLengthManager {
//     public:
//     bezier_curve::BezierCurve *bezier_curve_;
//     float *curve_param_;     // record all the control point of bezier curve
//     float *arcLengths_;
//     BezierArcLengthManager();
//     ~BezierArcLengthManager();

//     private:

// };

__device__ static float derivativeNormFunction(float t, const bezier_curve::BezierCurve *bezier_curve, const float *curve_param) {    
    // int step = static_cast<int>(t * ARC_LENGTH_SAMPLES);
    
    // // Calculate the derivatives in the x and y directions
    // float2 derivative = bezier_curve::GetBezierPositionVelocityBasedLookup(
    //     bezier_curve, curve_param, step, 0, BEZIER_SIZE - 1, 
    //     BEZIER_SIZE, 2*BEZIER_SIZE-1);

    float2 derivative = bezier_curve::GetBezierPositionVelocity(bezier_curve, curve_param, t, 0, BEZIER_SIZE - 1, 
            BEZIER_SIZE, 2*BEZIER_SIZE-1);
    // if(threadIdx.x == 0)    printf("vx %f, vy %f\n", derivative.x, derivative.y);
    // Returns the length of the tangent line
    return sqrtf(derivative.x * derivative.x + derivative.y * derivative.y);
}

__device__ __forceinline__ float calculateLength(float start, float end, const bezier_curve::BezierCurve* bezier_curve, const float *curve_param){
    // use adaptive Simpson integre to calculate the length of xy trajectory
    // return integration::adaptive_simpson_3_8_device(derivativeNormFunction, start, end, bezier_curve, curve_param);
    return integration::iterative_adaptive_simpson(derivativeNormFunction, start, end, bezier_curve, curve_param);
}

// Calculate the total arc-length of the curve
// 1 block, each thread for one trajectory total len
__global__ void calculateTotalLength(const bezier_curve::BezierCurve* bezier_curve, 
    float *all_curve_param, float *AllTraj_TotalLen,
    int curve_param_size = 3*BEZIER_SIZE, 
    int num_traj = CUDA_SOLVER_POP_SIZE);

// each thread for t, each block for one trajectory
__global__ void initAllTrajArcLengthMap(const bezier_curve::BezierCurve* bezier_curve, 
    float *all_curve_param, float *AllTraj_ArcLen, 
    float *AllTraj_TotalLen = nullptr,
    int curve_param_size = 3*BEZIER_SIZE,
    int num_traj = CUDA_SOLVER_POP_SIZE);

}
}

#endif
