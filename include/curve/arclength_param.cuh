#ifndef CUDA_ARCLENGTH_PARAM_H
#define CUDA_ARCLENGTH_PARAM_H

#include "curve/bezier_curve.cuh"
#include "curve/numerical_integration.cuh"
#include "utils/config.h"

namespace cudaprocess {
namespace arclength {

extern float *d_allTraj_arcLenTable;
extern float *d_allTraj_totalLen;

extern float *h_allTraj_arcLenTable;
extern float *h_allTraj_totalLen;

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

// Get parameter t for a given arc-length s
__device__ __forceinline__ float getParameterForArcLength(const float* arcLengthTable,
                                                        const float s,
                                                        const float totalLength,
                                                        const int tableSize = (ARC_LENGTH_SAMPLES + 1)) {
    if (s <= 0.0f) return 0.0f;
    if (s >= totalLength) return 1.0f;
    
    // Binary search to find the closest pre-computed point
    int left = 0;
    int right = tableSize - 1;
    
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (arcLengthTable[mid] < s) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    // Linear interpolation between the closest points
    float t0 = static_cast<float>(left) / (tableSize - 1);
    float t1 = static_cast<float>(right) / (tableSize - 1);
    float s0 = arcLengthTable[left];
    float s1 = arcLengthTable[right];
    
    // Avoid division by zero
    if (fabsf(s1 - s0) < 1e-6f) {
        return t0;
    }
    
    return t0 + (s - s0) * (t1 - t0) / (s1 - s0);
}


// More accurate arc-length parameter calculation using Newton's method
__device__ __forceinline__ float getParameterForArcLengthNewton(const bezier_curve::BezierCurve* bezier_curve,
                                                const float* curve_param,
                                                const float* arcLengthTable,
                                                const float s, 
                                                const float totalLength,
                                                const int tableSize = (ARC_LENGTH_SAMPLES + 1),
                                                const float tol = 1e-6f, 
                                                const int maxIter = 20){
    if (s <= 0.0f) return 0.0f;
    if (s >= totalLength) return 1.0f;
    
    // Initialize with linear interpolation
    float t = getParameterForArcLength(arcLengthTable, s, totalLength);
    
    // Newton's method iteration
    for (int iter = 0; iter < maxIter; iter++) {
        float currentLength = calculateLength(0.0f, t, bezier_curve, curve_param);
        float error = currentLength - s;
        
        if (fabsf(error) < tol) {
            break;
        }
        
        // Calculate first derivative (speed at parameter t)
        float2 derivative = bezier_curve::GetBezierPositionVelocity(
            bezier_curve, curve_param, t, 0, BEZIER_SIZE - 1, 
            BEZIER_SIZE, 2*BEZIER_SIZE-1);
        float speed = sqrtf(derivative.x * derivative.x + derivative.y * derivative.y);
        
        // Avoid division by zero
        if (speed < 1e-8f) {
            break;
        }
        
        // Newton update: t = t - f(t)/f'(t)
        t = t - error / speed;
        
        // Ensure t stays in valid range [0,1]
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    
    return t;
}

// Calculate the total arc-length of the curve
// 1 block, each thread for one trajectory total len
__global__ void calculateTotalLength(const bezier_curve::BezierCurve* bezier_curve, 
    float *all_curve_param, float *allTraj_totalLen,
    int curve_param_size = 3*BEZIER_SIZE, 
    int num_traj = CUDA_SOLVER_POP_SIZE);

// each thread for t, each block for one trajectory
__global__ void initAllTrajArcLengthMap(const bezier_curve::BezierCurve* bezier_curve, 
                                        float *all_curve_param, float *allTraj_arcLenTable, 
                                        float *allTraj_totalLen = nullptr,
                                        int curve_param_size = 3*BEZIER_SIZE,
                                        int num_traj = CUDA_SOLVER_POP_SIZE);


// each thread for one trajectory, each block for one timestep
__global__ void DecodeStateBasedArcLength(const bezier_curve::BezierCurve* bezier_curve,
                                        float* all_curve_param,
                                        float* cluster_state,
                                        float* allTraj_arcLengthTable,
                                        float* allTraj_arcTotalLength,
                                        const int arcLengthTableSize = (ARC_LENGTH_SAMPLES + 1),
                                        const int curve_param_size = 3*BEZIER_SIZE);

}
}

#endif
