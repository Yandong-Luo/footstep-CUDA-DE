#ifndef CUDA_BEZIER_CURVE_H
#define CUDA_BEZIER_CURVE_H

#include "utils/config.h"
#include "footstep/footstep_utils.cuh"

namespace cudaprocess {
namespace bezier_curve {

typedef CudaVector<float, BEZIER_SIZE> CudaBezierParam;

struct BezierCurve {
    float binomial_coeff_[BEZIER_SIZE],
          binomial_deriv_coeff_[BEZIER_SIZE-1],
          bernstein_weights_[CURVE_NUM_STEPS][BEZIER_SIZE + 1],
          bernstein_deriv_weights_[CURVE_NUM_STEPS][BEZIER_SIZE];
	float2 control_points[BEZIER_SIZE];
    float fixed_point_idx[NUM_FIXED_CP];
};

// ---- Declaration of utility functions ----
__device__ __forceinline__ float2 PolarToCartesian(float radius, float theta);
__device__ __forceinline__ float2 PolarVelocityToCartesian(float2 position, float2 velocity);
__device__ __forceinline__ void PolarAllState2Cartesian(float2 position, float2 velocity, float *state, int step_idx);

// ---- Declaration of Bezier functions ----
__global__ void PrepareBinomialandFixedPoint(BezierCurve* curve);
__device__ __forceinline__ float GetBezierAt(BezierCurve *curve, float *params, int step_idx, int l, int r);
__device__ __forceinline__ float2 GetBezierPosition(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float2 GetBezierPositionVelocity(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ void GetTrajStateFromBezier(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, float *state, bool convert = false);
// __device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state, bool convert = false);
__device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state, bool convert = false){

    float2 position{0.0f, 0.0f};
	float2 velocity{0.0f, 0.0f};
	const int n = BEZIER_SIZE - 1;

    for (int i = 0; i < BEZIER_SIZE; ++i) {
        float bernstein_t = curve->bernstein_weights_[t][i];
		position.x += bernstein_t * params[i + l];
		position.y += bernstein_t * params[i + ll];
		
		if(i < n){
			float deriv_ctrl_x = n * (params[i + 1 + l] - params[i + l]);
			float deriv_ctrl_y = n * (params[i + 1 + ll] - params[i + ll]);

			float bernstein_deriv_t = curve->bernstein_deriv_weights_[t][i];

			velocity.x += bernstein_deriv_t * deriv_ctrl_x;
        	velocity.y += bernstein_deriv_t * deriv_ctrl_y;
		}
	}

    if(!convert){
		// 存储状态
		// int idx = t * footstep::state_dims;
		// state[idx] = position.x;
		// state[idx + 1] = position.y;
		// state[idx + 2] = velocity.x;
		// state[idx + 3] = velocity.y;
		// state[idx + 4] = atan2f(position.y, position.x);		// theta

        state[0] = position.x;
        state[1] = position.y;
        state[2] = velocity.x;
        state[3] = velocity.y;
        state[4] = atan2f(position.y, position.x);  // theta

		// state[idx + 5] = 0.0;		// radius for polar coordinate system
        // printf("Writing state at t=%d to indices: %d through %d\n", 
        //     t, t*footstep::state_dims, t*footstep::state_dims+4);
	}
    else{
		PolarAllState2Cartesian(position, velocity, state, t);
	}
}
__device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float2 GetBezierPositionBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float GetBezierAtBasedLookup(BezierCurve *curve, float *params, int t, int l, int r);
__device__ __forceinline__ void GetVec(BezierCurve *curve, CudaBezierParam *out, float t, bool convert = false);

// ---- BezierCurveManager 声明 ----
struct BezierCurveManager {
    BezierCurve *curve_;
    BezierCurveManager();
    ~BezierCurveManager();
};

} // namespace bezier_curve
} // namespace cudaprocess

#endif
