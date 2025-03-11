#ifndef CUDA_BEZIER_CURVE_H
#define CUDA_BEZIER_CURVE_H

#include "utils/config.h"
#include "footstep/footstep_utils.cuh"

namespace cudaprocess {
namespace bezier_curve {

typedef CudaVector<float, BEZIER_SIZE> CudaBezierParam;

extern float *d_all_curve_param;
extern float *h_all_curve_param;

struct BezierCurve {
    float binomial_coeff_[BEZIER_SIZE],
          binomial_deriv_coeff_[BEZIER_SIZE-1],
          bernstein_weights_[CURVE_NUM_STEPS][BEZIER_SIZE + 1],
          bernstein_deriv_weights_[CURVE_NUM_STEPS][BEZIER_SIZE];
	float3 control_points[BEZIER_SIZE];
    bool is_point_xy_fixed[BEZIER_SIZE] = {0};
	bool is_theta_point_fixed[BEZIER_SIZE] = {0};
};

// ---- Declaration of utility functions ----
// __device__ __forceinline__ float2 PolarToCartesian(float radius, float theta);
// __device__ __forceinline__ float2 PolarVelocityToCartesian(float2 position, float2 velocity);
// __device__ __forceinline__ void PolarAllState2Cartesian(float2 position, float2 velocity, float *state, int step_idx);

// ---- Declaration of Bezier functions ----
__global__ void PrepareBinomialandFixedPoint(BezierCurve* curve);

// Get the position of bezier curve when timestep = t (1 means the final step, 0 means the start)
__device__ __forceinline__ float GetBezierAt(BezierCurve *curve, float *params, int step_idx, int l, int r) {
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
	t_powers[0] = one_minus_t_powers[0] = 1;

	for (int i{1}; i < BEZIER_SIZE; ++i) {
		t_powers[i] = t_powers[i - 1] * t;
		one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1 - t);
	}
	float res{0.};

	for (int i = 0; i < BEZIER_SIZE; ++i) {
		res += curve->binomial_coeff_[i] * (t_powers[i] * one_minus_t_powers[BEZIER_SIZE - 1 - i] * params[i + l]);
	}

	return res;
}


// Get the x and y position of bezier curve when timestep = t (1 means the final step, 0 means the start)
__device__ __forceinline__ float2 GetBezierPosition(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14) {
	//   printf("bbbbb! %d %d %d %d", l, r, ll, rr);
	//   // for(;;);
	// }
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
	t_powers[0] = one_minus_t_powers[0] = 1;

	for (int i{1}; i < BEZIER_SIZE; ++i) {
		t_powers[i] = t_powers[i - 1] * t;
		one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1 - t);
	}

	float2 ret{0., 0.};

	for (int i = 0; i < BEZIER_SIZE; ++i) {
		float bernstein_t = curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[BEZIER_SIZE - 1 - i];		// B(t) in formula
		ret.x += bernstein_t * params[i + l];
		ret.y += bernstein_t * params[i + ll];
	}
	return ret;
}


// __device__ __forceinline__ float2 GetBezierPositionVelocity(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float2 GetBezierPositionVelocity(const BezierCurve *curve, const float *params, float t, int l, int r, int ll, int rr) {
	// 计算实际的 t 值 (0 到 1 之间)
    // float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
    
    float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
    t_powers[0] = one_minus_t_powers[0] = 1;
    const int n = BEZIER_SIZE - 1;

    // 计算t幂
    for (int i = 1; i < BEZIER_SIZE; ++i) {
        t_powers[i] = t_powers[i-1] * t;
        one_minus_t_powers[i] = one_minus_t_powers[i-1] * (1 - t);
    }

    // 计算位置
    float2 position{0.0f, 0.0f};
	float2 velocity{0.0f, 0.0f};
    for (int i = 0; i < BEZIER_SIZE; ++i) {
        float bernstein_t = curve->binomial_coeff_[i] * 
                           t_powers[i] * 
                           one_minus_t_powers[BEZIER_SIZE - 1 - i];
        position.x += bernstein_t * params[i + l];
        position.y += bernstein_t * params[i + ll];
        
		// for velocity
		if(i < n){
			float deriv_ctrl_x = n * (params[i + 1 + l] - params[i + l]);
			float deriv_ctrl_y = n * (params[i + 1 + ll] - params[i + ll]);

			float bernstein_deriv_t = curve->binomial_deriv_coeff_[i] * 
							t_powers[i] * 
							one_minus_t_powers[n - 1 - i];

			velocity.x += bernstein_deriv_t * deriv_ctrl_x;
			velocity.y += bernstein_deriv_t * deriv_ctrl_y;
		}
    }
	return velocity; 
}

// __device__ __forceinline__ void GetTrajStateFromBezier(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, float *state, bool convert = false);
__device__ __forceinline__ void GetTrajStateFromBezier(const BezierCurve *curve, const float *params, float t, int l, int r, int ll, int rr, int lll, int rrr, float *state){
    // // 计算实际的 t 值 (0 到 1 之间)
    // float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
    
    float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
    t_powers[0] = one_minus_t_powers[0] = 1;
    // const int n = BEZIER_SIZE - 1;

    // 计算t幂
    for (int i = 1; i < BEZIER_SIZE; ++i) {
        t_powers[i] = t_powers[i-1] * t;
        one_minus_t_powers[i] = one_minus_t_powers[i-1] * (1 - t);

		// printf("current step:%d j:%d t_powers:%f\n", threadIdx.x, i, t_powers[i]);
		// printf("current step:%d j:%d one_minus_t_powers:%f\n", threadIdx.x, i, one_minus_t_powers[i]);
    }

    // 计算位置
    float2 position{0.0f, 0.0f};
	// float2 velocity{0.0f, 0.0f};
	float theta = 0.0f;
    for (int i = 0; i < BEZIER_SIZE; ++i) {
        float bernstein_t = curve->binomial_coeff_[i] * 
                           t_powers[i] * 
                           one_minus_t_powers[BEZIER_SIZE - 1 - i];
        position.x += bernstein_t * params[i + l];
        position.y += bernstein_t * params[i + ll];
		theta += bernstein_t * params[i + lll];
        
		// for velocity
		// if(i < n){
		// 	float deriv_ctrl_x = n * (params[i + 1 + l] - params[i + l]);
		// 	float deriv_ctrl_y = n * (params[i + 1 + ll] - params[i + ll]);

		// 	float bernstein_deriv_t = curve->binomial_deriv_coeff_[i] * 
		// 					t_powers[i] * 
		// 					one_minus_t_powers[n - 1 - i];

		// 	velocity.x += bernstein_deriv_t * deriv_ctrl_x;
		// 	velocity.y += bernstein_deriv_t * deriv_ctrl_y;
		// }
    }

	// 存储状态
	// int idx = t * footstep::state_dims;
	// printf("x:%f y:%f theta:%f\n", position.x, position.y, theta);
	state[0] = position.x;
	state[1] = position.y;
	state[2] = 0.0f;
	state[3] = 0.0f;
	state[4] = theta;		// theta
}

// __device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state, bool convert = false);
__device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, int lll, int rrr, float *state){

    float2 position{0.0f, 0.0f};
	float2 velocity{0.0f, 0.0f};
	float theta = 0.0f;
	const int n = BEZIER_SIZE - 1;

    for (int i = 0; i < BEZIER_SIZE; ++i) {
        float bernstein_t = curve->bernstein_weights_[t][i];
		position.x += bernstein_t * params[i + l];
		position.y += bernstein_t * params[i + ll];
		theta += bernstein_t * params[i + lll];
		
		if(i < n){
			float deriv_ctrl_x = n * (params[i + 1 + l] - params[i + l]);
			float deriv_ctrl_y = n * (params[i + 1 + ll] - params[i + ll]);

			float bernstein_deriv_t = curve->bernstein_deriv_weights_[t][i];

			velocity.x += bernstein_deriv_t * deriv_ctrl_x;
        	velocity.y += bernstein_deriv_t * deriv_ctrl_y;
		}
	}

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
	state[4] = theta;  // theta

	// state[idx + 5] = 0.0;		// radius for polar coordinate system
	// printf("Writing state at t=%d to indices: %d through %d\n", 
	//     t, t*footstep::state_dims, t*footstep::state_dims+4);
}
// __device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(const BezierCurve *curve, const float *params, int t, int l, int r, int ll, int rr) {
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
	return velocity;
}

// __device__ __forceinline__ float2 GetBezierPositionBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false);
__device__ __forceinline__ float3 GetBezierPositionBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, int lll, int rrr) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14 || (x > 50)) {
	//   printf("ccccc! %d %d %d %d %d", l, r, ll, rr, x);
	//   // for(;;);
	// }
	float3 ret{0.0f, 0.0f, 0.0f};
	for (int i = 0; i < BEZIER_SIZE; ++i) {
		float bernstein_t = curve->bernstein_weights_[t][i];
		ret.x += bernstein_t * params[i + l];
		ret.y += bernstein_t * params[i + ll];
	}
	return ret;
}

// __device__ __forceinline__ float GetBezierAtBasedLookup(BezierCurve *curve, float *params, int t, int l, int r);
__device__ __forceinline__ float GetBezierAtBasedLookup(const BezierCurve *curve, const float *params, int t, int l, int r) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14 || (x > 50)) {
	//   printf("ccccc! %d %d %d %d %d", l, r, ll, rr, x);
	//   // for(;;);
	// }
	float ret = 0;
	for (int i = 0; i < BEZIER_SIZE; ++i) {
		float bernstein_t = curve->bernstein_weights_[t][i];
		ret += bernstein_t * params[i + l];
	}
	return ret;
}

// __device__ __forceinline__ void GetVec(BezierCurve *curve, CudaBezierParam *out, float t, bool convert = false);
__device__ __forceinline__ void GetVec(BezierCurve *curve, CudaBezierParam *out, float t, bool convert) {
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
	t_powers[0] = one_minus_t_powers[0] = 1;
	for (int i{1}; i < BEZIER_SIZE; ++i) {
		t_powers[i] = t_powers[i - 1] * t;
		one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1.f - t);
	}
	for (int i = 0; i < BEZIER_SIZE; ++i) {
		out->data[i] = curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[BEZIER_SIZE - 1 - i];
	}
}

// ---- BezierCurveManager 声明 ----
struct BezierCurveManager {
    BezierCurve *curve_;
    BezierCurveManager();
    ~BezierCurveManager();
};

} // namespace bezier_curve
} // namespace cudaprocess

#endif
