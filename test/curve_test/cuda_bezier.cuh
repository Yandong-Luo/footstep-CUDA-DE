#ifndef CUDA_BEZIER_CURVE_H
#define CUDA_BEZIER_CURVE_H

// #include "planner/trajectory_generator/cudapcr/base.cuh"
// #include "planner/trajectory_generator/cudapcr/types.h"
// #include "utils/config.h"
// #include "footstep/footstep_utils.cuh"

#define BEZIER_SIZE 7
#define NUM_STEPS 30

namespace cudaprocess{

namespace bezier_curve{

// typedef CudaVector<float, BEZIER_SIZE> CudaBezierParam;

struct BezierCurve {
	// combination coefficient C_n^i, binomial_deriv_coeff_ C_n-1^i; bernstein_weights_ C_n^i * t^j * (1-t)^{n-j}, bernstein_deriv_weights_
	float binomial_coeff_[BEZIER_SIZE], binomial_deriv_coeff_[BEZIER_SIZE-1], bernstein_weights_[NUM_STEPS + 1][BEZIER_SIZE + 1], bernstein_deriv_weights_[NUM_STEPS + 1][BEZIER_SIZE];
};

// *************** Polar to Cartesian *****************
__device__ __forceinline__ float2 PolarToCartesian(float radius, float theta) {
	return make_float2(radius * __cosf(theta), radius * __sinf(theta));
}

__device__ __forceinline__ float2 PolarVelocityToCartesian(float2 position, float2 velocity) {
	float radius = position.x, theta = position.y;
	float v_r = velocity.x, v_theta = velocity.y;
	float Vx = v_r * __cosf(theta) - radius * v_theta * __sinf(theta);
	float Vy = v_r * __sinf(theta) + radius * v_theta * __cosf(theta);
	return make_float2(Vx, Vy);
}

// __device__ __forceinline__ void GetVec(BezierCurve *curve, CudaBezierParam *out, float t, bool convert = false) {
// 	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
// 	t_powers[0] = one_minus_t_powers[0] = 1;
// 	for (int i{1}; i < BEZIER_SIZE; ++i) {
// 		t_powers[i] = t_powers[i - 1] * t;
// 		one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1.f - t);
// 	}
// 	for (int i = 0; i < BEZIER_SIZE; ++i) {
// 		out->data[i] = curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[BEZIER_SIZE - 1 - i];
// 	}
// }

__device__ __forceinline__ void PolarAllState2Cartesian(float2 position, float2 velocity, float *state, int step_idx){
    float radius = position.x, theta = position.y;
    float v_r = velocity.x, v_theta = velocity.y;
	int idx = step_idx * 6;
	state[idx] = radius * __cosf(theta);
    state[idx + 1] = radius * __sinf(theta);
    state[idx + 2] = v_r * __cosf(theta) - radius * v_theta * __sinf(theta);
    state[idx + 3] = v_r * __sinf(theta) + radius * v_theta * __cosf(theta);
    state[idx + 4] = theta;
	state[idx + 5] = radius;
}

__global__ void PrepareBinomial(BezierCurve* curve) {
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];	// record t^j, (1-t)^{n-j}

	float tmp_binomial_coeff[BEZIER_SIZE][BEZIER_SIZE + 1];
	
	tmp_binomial_coeff[0][0] = tmp_binomial_coeff[1][1] = 1;
	tmp_binomial_coeff[0][1] = 0;
	tmp_binomial_coeff[1][0] = 1;

	// Pascal’s Triangle to calculate combination Coefficients
	for (int i{2}; i < BEZIER_SIZE; ++i) {
		tmp_binomial_coeff[i][0] = tmp_binomial_coeff[i][i] = 1;
		for (int j{1}; j < i; ++j) {
			tmp_binomial_coeff[i][j] = tmp_binomial_coeff[i - 1][j - 1] + tmp_binomial_coeff[i - 1][j];		// C_j^i
		}
	}
	// save Binomial Coefficients to comb
	for (int i{0}; i < BEZIER_SIZE; ++i) {
		curve->binomial_coeff_[i] = tmp_binomial_coeff[BEZIER_SIZE - 1][i];
		// printf("combination coefficients for position:%d %f\n", i, curve->binomial_coeff_[i]);
	}

	const int num_steps = NUM_STEPS;
    const int n = BEZIER_SIZE - 1;  // 贝塞尔曲线的阶数
    
    // 预计算低一阶的二项式系数(用于导数计算)
    float lower_binomial_coeff[BEZIER_SIZE - 1];  // n-1阶贝塞尔曲线的二项式系数
    for (int i = 0; i < BEZIER_SIZE - 1; ++i) {
		curve->binomial_deriv_coeff_[i] = tmp_binomial_coeff[BEZIER_SIZE - 2][i];
		// printf("lower coefficients for position:%d %f\n", i, lower_binomial_coeff[i]);
    }

	for (int i = 0; i <= num_steps; ++i) {
		float t = i * (1.0f / num_steps);
		t_powers[0] = one_minus_t_powers[0] = 1;

		for (int j = 1; j < BEZIER_SIZE; ++j) {
			t_powers[j] = t_powers[j - 1] * t;
			one_minus_t_powers[j] = one_minus_t_powers[j - 1] * (1 - t);

			// printf("current step:%d j:%d t_powers:%f\n", i, j, t_powers[j]);
			// printf("current step:%d j:%d one_minus_t_powers:%f\n", i, j, one_minus_t_powers[j]);
		}

		for (int j = 0; j < BEZIER_SIZE; ++j) {
			curve->bernstein_weights_[i][j] = curve->binomial_coeff_[j] * t_powers[j] * one_minus_t_powers[BEZIER_SIZE - 1 - j];
			
			// 计算速度的Bernstein权重
			if(j < n){
				
				curve->bernstein_deriv_weights_[i][j] = curve->binomial_deriv_coeff_[j] * t_powers[j] * one_minus_t_powers[n - 1 - j];
				// if(i == 0 && j == 0){
				// 	printf("curve->binomial_deriv_coeff_:%f\n", curve->binomial_deriv_coeff_[j]);
				// 	printf("t_powers:%f\n", t_powers[j]);
				// 	printf("one_minus_t_powers:%f\n", one_minus_t_powers[n-1-j]);
				// 	printf("bernstein_deriv_weights_:%f\n", curve->bernstein_deriv_weights_[i][j]);
				// }
			}
		}
	}
	__syncthreads();
}

// Get the position of bezier curve when timestep = t (1 means the final step, 0 means the start)
__device__ __forceinline__ float GetBezierAt(BezierCurve *curve, float *params, int step_idx, int l, int r) {
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / NUM_STEPS);
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
__device__ __forceinline__ float2 GetBezierPosition(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert = false) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14) {
	//   printf("bbbbb! %d %d %d %d", l, r, ll, rr);
	//   // for(;;);
	// }
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / NUM_STEPS);
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
	return convert ? PolarToCartesian(ret.x, ret.y) : ret;
}

__device__ __forceinline__ float2 GetBezierPositionVelocity(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert = false) {
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / NUM_STEPS);
    
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
	return convert ? PolarVelocityToCartesian(position, velocity) : velocity; 
}

__device__ __forceinline__ void GetTrajStateFromBezier(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, float *state, bool convert = false){
    // 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / NUM_STEPS);
    
    float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
    t_powers[0] = one_minus_t_powers[0] = 1;
    const int n = BEZIER_SIZE - 1;

    // 计算t幂
    for (int i = 1; i < BEZIER_SIZE; ++i) {
        t_powers[i] = t_powers[i-1] * t;
        one_minus_t_powers[i] = one_minus_t_powers[i-1] * (1 - t);

		// printf("current step:%d j:%d t_powers:%f\n", threadIdx.x, i, t_powers[i]);
		// printf("current step:%d j:%d one_minus_t_powers:%f\n", threadIdx.x, i, one_minus_t_powers[i]);
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

	if(!convert){
		// 存储状态
		int idx = step_idx * 6;
		state[idx] = position.x;
		state[idx + 1] = position.y;
		state[idx + 2] = velocity.x;
		state[idx + 3] = velocity.y;
		state[idx + 4] = 0.0;		// theta
		state[idx + 5] = 0.0;		// radius for polar coordinate system
	}
    else{
		PolarAllState2Cartesian(position, velocity, state, step_idx);
	}
}

__device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state, bool convert){

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
		int idx = t * 6;
		state[idx] = position.x;
		state[idx + 1] = position.y;
		state[idx + 2] = velocity.x;
		state[idx + 3] = velocity.y;
		state[idx + 4] = 0.0;		// theta
		state[idx + 5] = 0.0;		// radius for polar coordinate system
	}
    else{
		PolarAllState2Cartesian(position, velocity, state, t);
	}
}

__device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false) {
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
	return convert ? PolarVelocityToCartesian(position, velocity) : velocity;
}


__device__ __forceinline__ float2 GetBezierPositionBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14 || (x > 50)) {
	//   printf("ccccc! %d %d %d %d %d", l, r, ll, rr, x);
	//   // for(;;);
	// }
	float2 ret{0., 0.};
	for (int i = 0; i < BEZIER_SIZE; ++i) {
		float bernstein_t = curve->bernstein_weights_[t][i];
		ret.x += bernstein_t * params[i + l];
		ret.y += bernstein_t * params[i + ll];
	}
	return ret;
}

__device__ __forceinline__ float GetBezierAtBasedLookup(BezierCurve *curve, float *params, int t, int l, int r) {
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

struct BezierCurveManager {
    BezierCurve *curve_;
    
    BezierCurveManager() { 
        cudaMalloc(&curve_, sizeof(BezierCurve)); 
    }
    
    ~BezierCurveManager() { 
        cudaFree(curve_); 
    }
};
}
}

#endif
