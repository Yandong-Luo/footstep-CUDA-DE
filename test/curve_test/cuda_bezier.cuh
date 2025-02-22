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
	// combination coefficient C_n^i, bernstein_weights_ t^j * (1-t)^{n-j}, bernstein_deriv_weights_
	float binomial_coeff_[BEZIER_SIZE], bernstein_weights_[BEZIER_SIZE + 1][BEZIER_SIZE + 1], bernstein_deriv_weights_[BEZIER_SIZE + 1][BEZIER_SIZE];
};

// *************** Polar to Cartesian *****************
__device__ __forceinline__ float2 PolarToCartesian(float radius, float theta) {
	return make_float2(radius * __cosf(theta), radius * __sinf(theta));
}

__device__ __forceinline__ float2 PolarVelocityToCartesian(float radius, float theta, float V_r, float V_theta) {
	float Vx = V_r * __cosf(theta) - radius * V_theta * __sinf(theta);
	float Vy = V_r * __sinf(theta) + radius * V_theta * __cosf(theta);
	return make_float2(Vx, Vy);
}

__global__ void PrepareBinomial(BezierCurve* curve) {
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];	// record t^j, (1-t)^{n-j}

	float tmp_binomial_coeff[BEZIER_SIZE][BEZIER_SIZE + 1];
	
	tmp_binomial_coeff[0][0] = tmp_binomial_coeff[1][1] = 1;
	tmp_binomial_coeff[0][1] = 0;
	tmp_binomial_coeff[1][0] = 1;

	// Pascal’s Triangle to calculate Binomial Coefficients
	for (int i{2}; i < BEZIER_SIZE; ++i) {
		tmp_binomial_coeff[i][0] = tmp_binomial_coeff[i][i] = 1;
		for (int j{1}; j < i; ++j) {
			tmp_binomial_coeff[i][j] = tmp_binomial_coeff[i - 1][j - 1] + tmp_binomial_coeff[i - 1][j];
		}
	}
	// save Binomial Coefficients to comb
	for (int i{0}; i < BEZIER_SIZE; ++i) {
		curve->binomial_coeff_[i] = tmp_binomial_coeff[BEZIER_SIZE - 1][i];
	}

	const int num_steps = NUM_STEPS;
    const int n = BEZIER_SIZE - 1;  // 贝塞尔曲线的阶数
    
    // 预计算低一阶的二项式系数(用于导数计算)
    float lower_binomial_coeff[BEZIER_SIZE - 1];  // n-1阶贝塞尔曲线的二项式系数
    for (int i = 0; i < BEZIER_SIZE - 1; ++i) {
        lower_binomial_coeff[i] = tmp_binomial_coeff[BEZIER_SIZE - 2][i];
    }

	// 
	for (int i = 0; i <= num_steps; ++i) {
		t_powers[0] = one_minus_t_powers[0] = 1;
		for (int j{1}; j < BEZIER_SIZE; ++j) {
			t_powers[j] = t_powers[j - 1] * (i * 1. / num_steps);
			one_minus_t_powers[j] = one_minus_t_powers[j - 1] * (1 - (i * 1. / num_steps));
		}
		for (int j{0}; j < BEZIER_SIZE; ++j) {
			curve->bernstein_weights_[i][j] = curve->binomial_coeff_[j] * t_powers[j] * one_minus_t_powers[BEZIER_SIZE - 1 - j];
		}

		// // for velocity
		// for(int j = 0; j < BEZIER_SIZE - 1; ++j){
		// 	// curve->bernstein_deriv_weights_[i][j] = (BEZIER_SIZE - 1) * (curve->bernstein_weights_[i][j + 1] - curve->bernstein_weights_[i][j]);
        //     curve->bernstein_deriv_weights_[i][j] = (BEZIER_SIZE - 1) * (curve->bernstein_weights_[i][j] - curve->bernstein_weights_[i][j + 1]);
		// }
		// curve->bernstein_deriv_weights_[i][BEZIER_SIZE - 1] = 0; 

        // 计算速度的Bernstein权重
        // 使用n-1阶贝塞尔曲线的Bernstein多项式，乘以n
        for(int j = 0; j < BEZIER_SIZE - 1; ++j) {
            curve->bernstein_deriv_weights_[i][j] = n * lower_binomial_coeff[j] * 
                                                 t_powers[j] * 
                                                 one_minus_t_powers[BEZIER_SIZE - 2 - j];
        }
        curve->bernstein_deriv_weights_[i][BEZIER_SIZE - 1] = 0; 
	}
}

// Get the position of bezier curve when timestep = t (1 means the final step, 0 means the start)
__device__ __forceinline__ float GetBezierAt(BezierCurve *curve, float *params, float t, int l, int r) {
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
__device__ __forceinline__ float2 GetBezierPosition(BezierCurve *curve, float *params, float t, int l, int r, int ll, int rr, bool convert = false) {
	// if (l != 0 || r != 7 || ll != 7 || rr != 14) {
	//   printf("bbbbb! %d %d %d %d", l, r, ll, rr);
	//   // for(;;);
	// }
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

__device__ __forceinline__ float2 GetBezierPositionVelocity(BezierCurve *curve, float *params, float t, int l, int r, int ll, int rr, bool convert = false) {
	float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
	t_powers[0] = one_minus_t_powers[0] = 1;

	// 计算 t^i 和 (1 - t)^i
	for (int i{1}; i < BEZIER_SIZE - 1; ++i) {
			t_powers[i] = t_powers[i - 1] * t;
			one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1 - t);
	}

	float2 velocity{0., 0.};
	int n = BEZIER_SIZE - 1; // 阶数 n

	// 计算贝塞尔速度
	for (int i = 0; i < n; ++i) {
        float bernstein_t = n * curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[n - 1 - i];

        velocity.x += bernstein_t * (params[i + 1 + l] - params[i + l]);
        velocity.y += bernstein_t * (params[i + 1 + ll] - params[i + ll]);
	}

	if (convert) {
		float radius = 0.0, theta = 0.0;
		for (int i = 0; i < BEZIER_SIZE; ++i) {
            float bernstein_t = curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[n - i];
            radius += bernstein_t * params[i + l];
            theta += bernstein_t * params[i + ll];
		}
		return PolarVelocityToCartesian(radius, theta, velocity.x, velocity.y);
	}
	return velocity;
}

__device__ __forceinline__ void GetTrajStateFromBezier(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state){
    float t_powers[BEZIER_SIZE], one_minus_t_powers[BEZIER_SIZE];
	t_powers[0] = one_minus_t_powers[0] = 1;

	// 计算 t^i 和 (1 - t)^i
	for (int i{1}; i < BEZIER_SIZE - 1; ++i) {
        t_powers[i] = t_powers[i - 1] * t;
        one_minus_t_powers[i] = one_minus_t_powers[i - 1] * (1 - t);
	}
    float2 position{0.0f, 0.0f};
	float2 velocity{0.0f, 0.0f};
	int n = BEZIER_SIZE - 1; // 阶数 n

	// 计算贝塞尔速度
	for (int i = 0; i < BEZIER_SIZE; ++i) {
        if(i < BEZIER_SIZE - 1){
            float bernstein_t = n * curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[n - 1 - i];
            velocity.x += bernstein_t * (params[i + 1 + l] - params[i + l]);
            velocity.y += bernstein_t * (params[i + 1 + ll] - params[i + ll]);
        }

        float bernstein_t = curve->binomial_coeff_[i] * t_powers[i] * one_minus_t_powers[BEZIER_SIZE - 1 - i];		// B(t) in formula
		position.x += bernstein_t * params[i + l];
		position.y += bernstein_t * params[i + ll];
	}
    
    int idx = t * 5; // 每个时间步有5个状态值
    
    state[idx] = position.x;
    state[idx + 1] = position.y;
    state[idx + 2] = velocity.x;
    state[idx + 3] = velocity.y;
    state[idx + 4] = 0.0; // 或者根据需要设置角度
	// return PolarAllState2Cartesian(position, velocity, state);
}

__device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state){

    float2 position{0.0f, 0.0f};
	float2 velocity{0.0f, 0.0f};

    for (int i = 0; i < BEZIER_SIZE; ++i) {
        if(i < BEZIER_SIZE - 1){
            float bernstein_deriv_t = curve->bernstein_deriv_weights_[t][i];  // 查表获取导数
            velocity.x += bernstein_deriv_t * (params[i + 1 + l] - params[i + l]);
            velocity.y += bernstein_deriv_t * (params[i + 1 + ll] - params[i + ll]);
        }
        float bernstein_t = curve->bernstein_weights_[t][i];
		position.x += bernstein_t * params[i + l];
		position.y += bernstein_t * params[i + ll];
	}

    int idx = t * 5; // 每个时间步有5个状态值
    
    state[idx] = position.x;
    state[idx + 1] = position.y;
    state[idx + 2] = velocity.x;
    state[idx + 3] = velocity.y;
    state[idx + 4] = 0.0; // 或者根据需要设置角度
	// return PolarAllState2Cartesian(position, velocity, state);
}

// __device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t_idx, int l, int r, int ll, int rr, float *state) {
//     // 位置计算保持不变
//     float2 position{0.0f, 0.0f};
//     for (int i = 0; i < BEZIER_SIZE; ++i) {
//         float bernstein_t = curve->bernstein_weights_[t_idx][i];
//         position.x += bernstein_t * params[i + l];
//         position.y += bernstein_t * params[i + ll];
//     }
    
//     // 计算导数值
//     float2 velocity{0.0f, 0.0f};
//     const int n = BEZIER_SIZE - 1; // 贝塞尔曲线的阶数
    
//     // 首先计算导数贝塞尔曲线的控制点 (注意：乘以n)
//     float deriv_control_x[BEZIER_SIZE-1];
//     float deriv_control_y[BEZIER_SIZE-1];
    
//     for (int i = 0; i < n; ++i) {
//         deriv_control_x[i] = (params[i+1+l] - params[i+l]) * n;
//         deriv_control_y[i] = (params[i+1+ll] - params[i+ll]) * n;
//     }
    
//     // 然后使用(n-1)阶贝塞尔曲线的Bernstein权重
//     // bernstein_deriv_weights_中已经包含了n的因子
//     for (int i = 0; i < n; ++i) {
//         float bernstein_deriv = curve->bernstein_deriv_weights_[t_idx][i] / n; // 移除n因子，因为已经乘到控制点上了
//         velocity.x += bernstein_deriv * deriv_control_x[i];
//         velocity.y += bernstein_deriv * deriv_control_y[i];
//     }
    
//     // 计算角度
//     float theta = atan2f(velocity.y, velocity.x);
    
//     // 存储结果
//     int idx = t_idx * 5;
//     state[idx] = position.x;
//     state[idx + 1] = position.y;
//     state[idx + 2] = velocity.x;
//     state[idx + 3] = velocity.y;
//     state[idx + 4] = theta;
// }

__device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert = false) {
	float2 velocity{0., 0.};
	float radius = 0.0f, theta = 0.0f;
	for (int i = 0; i < BEZIER_SIZE - 1; ++i) {
        float bernstein_deriv_t = curve->bernstein_deriv_weights_[t][i];  // 查表获取导数
        velocity.x += bernstein_deriv_t * (params[i + 1 + l] - params[i + l]);
        velocity.y += bernstein_deriv_t * (params[i + 1 + ll] - params[i + ll]);

        if(convert) 	radius += curve->bernstein_weights_[t][i] * params[i+l];
        if(convert) 	theta += curve->bernstein_weights_[t][i] * params[i+ll];
	}
	return convert ? PolarVelocityToCartesian(radius, theta, velocity.x, velocity.y) : velocity;
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

__device__ __forceinline__ float GetBezierAt(BezierCurve *curve, float *params, int t, int l, int r, bool convert = false) {
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

// __device__ __forceinline__ void PolarAllState2Cartesian(float2 position, float2 velocity, footstep::StateVector *state){
//     float radius = position.x, theta = position.y;
//     float v_r = velocity.x, v_theta = velocity.y;

// 	state->data[0] = radius * __cosf(theta);
//     state->data[1] = radius * __sinf(theta);
//     state->data[2] = v_r * __cosf(theta) - radius * v_theta * __sinf(theta);
//     state->data[3] = v_r * __sinf(theta) + radius * v_theta * __cosf(theta);
//     state->data[4] = theta;
// }

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
