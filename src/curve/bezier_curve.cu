#include "curve/bezier_curve.cuh"

namespace cudaprocess{

namespace bezier_curve{

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

// *************** Bezier *****************
__global__ void PrepareBinomialandFixedPoint(BezierCurve* curve) {
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

	const int num_steps = CURVE_NUM_STEPS - 1;
    const int n = BEZIER_SIZE - 1;  // 贝塞尔曲线的阶数
    
    // 预计算低一阶的二项式系数(用于导数计算)
    for (int i = 0; i < BEZIER_SIZE - 1; ++i) {
		curve->binomial_deriv_coeff_[i] = tmp_binomial_coeff[BEZIER_SIZE - 2][i];
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

    // calculate the fixed point based on robot state
    float x_start = footstep::init_state[0];
    float y_start = footstep::init_state[1];
    float vx_start = footstep::init_state[2];
    float vy_start = footstep::init_state[3];
    float theta_start = footstep::init_state[4];

    float x_goal = footstep::goal_state[0];
    float y_goal = footstep::goal_state[1];
    float vx_goal = footstep::goal_state[2];
    float vy_goal = footstep::goal_state[3];
    float theta_goal = footstep::goal_state[4];

	float sum_v = sqrtf(footstep::ux_ub * footstep::ux_ub + footstep::uy_ub + footstep::uy_ub);

    // Start point (P0)
    curve->control_points[0].x = x_start;
    curve->control_points[0].y = y_start;
	curve->control_points[0].z = theta_start;
	curve->is_point_xy_fixed[0] = 1;
	curve->is_theta_point_fixed[0] = 1;
    
    // First control point (P1)
    // 如果速度不为0，使用速度
    if (abs(vx_start) > 1e-6 || abs(vy_start) > 1e-6) {
        // B'(0) = n(P1 - P0) = (vx_start, vy_start)
        // 所以 P1 = P0 + (vx_start/n, vy_start/n)
        curve->control_points[1].x = x_start + vx_start / static_cast<float>(n);
        curve->control_points[1].y = y_start + vy_start / static_cast<float>(n);
		curve->is_point_xy_fixed[1] = 1;
    } else {
        // 如果速度为0，使用角度
        // B'(0) = n(P1 - P0) = (cos(theta_0), sin(theta_0))
        // 所以 P1 = P0 + (cos(theta_0)/n, sin(theta_0)/n)
		
        curve->control_points[1].x = x_start + sum_v * __cosf(theta_start) / static_cast<float>(n);
        curve->control_points[1].y = y_start + sum_v * __sinf(theta_start) / static_cast<float>(n);
		curve->is_point_xy_fixed[1] = 1;
    }
    
    // Second to last control point (Pn-1)
    // 如果速度不为0，使用速度
    if (abs(vx_goal) > 1e-6 || abs(vy_goal) > 1e-6) {
        // B'(1) = n(Pn - Pn-1) = (vx_end, vy_end)
        // 所以 Pn-1 = Pn - (vx_end/n, vy_end/n)
        curve->control_points[n-1].x = x_goal - vx_goal / static_cast<float>(n);
        curve->control_points[n-1].y = y_goal - vy_goal / static_cast<float>(n);
		curve->is_point_xy_fixed[n-1] = 1;
    } else {
        // 如果速度为0，使用角度
        // B'(1) = n(Pn - Pn-1) = (cos(theta_n), sin(theta_n))
        // 所以 Pn-1 = Pn - (cos(theta_n)/n, sin(theta_n)/n)
        // controlPoints.push_back({
        //     x_end - cos(theta_end)/n,
        //     y_end - sin(theta_end)/n
        // });
        curve->control_points[n-1].x = x_goal - sum_v * __cosf(theta_goal) / static_cast<float>(n);
        curve->control_points[n-1].y = y_goal - sum_v * __sinf(theta_goal) / static_cast<float>(n);
		curve->is_point_xy_fixed[n-1] = 1;
    }
    
    // End point (Pn)
    curve->control_points[n].x = x_goal;
    curve->control_points[n].y = y_goal;
	curve->control_points[n].z = theta_goal;
	curve->is_point_xy_fixed[n] = 1;
	curve->is_theta_point_fixed[n] = 1;
}

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
__device__ __forceinline__ float2 GetBezierPosition(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert) {
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
	return convert ? PolarToCartesian(ret.x, ret.y) : ret;
}

__device__ __forceinline__ float2 GetBezierPositionVelocity(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, bool convert) {
	// 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
    
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

__device__ __forceinline__ void GetTrajStateFromBezier(BezierCurve *curve, float *params, int step_idx, int l, int r, int ll, int rr, float *state, bool convert){
    // 计算实际的 t 值 (0 到 1 之间)
    float t = step_idx * (1.0f / (CURVE_NUM_STEPS - 1));
    
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

// __device__ __forceinline__ void GetTrajStateFromBezierBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, float *state, bool convert){

//     float2 position{0.0f, 0.0f};
// 	float2 velocity{0.0f, 0.0f};
// 	const int n = BEZIER_SIZE - 1;

//     for (int i = 0; i < BEZIER_SIZE; ++i) {
//         float bernstein_t = curve->bernstein_weights_[t][i];
// 		position.x += bernstein_t * params[i + l];
// 		position.y += bernstein_t * params[i + ll];
		
// 		if(i < n){
// 			float deriv_ctrl_x = n * (params[i + 1 + l] - params[i + l]);
// 			float deriv_ctrl_y = n * (params[i + 1 + ll] - params[i + ll]);

// 			float bernstein_deriv_t = curve->bernstein_deriv_weights_[t][i];

// 			velocity.x += bernstein_deriv_t * deriv_ctrl_x;
//         	velocity.y += bernstein_deriv_t * deriv_ctrl_y;
// 		}
// 	}

//     if(!convert){
// 		// 存储状态
// 		int idx = t * 6;
// 		state[idx] = position.x;
// 		state[idx + 1] = position.y;
// 		state[idx + 2] = velocity.x;
// 		state[idx + 3] = velocity.y;
// 		state[idx + 4] = 0.0;		// theta
// 		state[idx + 5] = 0.0;		// radius for polar coordinate system
// 	}
//     else{
// 		PolarAllState2Cartesian(position, velocity, state, t);
// 	}
// }

__device__ __forceinline__ float2 GetBezierPositionVelocityBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert) {
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


__device__ __forceinline__ float2 GetBezierPositionBasedLookup(BezierCurve *curve, float *params, int t, int l, int r, int ll, int rr, bool convert) {
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

BezierCurveManager::BezierCurveManager() { 
	cudaMalloc(&curve_, sizeof(BezierCurve)); 
}
    
BezierCurveManager::~BezierCurveManager() { 
	cudaFree(curve_); 
}
}
}
