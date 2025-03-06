#ifndef CUDA_NUMERICAL_INTEGRATION_H
#define CUDA_NUMERICAL_INTEGRATION_H

// #include "curve/bezier_curve.cuh"
#include "utils/config.h"

namespace cudaprocess {
namespace integration {

    // ######################### CUDA VERSION #####################
    // Simpson 3/8 - CUDA device
    __device__ __forceinline__ static float simpson_3_8_device(float (*func)(float, const bezier_curve::BezierCurve*, const float*), 
                                             const float& L, 
                                             const float& R, 
                                             const bezier_curve::BezierCurve* bezier_curve,
                                             const float *curve_param) {
        float mid_L = (2*L + R) / 3.0f;
        float mid_R = (L + 2*R) / 3.0f;
        
        return (func(L, bezier_curve, curve_param) + 
                3.0f * func(mid_L, bezier_curve, curve_param) + 
                3.0f * func(mid_R, bezier_curve, curve_param) + 
                func(R, bezier_curve, curve_param)) * (R - L) / 8.0f;
    }
    
    // adaptive Simpson 3/8 - CUDA device
    __device__ __forceinline__ static float adaptive_simpson_3_8_device(float (*func)(float, const bezier_curve::BezierCurve*, const float*), 
                                                     const float& L, 
                                                     const float& R, 
                                                     const bezier_curve::BezierCurve* bezier_curve, 
                                                     const float *curve_param,
                                                     const float& eps = 0.0001f,
                                                     int depth = 0,
                                                     int max_depth = 10) {
        const float mid = (L + R) / 2.0f;
        
        float ST = simpson_3_8_device(func, L, R, bezier_curve, curve_param);
        float SL = simpson_3_8_device(func, L, mid, bezier_curve, curve_param);
        float SR = simpson_3_8_device(func, mid, R, bezier_curve, curve_param);
        
        float ans = SL + SR;
        
        // 检查收敛或达到最大深度
        if (fabsf(ans - ST) <= 15.0f * eps || depth >= max_depth) {
            return SL + SR + (ans - ST) / 15.0f;
        }
        
        return adaptive_simpson_3_8_device(func, L, mid, bezier_curve, curve_param, eps / 2.0f, depth + 1, max_depth) + 
               adaptive_simpson_3_8_device(func, mid, R, bezier_curve, curve_param, eps / 2.0f, depth + 1, max_depth);
    }

    __device__ __forceinline__ static float iterative_adaptive_simpson(float (*func)(float, const bezier_curve::BezierCurve*, const float*),
                                                float a, float b, 
                                                const bezier_curve::BezierCurve* bezier_curve,
                                                const float* curve_param,
                                                float eps = 0.0001f) {
        // Stack for storing intervals to process
        #define MAX_STACK 50  // Reasonable stack size
        float stack_L[MAX_STACK], stack_R[MAX_STACK], stack_eps[MAX_STACK];
        int stack_size = 1;
        
        // Initial push
        stack_L[0] = a;
        stack_R[0] = b;
        stack_eps[0] = eps;
        
        float total = 0.0f;
        
        // Iterative processing
        while (stack_size > 0) {
            // Pop
            stack_size--;
            float L = stack_L[stack_size];
            float R = stack_R[stack_size];
            float current_eps = stack_eps[stack_size];
            float mid = (L + R) / 2.0f;
            
            // Calculate approximations
            float ST = simpson_3_8_device(func, L, R, bezier_curve, curve_param);
            float SL = simpson_3_8_device(func, L, mid, bezier_curve, curve_param);
            float SR = simpson_3_8_device(func, mid, R, bezier_curve, curve_param);
            
            float approx_sum = SL + SR;
            
            // Check convergence
            if (fabsf(approx_sum - ST) <= 15.0f * current_eps || stack_size >= MAX_STACK - 2) {
                total += SL + SR + (approx_sum - ST) / 15.0f;
            } else {
                // Push right interval
                stack_L[stack_size] = mid;
                stack_R[stack_size] = R;
                stack_eps[stack_size] = current_eps / 2.0f;
                stack_size++;
                
                // Push left interval
                stack_L[stack_size] = L;
                stack_R[stack_size] = mid;
                stack_eps[stack_size] = current_eps / 2.0f;
                stack_size++;
            }
        }
        
        return total;
    }

    // ########################### cpu ##################
    // Simpson 3/8 - CPU
    template <typename F>
    static float simpson_3_8(F&& func, const float& L, const float& R) {
        float mid_L = (2*L + R) / 3.0f;
        float mid_R = (L + 2*R) / 3.0f;
        
        return (func(L) + 
                3.0f * func(mid_L) + 
                3.0f * func(mid_R) + 
                func(R)) * (R - L) / 8.0f;
    }
    
    // adaptive Simpson 3/8 - CPU
    template <typename F>
    static float adaptive_simpson_3_8(F&& func, 
                                    const float& L, 
                                    const float& R, 
                                    const float& eps = 0.0001f) {
        const float mid = (L + R) / 2.0f;
        
        float ST = simpson_3_8(func, L, R);
        float SL = simpson_3_8(func, L, mid);
        float SR = simpson_3_8(func, mid, R);
        
        float ans = SL + SR;
        
        if (fabsf(ans - ST) <= 15.0f * eps) {
            return SL + SR + (ans - ST) / 15.0f;
        }
        
        return adaptive_simpson_3_8(func, L, mid, eps / 2.0f) + 
               adaptive_simpson_3_8(func, mid, R, eps / 2.0f);
    }
}
}

#endif