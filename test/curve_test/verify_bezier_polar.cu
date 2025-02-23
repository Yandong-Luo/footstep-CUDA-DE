#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <iomanip>
#include "cuda_bezier.cuh"

#define BEZIER_SIZE 7
#define NUM_STEPS 30
#define OUTPUT_DIMS 6

// 初始状态和目标状态
const float init_state[5] = {0.29357406, 0.29125562, -0.01193462, -0.01774755, 1.58432257};
const float goal_state[5] = {1.5, 2.8, 0, 0, 0};

using namespace cudaprocess;
using namespace bezier_curve;

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 极坐标点结构
struct PolarPoint {
    float r;
    float theta;
};

// 极坐标系下的状态结构
struct PolarState {
    float r;          // 半径
    float phi;        // 位置角度
    float r_dot;      // 径向速度
    float phi_dot;    // 角速度
    float theta;      // 状态角度（orientation）
};

// 将状态从笛卡尔坐标系转换到极坐标系
PolarState convertToPolar(float x, float y, float vx, float vy, float theta) {
    PolarState polar;
    
    // 1. 位置转换
    polar.r = sqrt(x * x + y * y);
    polar.phi = atan2(y, x);
    
    // 2. 速度转换
    // ṙ = cos(φ)ẋ + sin(φ)ẏ
    // φ̇ = (-sin(φ)ẋ + cos(φ)ẏ)/r
    polar.r_dot = cos(polar.phi) * vx + sin(polar.phi) * vy;
    polar.phi_dot = (-sin(polar.phi) * vx + cos(polar.phi) * vy) / polar.r;
    
    // 3. 状态角度保持不变
    polar.theta = theta;
    
    return polar;
}

// 在极坐标系统下生成控制点
void GenerateImprovedControlPoints(float* params, 
                                 float x_start, float y_start, float vx_start, float vy_start, float theta_start,
                                 float x_end, float y_end, float vx_end, float vy_end, float theta_end) {
    // 转换起点和终点状态到极坐标系
    PolarState start = convertToPolar(x_start, y_start, vx_start, vy_start, theta_start);
    PolarState end = convertToPolar(x_end, y_end, vx_end, vy_end, theta_end);
    
    const int n = 6; // 6阶贝塞尔曲线
    
    // Start point (P0)
    params[0] = start.r;
    params[BEZIER_SIZE] = start.phi;
    
    // First control point (P1): P1 = (r0 + ṙ0/n, φ0 + φ̇0/n)
    params[1] = start.r + start.r_dot/n;
    params[BEZIER_SIZE + 1] = start.phi + start.phi_dot/n;
    
    // Generate intermediate control points
    for (int i = 2; i < 5; i++) {
        float t = i / 6.0f;
        float r_blend = (1 - t) * start.r + t * end.r;
        float theta_blend = (1 - t) * start.theta + t * end.theta;
        
        float r_influence = 0.3f * (1 - pow(2*t-1, 2));
        r_blend += r_influence;
        
        params[i] = r_blend;
        params[BEZIER_SIZE + i] = theta_blend;
    }
    
    // Second to last control point (Pn-1): Pn-1 = (rn - ṙn/n, φn - φ̇n/n)
    params[5] = end.r - end.r_dot/n;
    params[BEZIER_SIZE + 5] = end.phi - end.phi_dot/n;
    
    // End point (Pn)
    params[6] = end.r;
    params[BEZIER_SIZE + 6] = end.phi;
}

// CUDA 核函数
__global__ void CUDA_BezierPosition(BezierCurve* d_curve, float* d_params, float* d_results) {
    int t_idx = threadIdx.x;
    GetTrajStateFromBezier(d_curve, d_params, t_idx, 0, BEZIER_SIZE-1, 
                          BEZIER_SIZE, 2*BEZIER_SIZE-1, d_results, true);
}

int main() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }
    
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using CUDA device: %s\n", deviceProp.name);
    
    float params[2 * BEZIER_SIZE];
    GenerateImprovedControlPoints(params, 
                               init_state[0], init_state[1], init_state[2], init_state[3], init_state[4],
                               goal_state[0], goal_state[1], goal_state[2], goal_state[3], goal_state[4]);
    
    // 打印控制点（极坐标形式）
    printf("Control Points (Polar):\n");
    for (int i = 0; i < BEZIER_SIZE; ++i) {
        printf("P%d: r = %.6f, φ = %.6f\n", 
               i, params[i], params[i + BEZIER_SIZE]);
    }
    printf("\n");
    
    // CUDA内存分配和计算
    float* d_params;
    float* d_results;
    CHECK_CUDA_ERROR(cudaMalloc(&d_params, sizeof(float) * 2 * BEZIER_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, sizeof(float) * OUTPUT_DIMS*(NUM_STEPS + 1)));
    
    std::shared_ptr<BezierCurveManager> bezier_curve_manager_ = std::make_shared<BezierCurveManager>();
    PrepareBinomial<<<1,1>>>(bezier_curve_manager_->curve_);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params, sizeof(float) * 2 * BEZIER_SIZE, cudaMemcpyHostToDevice));
    
    CUDA_BezierPosition<<<1, NUM_STEPS + 1>>>(bezier_curve_manager_->curve_, d_params, d_results);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    std::vector<float> cuda_results(OUTPUT_DIMS * (NUM_STEPS + 1));
    CHECK_CUDA_ERROR(cudaMemcpy(cuda_results.data(), d_results, sizeof(float) * OUTPUT_DIMS * (NUM_STEPS + 1), cudaMemcpyDeviceToHost));
    
    // 打印表头
    std::cout << std::setw(8) << "Step" 
              << std::setw(10) << "t" 
              << std::setw(12) << "r"
              << std::setw(12) << "theta"
              << std::setw(12) << "dot r"
              << std::setw(12) << "dot theta"
              << std::setw(12) << "X"
              << std::setw(12) << "Y"
              << std::setw(12) << "Vx"
              << std::setw(12) << "Vy"
              << std::setw(15) << "V_magnitude" 
              << std::endl;
    
    std::cout << std::string(120, '-') << std::endl;
    
    // 打印结果
    for (int i = 0; i <= NUM_STEPS; ++i) {
        float t = static_cast<float>(i) / NUM_STEPS;
        float x = cuda_results[i * OUTPUT_DIMS];
        float y = cuda_results[i * OUTPUT_DIMS + 1];
        float vx = cuda_results[i * OUTPUT_DIMS + 2];
        float vy = cuda_results[i * OUTPUT_DIMS + 3];
        float theta = cuda_results[i * OUTPUT_DIMS + 4];
        float radius = cuda_results[i * OUTPUT_DIMS + 5];
        
        // 计算极坐标系下的速度分量
        float vr = (x * vx + y * vy) / radius;
        float vtheta = (x * vy - y * vx) / (radius * radius);
        
        // 计算速度大小
        float v_magnitude = sqrt(vx * vx + vy * vy);
        
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(8) << i
                  << std::setw(10) << t
                  << std::setw(12) << radius
                  << std::setw(12) << theta
                  << std::setw(12) << vr
                  << std::setw(12) << vtheta
                  << std::setw(12) << x
                  << std::setw(12) << y
                  << std::setw(12) << vx
                  << std::setw(12) << vy
                  << std::setw(15) << v_magnitude
                  << std::endl;
    }
    
    // 释放CUDA资源
    CHECK_CUDA_ERROR(cudaFree(d_params));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    
    return 0;
}