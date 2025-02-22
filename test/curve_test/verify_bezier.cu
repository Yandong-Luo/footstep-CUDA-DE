#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <iomanip>
#include "cuda_bezier.cuh"

#define BEZIER_SIZE 7
#define NUM_STEPS 30

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

// 使用改进的逻辑生成控制点
void GenerateImprovedControlPoints(float* params, 
                                 float x_start, float y_start, float vx_start, float vy_start, float theta_start,
                                 float x_end, float y_end, float vx_end, float vy_end, float theta_end) {
    // 第一个控制点 - 起始位置
    params[0] = x_start;
    params[BEZIER_SIZE] = y_start;
    
    // 第二个控制点 - 受起始速度影响
    params[1] = x_start + vx_start / 6.0f;
    params[BEZIER_SIZE + 1] = y_start + vy_start / 6.0f;
    
    // 中间控制点 - 使用插值和角度影响
    for (int i = 2; i < 5; i++) {
        float t = i / 6.0f;
        // 基本位置通过线性插值
        float x = (1 - t) * x_start + t * x_end;
        float y = (1 - t) * y_start + t * y_end;
        
        // 添加角度的影响
        float theta_blend = (1 - t) * theta_start + t * theta_end;
        float radius = 0.3f * (1 - pow(2*t-1, 2)); // 抛物线影响因子
        
        x += radius * cos(theta_blend);
        y += radius * sin(theta_blend);
        
        params[i] = x;
        params[BEZIER_SIZE + i] = y;
    }
    
    // 倒数第二个控制点 - 受终止速度影响
    params[5] = x_end - vx_end / 6.0f;
    params[BEZIER_SIZE + 5] = y_end - vy_end / 6.0f;
    
    // 最后一个控制点 - 终止位置
    params[6] = x_end;
    params[BEZIER_SIZE + 6] = y_end;
}

// CUDA 核函数
__global__ void CUDA_BezierPosition(BezierCurve* d_curve, float* d_params, float* d_results) {
    int t_idx = threadIdx.x;
    
    // 使用查表法计算贝塞尔曲线上的点
    // float2 pos = GetBezierPositionBasedLookup(d_curve, d_params, t_idx, 0, BEZIER_SIZE-1, 
    //                                          BEZIER_SIZE, 2*BEZIER_SIZE-1, false);
    
    // // 存储结果
    // d_results[2 * t_idx] = pos.x;
    // d_results[2 * t_idx + 1] = pos.y;

    GetTrajStateFromBezierBasedLookup(d_curve, d_params, t_idx, 0, BEZIER_SIZE-1, 
                                                 BEZIER_SIZE, 2*BEZIER_SIZE-1, d_results);
}

int main() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }
    
    // Print CUDA device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using CUDA device: %s\n", deviceProp.name);
    
    // 使用改进的方法生成控制点
    float params[2 * BEZIER_SIZE];
    GenerateImprovedControlPoints(params, 
                               init_state[0], init_state[1], init_state[2], init_state[3], init_state[4],
                               goal_state[0], goal_state[1], goal_state[2], goal_state[3], goal_state[4]);
    
    // 打印控制点
    printf("Control Points (Bezier curve of degree %d):\n", BEZIER_SIZE-1);
    for (int i = 0; i < BEZIER_SIZE; ++i) {
        printf("P%d: (%.6f, %.6f)\n", i, params[i], params[i + BEZIER_SIZE]);
    }
    
    // 分配 CUDA 设备内存
    float* d_params;
    float* d_results;

    CHECK_CUDA_ERROR(cudaMalloc(&d_params, sizeof(float) * 2 * BEZIER_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, sizeof(float) * 5*(NUM_STEPS + 1)));

    std::shared_ptr<BezierCurveManager> bezier_curve_manager_ = std::make_shared<BezierCurveManager>();
    
    PrepareBinomial<<<1,1>>>(bezier_curve_manager_->curve_);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_params, params, sizeof(float) * 2 * BEZIER_SIZE, cudaMemcpyHostToDevice));

    // 启动 CUDA 计算
    CUDA_BezierPosition<<<1, NUM_STEPS + 1>>>(bezier_curve_manager_->curve_, d_params, d_results);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<float> cuda_results(5 * (NUM_STEPS + 1));
    CHECK_CUDA_ERROR(cudaMemcpy(cuda_results.data(), d_results, sizeof(float) * 5 * (NUM_STEPS + 1), cudaMemcpyDeviceToHost));

    // 打印 CUDA 贝塞尔曲线上的点
    printf("\n%-5s %-15s %-15s %-15s %-15s %-15s\n", "t", "X", "Y", "Vx", "Vy", "Theta");
    printf("-----------------------------------------------------------------------\n");
    
    for (int i = 0; i <= NUM_STEPS; ++i) {
        float t = static_cast<float>(i) / NUM_STEPS;
        printf("%.3f %-15.6f %-15.6f %-15.6f %-15.6f %-15.6f\n", 
               t, 
               cuda_results[i * 5],     // x
               cuda_results[i * 5 + 1], // y
               cuda_results[i * 5 + 2], // vx
               cuda_results[i * 5 + 3], // vy
               cuda_results[i * 5 + 4]  // theta
        );
    }

    // 释放 CUDA 资源
    CHECK_CUDA_ERROR(cudaFree(d_params));
    CHECK_CUDA_ERROR(cudaFree(d_results));

    return 0;
}