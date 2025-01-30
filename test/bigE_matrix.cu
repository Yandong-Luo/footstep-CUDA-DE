#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define STATE_DIM 5  // E 是 5x5 矩阵

void construct_big_E(float* E, float* big_E, int N) {
    float *d_E, *d_Ek, *d_big_E;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配 GPU 内存
    cudaMalloc(&d_E, STATE_DIM * STATE_DIM * sizeof(float));
    cudaMalloc(&d_Ek, STATE_DIM * STATE_DIM * sizeof(float));
    cudaMalloc(&d_big_E, STATE_DIM * STATE_DIM * N * sizeof(float));

    // 复制 E 到 GPU
    cudaMemcpy(d_E, E, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ek, E, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS 计算矩阵幂次
    float alpha = 1.0f, beta = 0.0f;
    for (int k = 0; k < N; k++) {
        // 复制当前 Ek 到 big_E
        cudaMemcpy(d_big_E + k * STATE_DIM * STATE_DIM, d_Ek, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice);

        // 计算 Ek+1 = E * Ek
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    STATE_DIM, STATE_DIM, STATE_DIM,
                    &alpha, d_E, STATE_DIM, d_Ek, STATE_DIM,
                    &beta, d_Ek, STATE_DIM);
    }

    // 复制结果回 CPU
    cudaMemcpy(big_E, d_big_E, STATE_DIM * STATE_DIM * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理
    cublasDestroy(handle);
    cudaFree(d_E);
    cudaFree(d_Ek);
    cudaFree(d_big_E);
}

// 测试代码
int main() {
    int N = 30;  // 计算 100 步
    float E[STATE_DIM * STATE_DIM] = {
        1, 0, 0.513166, 0, 0,
        0, 1, 0, 0.513166, 0,
        0, 0, 1.89298, 0, 0,
        0, 0, 0, 1.89298, 0,
        0, 0, 0, 0, 1
    };

    float* big_E = new float[STATE_DIM * STATE_DIM * N];  // 存储大矩阵

    // 调用 cuBLAS 计算
    construct_big_E(E, big_E, N);

    // 打印部分结果
    for (int i = 0; i < N; i++) {  // 仅展示前 5 次幂
        std::cout << "E^" << (i+1) << ":\n";
        for (int j = 0; j < STATE_DIM; j++) {
            for (int k = 0; k < STATE_DIM; k++) {
                std::cout << big_E[i * STATE_DIM * STATE_DIM + j * STATE_DIM + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------------------" << std::endl;
    }

    delete[] big_E;
    return 0;
}