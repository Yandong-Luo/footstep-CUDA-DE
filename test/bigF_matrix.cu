// #include <iostream>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// #define STATE_DIM 5    
// #define CONTROL_DIM 3  
// #define N 10          

// void transpose_matrix(float* out, const float* in, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             out[j * rows + i] = in[i * cols + j];
//         }
//     }
// }

// void compute_power(cublasHandle_t handle, const float* d_E, int power, float* d_result) {
//     float alpha = 1.0f, beta = 0.0f;
//     float* d_temp;
//     cudaMalloc(&d_temp, STATE_DIM * STATE_DIM * sizeof(float));
    
//     if (power == 0) {
//         float identity[STATE_DIM * STATE_DIM] = {0};
//         for(int i = 0; i < STATE_DIM; i++) {
//             identity[i * STATE_DIM + i] = 1.0f;
//         }
//         // 转置单位矩阵（实际上单位矩阵转置后还是自己）
//         cudaMemcpy(d_result, identity, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
//     } else {
//         cudaMemcpy(d_result, d_E, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
        
//         for(int p = 1; p < power; p++) {
//             cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                        STATE_DIM, STATE_DIM, STATE_DIM,
//                        &alpha,
//                        d_E, STATE_DIM,
//                        d_result, STATE_DIM,
//                        &beta,
//                        d_temp, STATE_DIM);
//             cudaMemcpy(d_result, d_temp, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
//         }
//     }
    
//     cudaFree(d_temp);
// }

// void construct_big_F(const float* E, const float* F, float* big_F) {
//     cublasHandle_t handle;
//     cublasCreate(&handle);
    
//     // 为转置的矩阵分配内存
//     float* E_trans = new float[STATE_DIM * STATE_DIM];
//     float* F_trans = new float[STATE_DIM * CONTROL_DIM];
//     transpose_matrix(E_trans, E, STATE_DIM, STATE_DIM);
//     transpose_matrix(F_trans, F, STATE_DIM, CONTROL_DIM);
    
//     float *d_E, *d_F, *d_E_power, *d_result;
//     cudaMalloc(&d_E, STATE_DIM * STATE_DIM * sizeof(float));
//     cudaMalloc(&d_F, STATE_DIM * CONTROL_DIM * sizeof(float));
//     cudaMalloc(&d_E_power, STATE_DIM * STATE_DIM * sizeof(float));
//     cudaMalloc(&d_result, STATE_DIM * CONTROL_DIM * sizeof(float));
    
//     // 复制转置后的矩阵到设备
//     cudaMemcpy(d_E, E_trans, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_F, F_trans, STATE_DIM * CONTROL_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
//     float alpha = 1.0f, beta = 0.0f;
//     memset(big_F, 0, STATE_DIM * N * CONTROL_DIM * N * sizeof(float));
    
//     float* result_block = new float[STATE_DIM * CONTROL_DIM];
    
//     for(int i = 0; i < N; i++) {
//         for(int j = 0; j <= i; j++) {
//             compute_power(handle, d_E, i-j, d_E_power);
            
//             cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                        STATE_DIM, CONTROL_DIM, STATE_DIM,
//                        &alpha,
//                        d_E_power, STATE_DIM,
//                        d_F, STATE_DIM,
//                        &beta,
//                        d_result, STATE_DIM);
            
//             // 获取结果并转置回行主序
//             cudaMemcpy(result_block, d_result, STATE_DIM * CONTROL_DIM * sizeof(float), cudaMemcpyDeviceToHost);
            
//             // 转置并存储结果
//             for(int r = 0; r < STATE_DIM; r++) {
//                 for(int c = 0; c < CONTROL_DIM; c++) {
//                     big_F[(i * STATE_DIM + r) * (CONTROL_DIM * N) + j * CONTROL_DIM + c] = 
//                         result_block[c * STATE_DIM + r];
//                 }
//             }
//         }
//     }
    
//     delete[] E_trans;
//     delete[] F_trans;
//     delete[] result_block;
    
//     cudaFree(d_E);
//     cudaFree(d_F);
//     cudaFree(d_E_power);
//     cudaFree(d_result);
//     cublasDestroy(handle);
// }

// int main() {
//     float E[STATE_DIM * STATE_DIM] = {
//         1, 0, 0.513166, 0, 0,
//         0, 1, 0, 0.513166, 0,
//         0, 0, 1.89298, 0, 0,
//         0, 0, 0, 1.89298, 0,
//         0, 0, 0, 0, 1
//     };
    
//     float F[STATE_DIM * CONTROL_DIM] = {
//         -0.892976, -0, 0,
//         -0, -0.892976, 0,
//         -5.03416, -0, 0,
//         -0, -5.03416, 0,
//         0, 0, 1
//     };
    
//     float* big_F = new float[STATE_DIM * N * CONTROL_DIM * N]();
//     construct_big_F(E, F, big_F);
    
//     for(int i = 0; i < STATE_DIM * N; i++) {
//         for(int j = 0; j < CONTROL_DIM * N; j++) {
//             printf("%.6f ", big_F[i * (CONTROL_DIM * N) + j]);
//         }
//         printf("\n");
//     }
    
//     delete[] big_F;
//     return 0;
// }


#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define STATE_DIM 5  // E 是 5x5 矩阵
#define CONTROL_DIM 3  // F 是 5x3 矩阵

void construct_big_EF(float* E, float* F, float* big_E, float* big_F, int N) {
    float *d_E, *d_Ek, *d_big_E, *d_F, *d_big_F, *result_block;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配 GPU 内存
    cudaMalloc(&d_E, STATE_DIM * STATE_DIM * sizeof(float));
    cudaMalloc(&d_Ek, STATE_DIM * STATE_DIM * sizeof(float));
    cudaMalloc(&d_big_E, STATE_DIM * STATE_DIM * N * sizeof(float));
    cudaMalloc(&d_F, STATE_DIM * CONTROL_DIM * sizeof(float));
    cudaMalloc(&d_big_F, STATE_DIM * CONTROL_DIM * N * N* sizeof(float));
    cudaMalloc(&result_block, STATE_DIM * CONTROL_DIM * sizeof(float));  // 结果矩阵 F_kj

    float* d_E_power;
    cudaMalloc(&d_E_power, STATE_DIM * STATE_DIM * sizeof(float));

    cudaMemset(d_big_E, 0, STATE_DIM * STATE_DIM * N * sizeof(float));
    cudaMemset(d_big_F, 0, STATE_DIM * CONTROL_DIM * N * N * sizeof(float));

    // 复制 E, F 到 GPU
    cudaMemcpy(d_E, E, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ek, E, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, STATE_DIM * CONTROL_DIM * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    
    for (int k = 0; k < N; k++) {
        // 存储 E^k 到 big_E
        cudaMemcpy(d_big_E + k * STATE_DIM * STATE_DIM, d_Ek, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice);

        // bigF 计算
        for(int j = 0; j <= k; ++j){
            // if(j == k)  cudaMemcpy(bigF + k * state_dims * state_dims, d_Ek, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice);
            // power 是公式中E的次方，对应到bigE中进行查找还需要-1
            int power = k - j;
            if(power == 0){
                printf("....................\n");
                cudaMemcpy(result_block, F, STATE_DIM * CONTROL_DIM * sizeof(float), cudaMemcpyHostToDevice);
                printf("=====================+++++++++++++++++++++\n");
            }
            else{
                printf("??????????????????????????\n");
                cudaMemcpy(d_E_power, d_big_E + (power - 1) * STATE_DIM * STATE_DIM, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
                printf("=========================\n");
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        CONTROL_DIM, STATE_DIM, STATE_DIM,
                        &alpha, d_F, CONTROL_DIM, d_E_power, STATE_DIM,
                        &beta, result_block, CONTROL_DIM);
            }

            // 转置并存储结果
            for(int r = 0; r < STATE_DIM; r++) {
                for(int c = 0; c < CONTROL_DIM; c++) {
                    d_big_F[(k * STATE_DIM + r) * (CONTROL_DIM * N) + j * CONTROL_DIM + c] = 
                        result_block[c * STATE_DIM + r];
                }
            }
        }

        // 计算 E^(k+1) = E * E^k
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    STATE_DIM, STATE_DIM, STATE_DIM,
                    &alpha, d_E, STATE_DIM, d_Ek, STATE_DIM,
                    &beta, d_Ek, STATE_DIM);
    }

    // 复制结果回 CPU
    cudaMemcpy(big_E, d_big_E, STATE_DIM * STATE_DIM * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(big_F, d_big_F, STATE_DIM * CONTROL_DIM * N * N* sizeof(float), cudaMemcpyDeviceToHost);

    // 释放资源
    cublasDestroy(handle);
    cudaFree(d_E);
    cudaFree(d_Ek);
    cudaFree(d_big_E);
    cudaFree(d_F);
    cudaFree(d_big_F);
    cudaFree(result_block);
}

// 测试代码
int main() {
    int N = 30;  // 计算 N 步
    float E[STATE_DIM * STATE_DIM] = {
        1, 0, 0.513166, 0, 0,
        0, 1, 0, 0.513166, 0,
        0, 0, 1.89298, 0, 0,
        0, 0, 0, 1.89298, 0,
        0, 0, 0, 0, 1
    };

    float F[STATE_DIM * CONTROL_DIM] = {
        -0.892976, -0, 0,
        -0, -0.892976, 0,
        -5.03416, -0, 0,
        -0, -5.03416, 0,
        0, 0, 1
    };

    float* big_E = new float[STATE_DIM * STATE_DIM * N];  // 存储 bigE
    float* big_F = new float[STATE_DIM * CONTROL_DIM * N *N];  // 存储 bigF

    // 计算 bigE 和 bigF
    construct_big_EF(E, F, big_E, big_F, N);

    // 打印 bigE
    std::cout << "==== bigE ====" << std::endl;
    for (int i = 0; i < 5; i++) {  // 仅展示前 5 次幂
        std::cout << "E^" << (i+1) << ":\n";
        for (int j = 0; j < STATE_DIM; j++) {
            for (int k = 0; k < STATE_DIM; k++) {
                std::cout << big_E[i * STATE_DIM * STATE_DIM + j * STATE_DIM + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------------------" << std::endl;
    }

    // // 打印 bigF
    // std::cout << "==== bigF ====" << std::endl;
    // for (int i = 0; i < 5; i++) {  // 仅展示前 5 次步
    //     std::cout << "bigF[" << (i+1) << "]:\n";
    //     for (int j = 0; j < STATE_DIM; j++) {
    //         for (int k = 0; k < CONTROL_DIM * N; k++) {
    //             std::cout << big_F[i * STATE_DIM * CONTROL_DIM * N + j * CONTROL_DIM + k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "-------------------" << std::endl;
    // }

    for(int i = 0; i < STATE_DIM * N; i++) {
        for(int j = 0; j < CONTROL_DIM * N; j++) {
            printf("%.6f ", big_F[i * (CONTROL_DIM * N) + j]);
        }
        printf("\n");
    }

    delete[] big_E;
    delete[] big_F;
    return 0;
}
