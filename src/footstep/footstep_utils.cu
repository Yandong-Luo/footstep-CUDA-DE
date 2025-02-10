#include "footstep/footstep_utils.cuh"
namespace footstep{

    __constant__ float init_state[5] = {0.29357406,  0.29125562, -0.01193462, -0.01774755,  1.58432257};

    __constant__ float4 region1 = {0, 1, 0, 3};
    __constant__ float4 region2 = {1, 2, 0, 1};
    __constant__ float4 region3 = {2, 3, 0, 3};
    __constant__ float4 region4 = {1, 2, 2, 3};
    __constant__ float4 region5 = {0, 0.2, 1.4, 1.6};
    __constant__ float4 region6 = {1.4, 1.6, 2.8, 3};
    __constant__ float4 region7 = {2.8, 3, 1.4, 1.6};

    __constant__ float4 all_region[7] = {
        {0, 1, 0, 3},
        {1, 2, 0, 1},
        {2, 3, 0, 3},
        {1, 2, 2, 3},
        {0, 0.2, 1.4, 1.6},
        {1.4, 1.6, 2.8, 3},
        {2.8, 3, 1.4, 1.6}
    };

    // foothold bound circle center
    __constant__ float2 foothold_circles[foothold_circle_num] = {{0.0f, 1.0f}, {0.0f, -2.8f}};   // upper boundary and lower (first step num)
    __constant__ float2 foothold_circles2[foothold_circle_num] = {{0.0f, -1.0f}, {0.0f, 2.8f}};  // upper boundary and lower 
    __constant__ float foothold_radii[foothold_circle_num] = {0.95f, 3.0f};

    // velocity circle
    __constant__ float2 vel_circle[vel_circle_num] = {{-0.1f, 0.0f}, {3.0f, 0.0f}};      // forward and backward
    __constant__ float vel_circle_radii[vel_circle_num] = {0.3f, 3.3f};

    // target
    __constant__ float2 fk = {0.0f, 0.13f};
    __constant__ float2 fk2 = {0.0f,-0.13f};

    // __device__ float E[25] = {
    //     1.0f, 0.0f, 0.513166f,        0.0f,               0.0f,
    //     0.0f, 1.0f,                0.0f,     0.513166f,   0.0f,
    //     0.0f, 0.0f,      1.89298f,         0.0f,               0.0f,
    //     0.0f, 0.0f,                0.0f,    1.89298f,          0.0f,
    //     0.0f, 0.0f,                0.0f,            0.0f,               1.0f
    // };

    // __device__ float F[15] = {
    //     -0.892976f,    0.0f,                       0.0f,
    //     0.0f,                     -0.892976f,      0.0f,
    //     -5.03416f, 0.0f,                     0.0f,
    //     0.0f,                     -5.03416f, 0.0f,
    //     0.0f,                     0.0f,                       1.0f
    // };

    float h_E[25] = {
        1.0f, 0.0f, sinhf(omega * T)/omega,        0.0f,               0.0f,
        0.0f, 1.0f,                0.0f,     sinhf(omega * T)/omega,   0.0f,
        0.0f, 0.0f,      coshf(omega * T),         0.0f,               0.0f,
        0.0f, 0.0f,                0.0f,    coshf(omega * T),          0.0f,
        0.0f, 0.0f,                0.0f,            0.0f,             1.0f
    };

    float h_F[15] = {
        1.0f - coshf(omega * T),    0.0f,                       0.0f,
        0.0f,                     1.0f - coshf(omega * T),      0.0f,
        -omega * sinhf(omega * T), 0.0f,                     0.0f,
        0.0f,                     -omega * sinhf(omega * T), 0.0f,
        0.0f,                     0.0f,                       1.0f
    };

    __constant__ float G[35] = {0.0f};

    __constant__ float Q[25] = {
        0.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        0.0f, 0.0f,  0.0f,  0.0f, 0.0f,
        0.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        0.0f,  0.0f,  0.0f, 0.0f, 0.0f,
        0.0f,  0.0f,  0.0f, 0.0f, 0.0f
    };

    __constant__ float R[9] = {
        0.5f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    __constant__ float h_H1[60] = {
       1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
       0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
       0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
       0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f
   };

    __constant__ float H2[36] = {
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       1.0f, 0.0f, 0.0f,
       -1.0f, 0.0f, 0.0f,
       0.0f, 1.0f, 0.0f,
       0.0f, -1.0f, 0.0f,
       0.0f, 0.0f, 1.0f,
       0.0f, 0.0f, -1.0f
   };

    __constant__ float h[12] = {
        0.5,                // x speed upper boundary
        0.5,                // x speed lower boundary
        0.5,                // y speed upper boundary
        0.5,                // y speed lower boundary
        5 * PI / 4.0f,      // theta upper boundary
        5 * PI / 4.0f,      // theta lower boundary
        0.25,               // u_x upper boundary
        0.25,               // u_x lower boundary
        0.25,               // u_y upper boundary
        0.25,               // u_y lower boundary
        PI / 12.0f,         // u_theta upper boundary
        PI / 12.0f          // u_theta lower boundary
    };

    __constant__ float2 target_pos = {0.5f, 2.0f};

    // __constant__ float Inx[16] = {
    //     1.0f, 0.0f, 0.0f, 0.0f,
    //     0.0f, 1.0f, 0.0f, 0.0f,
    //     0.0f, 0.0f, 1.0f, 0.0f,
    //     0.0f, 0.0f, 0.0f, 1.0f
    // };
    // __managed__ float cluster_N_state[N * CUDA_SOLVER_POP_SIZE * state_dims] = {0.0f};
    float h_cluster_N_state[N * CUDA_SOLVER_POP_SIZE * state_dims] = {0.0f};
    float h_cluster_param[CUDA_SOLVER_POP_SIZE * CUDA_PARAM_MAX_SIZE] = {0.0f};
    float *d_cluster_N_state = nullptr;
    float *d_E = nullptr;  // Device pointer
    float *d_F = nullptr;  // Device pointer
    float *bigE = nullptr;
    float *bigF = nullptr;
    float *h_bigE = nullptr;
    float *h_bigF = nullptr;

    float *d_sol_state = nullptr;
    float *h_sol_state = nullptr;

    float *d_sol_score = nullptr;
    float *h_sol_score = nullptr;

    void ConstructEandF(cudaStream_t stream){

        CHECK_CUDA(cudaMemcpy(d_F, h_F, row_F * col_F * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_E, h_E, row_E * col_E * sizeof(float), cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpyToSymbol(d_F, h_F, row_F * col_F * sizeof(float)));
        // CHECK_CUDA(cudaMemcpyToSymbol(d_E, h_E, row_E * col_E * sizeof(float)));

        // float h_F2[15]; 
        // CHECK_CUDA(cudaMemcpy(h_F2, d_F, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("F matrix:\n");
        // for (int i = 0; i < row_F; i++) {
        //     for (int j = 0; j < col_F; j++) {
        //         printf("%f ", h_F2[i * col_F + j]);
        //     }
        //     printf("\n");
        // }

        // float h_E2[25]; 
        // CHECK_CUDA(cudaMemcpy(h_E2, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("E matrix:\n");
        // for (int i = 0; i < row_E; i++) {
        //     for (int j = 0; j < col_E; j++) {
        //         printf("%f ", h_E2[i * col_E + j]);
        //     }
        //     printf("\n");
        // }
    }

    __global__ void storeResult(float* bigF, float* result_block, int k, int j, int state_dims, int control_dims, int N) {
        int r = threadIdx.x;
        int c = threadIdx.y;
        if (r < state_dims && c < control_dims) {
            int index = (k * state_dims + r) * (control_dims * N) + j * control_dims + c;
            bigF[index] = result_block[r * control_dims + c];
        }
    }


    void ConstructBigEAndF(float *bigE, float *bigF, cublasHandle_t handle, cudaStream_t stream){
        float alpha = 1.0f, beta = 0.0f;
        // float Ek[25] = {0.0f};
        // float Ek_record[N][25] = {0.0f};

        float* d_Ek;
        CHECK_CUDA(cudaMalloc(&d_Ek, row_E * col_E * sizeof(float)));

        float* d_E_power;
        CHECK_CUDA(cudaMalloc(&d_E_power, row_E * col_E * sizeof(float)));

        float *result_block;
        CHECK_CUDA(cudaMalloc(&result_block, row_E * col_F * sizeof(float)));

        // CHECK_CUDA(cudaMemcpyFromSymbol(d_Ek, d_E, sizeof(d_E)));
        CHECK_CUDA(cudaMemcpy(d_Ek, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));
        
        for (int k = 0; k < N; k++) {
            // 复制当前 Ek 到 big_E
            CHECK_CUDA(cudaMemcpy(bigE + k * row_E * col_E, d_Ek, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));

            // 计算 Ek+1 = E * Ek
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        row_E, col_E, col_E,
                        &alpha, d_E, col_E, d_Ek, col_E,
                        &beta, d_Ek, col_E);

            for(int j = 0; j <= k; ++j){
                // power 是公式中E的次方，对应到bigE中进行查找还需要-1
                int power = k - j;
                if(power == 0){
                    CHECK_CUDA(cudaMemcpy(result_block, d_F, row_F * col_F * sizeof(float), cudaMemcpyDeviceToDevice));
                    // CHECK_CUDA(cudaMemcpyFromSymbol(result_block, F, sizeof(F)));

                    // float h_F2[15]; 
                    // CHECK_CUDA(cudaMemcpy(h_F2, result_block, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
                    // printf("F matrix:\n");
                    // for (int i = 0; i < row_F; i++) {
                    //     for (int j = 0; j < col_F; j++) {
                    //         printf("%f ", h_F2[i * col_F + j]);
                    //     }
                    //     printf("\n");
                    // }

                    // float h_E2[25]; 
                    // CHECK_CUDA(cudaMemcpy(h_E2, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToHost));
                    // printf("E matrix:\n");
                    // for (int i = 0; i < row_E; i++) {
                    //     for (int j = 0; j < col_E; j++) {
                    //         printf("%f ", h_E2[i * col_E + j]);
                    //     }
                    //     printf("\n");
                    // }
                }
                else{
                    CHECK_CUDA(cudaMemcpy(d_E_power, bigE + (power - 1) * row_E * col_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            col_F, row_E, col_E,
                            &alpha, d_F, col_F, d_E_power, col_E,
                            &beta, result_block, col_F);
                }
                // float *h_result_block;
                // CHECK_CUDA(cudaHostAlloc(&h_result_block, row_E * col_F * sizeof(float), cudaHostAllocDefault));
                // // cudaMemcpy(h_result_block, result_block, row_E * col_F * sizeof(float), cudaMemcpyDeviceToHost);
                // CHECK_CUDA(cudaMemcpy(h_result_block, result_block, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
                // printf("result_block at k=%d, j=%d:\n", k, j);
                // for (int r = 0; r < state_dims; r++) {
                //     for (int c = 0; c < control_dims; c++) {
                //         printf("%f ", h_result_block[r * control_dims + c]);
                //     }
                //     printf("\n");
                // }
                // cudaFreeHost(h_result_block);

                dim3 threads(state_dims, control_dims);
                storeResult<<<1, threads, 0, stream>>>(bigF, result_block, k, j, state_dims, control_dims, N);

                CHECK_CUDA(cudaStreamSynchronize(stream));
            }
        }
        cudaFree(d_Ek);
        cudaFree(d_E_power);
        cudaFree(result_block);
    }
}
