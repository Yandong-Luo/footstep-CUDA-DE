#include "footstep/footstep_utils.cuh"
#include <iostream>
namespace footstep{
    __constant__ float init_state[5] = {0.29357406,  0.29125562, -0.01193462, -0.01774755,  1.58432257};
    __constant__ float2 target_pos = {1.5f, 2.8f};
    __constant__ float goal_state[5] = {1.5f,  2.8f, 0.0f, 0.0f, 0.0f};

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

    // environment 2
    __constant__ float4 all_region2[14] = {
        {-0.1*scale, 0.1*scale, 0.0*scale, 1.0*scale},      // Left vertical bar
        {0.2*scale, 1.1*scale, 0.0*scale, 0.1*scale},      // Bottom horizontal
        {1.2*scale, 1.3*scale, 0.0*scale, 1.0*scale},      // Right vertical
        {0.2*scale, 1.1*scale, 0.9*scale, 1.0*scale},      // Top horizontal
        
        {0.2*scale, 0.9*scale, 0.7*scale, 0.8*scale},      // left center horizontal
        {1.0*scale, 1.1*scale, 0.2*scale, 0.8*scale},      // left center vertical
        
        {1.4*scale, 1.5*scale, 0.6*scale, 0.8*scale},      // center vertical upper
        {1.4*scale, 1.5*scale, 0.3*scale, 0.5*scale},      // center vertical lower
        
        {2.8*scale, 3.0*scale, 0.0*scale, 1.0*scale},      // Right vertical bar
        {1.8*scale, 2.7*scale, 0.0*scale, 0.1*scale},      // Bottom horizontal
        {1.6*scale, 1.7*scale, 0.0*scale, 1.0*scale},      // Left vertical
        {1.8*scale, 2.7*scale, 0.9*scale, 1.0*scale},      // Top horizontal
        {2.0*scale, 2.7*scale, 0.7*scale, 0.8*scale},      // right center horizontal
        {1.8*scale, 1.9*scale, 0.2*scale, 0.8*scale}       // right center vertical
    };

    // foothold bound circle center
    __constant__ float2 foothold_circles[foothold_circle_num] = {{0.0f, 1.0f}, {0.0f, -2.8f}};   // upper boundary and lower (first step num)
    __constant__ float2 foothold_circles2[foothold_circle_num] = {{0.0f, -1.0f}, {0.0f, 2.8f}};  // upper boundary and lower 
    __constant__ float foothold_radii[foothold_circle_num] = {0.95f, 3.0f};

    // velocity circle
    __constant__ float2 vel_circle[vel_circle_num] = {{-0.1f, 0.0f}, {3.0f, 0.0f}};      // forward and backward
    __constant__ float vel_circle_radii[vel_circle_num] = {0.3f, 3.3f};

    // target
    __constant__ float2 obj_circle = {0.0f, 0.13f};
    __constant__ float2 obj_circle2 = {0.0f,-0.13f};

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

    
    // __constant__ float2 target_pos = {2.85f, 0.5f};
    // __constant__ float Inx[16] = {
    //     1.0f, 0.0f, 0.0f, 0.0f,
    //     0.0f, 1.0f, 0.0f, 0.0f,
    //     0.0f, 0.0f, 1.0f, 0.0f,
    //     0.0f, 0.0f, 0.0f, 1.0f
    // };
    // __managed__ float cluster_N_state[N * CUDA_SOLVER_POP_SIZE * state_dims] = {0.0f};
    float *h_cluster_N_state = nullptr;
    float h_cluster_param[CUDA_SOLVER_POP_SIZE * CUDA_PARAM_MAX_SIZE] = {0.0f};
    float *d_cluster_N_state = nullptr;
    float *d_E = nullptr;  // Device pointer
    float *d_F = nullptr;  // Device pointer
    float *bigE = nullptr;
    float *bigF = nullptr;
    float *h_bigE = nullptr;
    float *h_bigF = nullptr;
    float *d_hugeE = nullptr;
    float *h_hugeE = nullptr;
    // float *d_hugeF = nullptr;
    // float *h_hugeF = nullptr;

    float *bigE_column = nullptr;
    float *h_bigE_column = nullptr;
    float *bigF_column = nullptr;
    float *h_bigF_column = nullptr;

    float *d_U = nullptr;
    float *h_U = nullptr;

    float *d_D = nullptr;
    float *h_D = nullptr;

    void **d_batch_D = nullptr;
    void **h_batch_D = nullptr;

    void **d_batch_u = nullptr;
    void **h_batch_u = nullptr;

    void **d_batch_hugeF = nullptr;
    void **h_batch_hugeF = nullptr;

    float *d_sol_state = nullptr;
    float *h_sol_state = nullptr;

    float *d_sol_score = nullptr;
    float *h_sol_score = nullptr;

    int *d_csr_offsets = nullptr, *d_csr_columns = nullptr;
    float *d_csr_values = nullptr;

    int nnz = 0;

    std::vector<int> bigF_csr_row_offsets;
    std::vector<int> bigF_csr_column_indices;
    std::vector<float> bigF_csr_values;

    void **d_batch_csr_offsets = nullptr;
    void **d_batch_csr_columns = nullptr;
    void **d_batch_csr_values = nullptr;
    void **h_batch_csr_offsets = nullptr;
    void **h_batch_csr_columns = nullptr;
    void **h_batch_csr_values = nullptr;

    int* row_array_HugeF = nullptr;
    int* col_array_HugeF = nullptr;
    int* nnz_array = nullptr;
    int* row_array_D = nullptr;
    int* col_array_D = nullptr;
    int* ld_array_D = nullptr;
    int* row_array_U = nullptr;
    int* col_array_U = nullptr;
    int* ld_array_U = nullptr;

    float *bigEx0_col = nullptr;
    float *h_bigEx0_col = nullptr;

    float *d_B = nullptr;
    float *h_B = nullptr;

    float *d_cluster_N_control = nullptr;
    float *h_cluster_N_control = nullptr;

    // void ConstructEandF(cudaStream_t stream){

    //     CHECK_CUDA(cudaMemcpy(d_F, h_F, row_F * col_F * sizeof(float), cudaMemcpyHostToDevice));
    //     CHECK_CUDA(cudaMemcpy(d_E, h_E, row_E * col_E * sizeof(float), cudaMemcpyHostToDevice));
    //     // CHECK_CUDA(cudaMemcpyToSymbol(d_F, h_F, row_F * col_F * sizeof(float)));
    //     // CHECK_CUDA(cudaMemcpyToSymbol(d_E, h_E, row_E * col_E * sizeof(float)));

    //     // float h_F2[15]; 
    //     // CHECK_CUDA(cudaMemcpy(h_F2, d_F, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
    //     // printf("F matrix:\n");
    //     // for (int i = 0; i < row_F; i++) {
    //     //     for (int j = 0; j < col_F; j++) {
    //     //         printf("%f ", h_F2[i * col_F + j]);
    //     //     }
    //     //     printf("\n");
    //     // }

    //     // float h_E2[25]; 
    //     // CHECK_CUDA(cudaMemcpy(h_E2, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToHost));
    //     // printf("E matrix:\n");
    //     // for (int i = 0; i < row_E; i++) {
    //     //     for (int j = 0; j < col_E; j++) {
    //     //         printf("%f ", h_E2[i * col_E + j]);
    //     //     }
    //     //     printf("\n");
    //     // }
    // }

    // __global__ void storeResult(float* bigF, float* result_block, int k, int j, int state_dims, int control_dims, int N) {
    //     int r = threadIdx.x;
    //     int c = threadIdx.y;
    //     if (r < state_dims && c < control_dims) {
    //         int index = (k * state_dims + r) * (control_dims * N) + j * control_dims + c;
    //         bigF[index] = result_block[r * control_dims + c];
    //     }
    // }


    // void ConstructBigEAndF(float *bigE, float *bigF, cublasHandle_t handle, cudaStream_t stream){
    //     float alpha = 1.0f, beta = 0.0f;
    //     // float Ek[25] = {0.0f};
    //     // float Ek_record[N][25] = {0.0f};

    //     float* d_Ek;
    //     CHECK_CUDA(cudaMalloc(&d_Ek, row_E * col_E * sizeof(float)));

    //     float* d_E_power;
    //     CHECK_CUDA(cudaMalloc(&d_E_power, row_E * col_E * sizeof(float)));

    //     float *result_block;
    //     CHECK_CUDA(cudaMalloc(&result_block, row_E * col_F * sizeof(float)));

    //     // CHECK_CUDA(cudaMemcpyFromSymbol(d_Ek, d_E, sizeof(d_E)));
    //     CHECK_CUDA(cudaMemcpy(d_Ek, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));
        
    //     for (int k = 0; k < N; k++) {
    //         // 复制当前 Ek 到 big_E
    //         CHECK_CUDA(cudaMemcpy(bigE + k * row_E * col_E, d_Ek, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));

    //         // 计算 Ek+1 = E * Ek
    //         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                     row_E, col_E, col_E,
    //                     &alpha, d_E, col_E, d_Ek, col_E,
    //                     &beta, d_Ek, col_E);

    //         for(int j = 0; j <= k; ++j){
    //             // power 是公式中E的次方，对应到bigE中进行查找还需要-1
    //             int power = k - j;
    //             if(power == 0){
    //                 CHECK_CUDA(cudaMemcpy(result_block, d_F, row_F * col_F * sizeof(float), cudaMemcpyDeviceToDevice));
    //                 // CHECK_CUDA(cudaMemcpyFromSymbol(result_block, F, sizeof(F)));

    //                 // float h_F2[15]; 
    //                 // CHECK_CUDA(cudaMemcpy(h_F2, result_block, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
    //                 // printf("F matrix:\n");
    //                 // for (int i = 0; i < row_F; i++) {
    //                 //     for (int j = 0; j < col_F; j++) {
    //                 //         printf("%f ", h_F2[i * col_F + j]);
    //                 //     }
    //                 //     printf("\n");
    //                 // }

    //                 // float h_E2[25]; 
    //                 // CHECK_CUDA(cudaMemcpy(h_E2, d_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToHost));
    //                 // printf("E matrix:\n");
    //                 // for (int i = 0; i < row_E; i++) {
    //                 //     for (int j = 0; j < col_E; j++) {
    //                 //         printf("%f ", h_E2[i * col_E + j]);
    //                 //     }
    //                 //     printf("\n");
    //                 // }
    //             }
    //             else{
    //                 CHECK_CUDA(cudaMemcpy(d_E_power, bigE + (power - 1) * row_E * col_E, row_E * col_E * sizeof(float), cudaMemcpyDeviceToDevice));
    //                 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                         col_F, row_E, col_E,
    //                         &alpha, d_F, col_F, d_E_power, col_E,
    //                         &beta, result_block, col_F);
    //             }
    //             // float *h_result_block;
    //             // CHECK_CUDA(cudaHostAlloc(&h_result_block, row_E * col_F * sizeof(float), cudaHostAllocDefault));
    //             // // cudaMemcpy(h_result_block, result_block, row_E * col_F * sizeof(float), cudaMemcpyDeviceToHost);
    //             // CHECK_CUDA(cudaMemcpy(h_result_block, result_block, row_F * col_F * sizeof(float), cudaMemcpyDeviceToHost));
    //             // printf("result_block at k=%d, j=%d:\n", k, j);
    //             // for (int r = 0; r < state_dims; r++) {
    //             //     for (int c = 0; c < control_dims; c++) {
    //             //         printf("%f ", h_result_block[r * control_dims + c]);
    //             //     }
    //             //     printf("\n");
    //             // }
    //             // cudaFreeHost(h_result_block);

    //             dim3 threads(state_dims, control_dims);
    //             storeResult<<<1, threads, 0, stream>>>(bigF, result_block, k, j, state_dims, control_dims, N);

    //             CHECK_CUDA(cudaStreamSynchronize(stream));
    //         }
    //     }
    //     cudaFree(d_Ek);
    //     cudaFree(d_E_power);
    //     cudaFree(result_block);
    // }

    void PrintMatrix(const Eigen::MatrixXf& matrix, const std::string& name) {
        std::cout << name << ":\n[";
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                std::cout << std::fixed << matrix(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "]\n";
    }

    // 打印CSR格式数据 - 完整版
    void PrintCSR(const std::vector<int>& offsets, const std::vector<int>& columns, 
        const std::vector<float>& values, const std::string& name) {
        std::cout << "=== CSR Format for " << name << " ===\n";

        std::cout << "Row offsets (" << offsets.size() << " elements): [";
        for (size_t i = 0; i < offsets.size(); ++i) {
            std::cout << offsets[i];
            if (i < offsets.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "Column indices (" << columns.size() << " elements): [";
        for (size_t i = 0; i < columns.size(); ++i) {
            std::cout << columns[i];
            if (i < columns.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "Values (" << values.size() << " elements): [";
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << std::fixed << values[i];
            if (i < values.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // 构建CSR格式
    void BuildCSRFromMatrix(const Eigen::MatrixXf& matrix, 
        std::vector<int>& row_offsets, 
        std::vector<int>& column_indices, 
        std::vector<float>& values) {
        // 清空输入向量
        row_offsets.clear();
        column_indices.clear();
        values.clear();

        // 为row_offsets预留空间（行数+1）
        row_offsets.reserve(matrix.rows() + 1);
        row_offsets.push_back(0); // 第一个元素总是0

        // 计算非零元素数量和CSR格式
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
            float val = matrix(i, j);
            if (val != 0.0f) {
                column_indices.push_back(j);
                values.push_back(val);
            }
        }
        row_offsets.push_back(values.size()); // 每行结束后的累计非零元素数
        }
    }

    void ComputeBigEAndF_RowMajor(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& E,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& F,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigE,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigF) {
        
        using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        // 计算并存储E的幂
        // 直接从E^1 = E开始
        RowMatrix E_power = E;  
        
        for (int k = 0; k < N; k++) {
            // 将E的当前幂存入bigE (E^(k+1))
            bigE.block(k * row_E, 0, row_E, col_E) = E_power;
            
            // 下一次循环需要计算E^(k+2)
            if (k < N-1) {
                E_power = E * E_power;
            }
        }
        
        // 填充bigF矩阵
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    // 对角线元素为F
                    bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = F;
                } else {
                    // 计算E^(i-j)F
                    int power = i - j;
                    if (power == 1) {
                        bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = E * F;
                    } else {
                        // 从bigE中获取已计算的E^power
                        RowMatrix temp_E_power = bigE.block((power-1) * row_E, 0, row_E, col_E);
                        bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = temp_E_power * F;
                    }
                }
            }
        }
    }

    void ConstructBigEAndFBasedEigen(){
        using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先
        using ColVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        Eigen::Map<RowMatrix> E(h_E, row_E, col_E);
        Eigen::Map<RowMatrix> F(h_F, row_F, col_F);
        
        // 创建bigE和bigF矩阵
        RowMatrix bigE_row(N * row_E, col_E);
        RowMatrix bigF_row(N * state_dims, N * control_dims);
        bigE_row.setZero();
        bigF_row.setZero();
        
        // 构建行优先格式的bigE和bigF
        ComputeBigEAndF_RowMajor(E, F, bigE_row, bigF_row);
        
        // 创建列优先格式的bigE和bigF（通过简单赋值转换存储方式）
        ColMatrix bigE_col = bigE_row;
        ColMatrix bigF_col = bigF_row;

        // ColMatrix bigF_transpose_bigF = bigF_col.transpose() * bigF_col;
        // ColMatrix bigF_T = bigF_col.transpose();

        // PrintMatrix(bigF_T, "bigF_transpose");
        // PrintMatrix(bigF_transpose_bigF, "bigF_transpose_bigF");
        // PrintMatrix(bigF_col, "bigF_col");

        // CHECK_CUDA(cudaMemcpy(h_bigE_column, bigE_col.data(), bigE_col))
        std::memcpy(h_bigE_column, bigE_col.data(), row_bigE * col_bigE * sizeof(float));
        std::memcpy(h_bigF_column, bigF_col.data(), row_bigF * col_bigF * sizeof(float));

        // PrintMatrix(bigF_col, "bigF_col");

        // 为bigF_col构建CSR格式
        BuildCSRFromMatrix(bigF_col, bigF_csr_row_offsets, bigF_csr_column_indices, bigF_csr_values);

        PrintCSR(bigF_csr_row_offsets, bigF_csr_column_indices, bigF_csr_values, "bigF_col");

        // Eigen::Map<ColVector> init_state_vec(h_init_state, row_bigEx0);
        Eigen::Map<ColVector> init_state_vec(const_cast<float*>(h_init_state), row_bigEx0);


        ColVector eigen_bigEx0 = bigE_col * init_state_vec;

        PrintMatrix(eigen_bigEx0, "eigen_bigEx0");

        std::memcpy(h_bigEx0_col, eigen_bigEx0.data(), row_bigEx0 * col_bigEx0 * sizeof(float));
    }

    void SetupCUDSSBatch(){
        // CHECK_CUDA(cudaHostAlloc(&footstep::row_array_HugeF, batch_size * sizeof(float *), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::row_array_HugeF, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::col_array_HugeF, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::nnz_array, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::row_array_D, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::col_array_D, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::ld_array_D, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::row_array_U, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::col_array_U, batch_size * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&footstep::ld_array_U, batch_size * sizeof(int), cudaHostAllocDefault));
        // row_array_HugeF = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // col_array_HugeF = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // nnz_array = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // row_array_D = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // col_array_D = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // ld_array_D = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // row_array_U = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // col_array_U = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));
        // ld_array_U = (int*)malloc(CUDA_SOLVER_POP_SIZE * sizeof(int));

        // batch D
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_D, batch_size * sizeof(float *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_D, batch_size * sizeof(float*)));
        
        // batch U
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_u, batch_size * sizeof(float *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_u, batch_size * sizeof(float*)));

        // batch huge F
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_hugeF, batch_size * sizeof(float *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_hugeF, batch_size * sizeof(float*)));

        nnz = bigF_csr_values.size();
        // csr row offsets
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_csr_offsets, batch_size * sizeof(int *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_csr_offsets, batch_size * sizeof(int*)));

        // csr colum indices
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_csr_columns, batch_size * sizeof(int *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_csr_columns, batch_size * sizeof(int*)));

        // csr value
        CHECK_CUDA(cudaHostAlloc(&footstep::h_batch_csr_values, batch_size * sizeof(float *), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&footstep::d_batch_csr_values, batch_size * sizeof(float*)));

        for (int i = 0; i < batch_size; ++i) {
            row_array_HugeF[i] = footstep::row_bigF;
            col_array_HugeF[i] = footstep::col_bigF;
            nnz_array[i] = footstep::nnz;
            
            row_array_D[i] = footstep::col_D;
            col_array_D[i] = 1;
            ld_array_D[i] = footstep::col_D;
            
            row_array_U[i] = footstep::col_U;
            col_array_U[i] = 1;
            ld_array_U[i] = footstep::col_U;
            
            // batch D
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_D[i], footstep::N * footstep::state_dims * sizeof(float)));  // 分配 GPU 内存

            // batch u
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_u[i], footstep::N * footstep::control_dims * sizeof(float)));  // 分配 GPU 内存

            // batch huge F
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_hugeF[i], footstep::row_bigF * footstep::col_bigF * sizeof(float)));  // 分配 GPU 内存
            CHECK_CUDA(cudaMemcpy(footstep::h_batch_hugeF[i], footstep::h_bigF_column, footstep::row_bigF * footstep::col_bigF * sizeof(float), cudaMemcpyHostToDevice));

            // csr row offsets
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_csr_offsets[i], bigF_csr_row_offsets.size() * sizeof(int)));  // 分配 GPU 内存
            CHECK_CUDA(cudaMemcpy(footstep::h_batch_csr_offsets[i], bigF_csr_row_offsets.data(), bigF_csr_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

            // csr colum indices
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_csr_columns[i], bigF_csr_column_indices.size() * sizeof(int)));  // 分配 GPU 内存
            CHECK_CUDA(cudaMemcpy(footstep::h_batch_csr_columns[i], bigF_csr_column_indices.data(), bigF_csr_column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

            // csr value
            CHECK_CUDA(cudaMalloc((void**)&footstep::h_batch_csr_values[i], bigF_csr_values.size() * sizeof(float)));  // 分配 GPU 内存
            CHECK_CUDA(cudaMemcpy(footstep::h_batch_csr_values[i], bigF_csr_values.data(), bigF_csr_values.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        // batch D
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_D, footstep::h_batch_D, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        // batch u
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_u, footstep::h_batch_u, batch_size* sizeof(float*), cudaMemcpyHostToDevice));
        // batch hugeF
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_hugeF, footstep::h_batch_hugeF, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

        // csr row offsets
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_csr_offsets, footstep::h_batch_csr_offsets, batch_size * sizeof(int*), cudaMemcpyHostToDevice));
        // csr colum indices
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_csr_columns, footstep::h_batch_csr_columns, batch_size * sizeof(int*), cudaMemcpyHostToDevice));
        // csr value
        CHECK_CUDA(cudaMemcpy(footstep::d_batch_csr_values, footstep::h_batch_csr_values, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

        // CHECK_CUDA(cudaMalloc(&d_csr_offsets, bigF_csr_row_offsets.size() * sizeof(int)));
        // CHECK_CUDA(cudaMalloc(&d_csr_columns, bigF_csr_column_indices.size() * sizeof(int)));
        // CHECK_CUDA(cudaMalloc(&d_csr_values, bigF_csr_values.size() * sizeof(float)));

        // CHECK_CUDA(cudaMemcpy(&d_csr_offsets, bigF_csr_row_offsets.data(), bigF_csr_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(&d_csr_columns, bigF_csr_column_indices.data(), bigF_csr_column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(&d_csr_values, bigF_csr_values.data(), bigF_csr_values.size() * sizeof(float), cudaMemcpyHostToDevice));

        // int** results_h = (int**)malloc(batch_size * sizeof(int*));
        // for (int i = 0; i < batch_size; ++i) {
        //     results_h[i] = (int*)malloc(bigF_csr_column_indices.size() * sizeof(int));
            
        //     // 将设备内存中的结果复制到主机内存
        //     CHECK_CUDA(cudaMemcpy(results_h[i], footstep::h_batch_csr_columns[i], 
        //         bigF_csr_column_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
            
        //     // 打印结果
        //     printf("Batch %d results:\n", i);
        //     printf("[");
        //     for (int j = 0; j < bigF_csr_column_indices.size(); ++j) {
        //         for(int k = 0; k < 1;++k){
        //             printf("%d ",results_h[i][k*bigF_csr_column_indices.size() + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("]");
        //     printf("\n");
        //     break;
        // }

        // for (int i = 0; i < batch_size; ++i) {
        //     free(results_h[i]);
        //     break;
        // }
        // free(results_h);
    }
}
