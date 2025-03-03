#include <magma_v2.h>
#include <magma_operators.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <stdlib.h>

// 全局常量
const int state_dims = 5;
const int control_dims = 3;
const int N = 30;
const int row_E = state_dims;
const int col_E = state_dims;
const int row_F = state_dims;
const int col_F = control_dims;
const int batch_size = 3;

void PrintMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    std::cout << name << ":\n[";
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::fixed << matrix(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

void ComputeBigEAndF_RowMajor(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& E,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& F,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigE,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigF) {
    
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

int main(int argc, char** argv){
    // int N = 5;
    int row_bigF = N * state_dims;
    int col_bigF = N * control_dims;
    int row_u = N * control_dims;
    int col_u = 1;
    int row_D = N * state_dims;
    int col_D = 1;

    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先

    // 创建bigE和bigF矩阵
    RowMatrix bigE_row(N * state_dims, state_dims);
    RowMatrix bigF_row(N * state_dims, N * control_dims);

    // 创建E和F矩阵，使用指定的值
    RowMatrix E(state_dims, state_dims);
    RowMatrix F(state_dims, control_dims);
    
    // 设置E矩阵的值
    E << 1, 0, 0.513166, 0, 0,
         0, 1, 0, 0.513166, 0,
         0, 0, 1.89298, 0, 0,
         0, 0, 0, 1.89298, 0,
         0, 0, 0, 0, 1;
    
    // 设置F矩阵的值
    F << -0.892976, -0, 0,
         -0, -0.892976, 0,
         -5.03416, -0, 0,
         -0, -5.03416, 0,
         0, 0, 1;
    
    bigE_row.setZero();
    bigF_row.setZero();

    ComputeBigEAndF_RowMajor(E, F, bigE_row, bigF_row);

    PrintMatrix(bigF_row, "bigF_row");

    double D[] = {
        0.287450, 0.282148, -0.022592, -0.033596, 1.584323,
              0.275856, 0.264908, -0.042766, -0.063596, 1.584323,
              0.253910, 0.232273, -0.080955, -0.120385, 1.584323,
              0.212367, 0.170495, -0.153246, -0.227887, 1.584323,
              0.133726, 0.053552, -0.290091, -0.431384, 1.584323,
              -0.015138, -0.167820, -0.549135, -0.816599, 1.584323,
              -0.296936, -0.586870, -1.039499, -1.545802, 1.584323,
              -0.830371, -1.380123, -1.967746, -2.926166, 1.584323,
              -1.840151, -2.881731, -3.724896, -5.539161, 1.584323,
              -3.751641, -5.724240, -7.051139, -10.485497, 1.584323,
              -7.370045, -11.105039, -13.347636, -19.848795, 1.584323,
              -14.219594, -21.290760, -25.266750, -37.573288, 1.584323,
              -27.185627, -40.572086, -47.829350, -71.125320, 1.584323,
              -51.730019, -77.071182, -90.539803, -134.638519, 1.584323,
              -98.191956, -146.163055, -171.389648, -254.867462, 1.584323,
              -186.143250, -276.952332, -324.436462, -482.457947, 1.584323,
              -352.632965, -524.533264, -614.150391, -913.281189, 1.584323,
              -667.793945, -993.197998, -1162.571777, -1728.819092, 1.584323,
              -1264.386108, -1880.368896, -2200.720215, -3272.612793, 1.584323,
              -2393.720459, -3559.761719, -4165.910156, -6194.976562, 1.584323,
              -4531.522949, -6738.812500, -7885.967285, -11726.941406, 1.584323,
              -8578.332031, -12756.678711, -14927.945312, -22198.816406, 1.584323,
              -16238.842773, -24148.351562, -28258.238281, -42021.820312, 1.584323,
              -30740.007812, -45712.511719, -53492.160156, -79546.289062, 1.584323,
              -58190.355469, -86532.953125, -101259.367188, -150579.203125, 1.584323,
              -110153.195312, -163805.046875, -191681.531250, -285042.781250, 1.584323,
              -208517.625000, -310079.281250, -362848.500000, -539579.125000, 1.584323,
              -394719.093750, -586972.812500, -686863.437500, -1021410.125000, 1.584323,
              -747193.875000, -1111125.625000, -1300215.875000, -1933504.750000, 1.584323,
              -1414420.250000, -2103334.250000, -2461277.250000, -3660077.750000, 1.584323
    };

    // double D[] = {
    //     -0.050405f, -0.009804f, -0.365249f, -0.073890f, -1.185696f, 
    //     -0.087216f, 0.051210f, 0.100456f, 0.904777f, -1.555607f, 
    //     0.087744f, 0.519485f, 1.767516f, 3.947194f, -1.578014f, 
    //     0.684432f, 1.680078f, 3.823951f, 6.861454f, -1.529942f, 
    //     1.366274f, 2.746448f, 1.040091f, 0.431384f, -1.584323f
    // };

    // 使用 Eigen 映射原始数据
    Eigen::Map<RowMatrix> DRowMajorMat(D, row_D, col_D);

    // 转换为列主序格式，以便与 MAGMA 兼容
    ColMatrix BigFColMajorMat = bigF_row;
    ColMatrix DColMajorMat = DRowMajorMat;
    
    // 初始化 MAGMA
    magma_init();
    
    // 分配主机内存
    double *h_bigF = nullptr;
    double *h_D = nullptr;
    double *h_B = nullptr;
    double *h_x = nullptr;
    
    // 使用 cudaHostAlloc 分配固定内存
    cudaError_t err;
    err = cudaHostAlloc(&h_bigF, row_bigF * col_bigF * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_bigF 内存分配失败: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaHostAlloc(&h_D, batch_size * row_D * col_D * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_D 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        return 1;
    }
    
    err = cudaHostAlloc(&h_x, batch_size * col_bigF * col_u * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_x 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        return 1;
    }

    err = cudaHostAlloc(&h_B, row_D * col_D * sizeof(double), cudaHostAllocDefault);
    
    // 从 Eigen 矩阵复制数据到 host 内存
    memcpy(h_bigF, BigFColMajorMat.data(), row_bigF * col_bigF * sizeof(double));
    for(int i = 0 ; i < batch_size; ++i){
        memcpy(h_D + i * row_D * col_D, DColMajorMat.data(), row_D * col_D * sizeof(double));
    }
    memcpy(h_B, DColMajorMat.data(), row_D * col_D * sizeof(double));
    
    printf("数据准备完成\n");
    
    // 分配设备内存
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_D = nullptr;
    
    err = cudaMalloc((void **)&d_A, row_bigF * col_bigF * sizeof(double));
    if (err != cudaSuccess) {
        printf("d_A 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        return 1;
    }
    
    err = cudaMalloc((void **)&d_D, batch_size * row_D * col_D * sizeof(double));
    if (err != cudaSuccess) {
        printf("d_B 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        return 1;
    }
    err = cudaMalloc((void **)&d_B, row_D * col_D * sizeof(double));
    
    // 创建 MAGMA 队列
    int device = 0;
    magma_queue_t queue;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);
    
    // 将数据传输到 GPU
    err = cudaMemcpy(d_A, h_bigF, row_bigF * col_bigF * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据传输到 d_A 失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }
    
    err = cudaMemcpy(d_D, h_D, batch_size * row_D * col_D * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("数据传输到 d_B 失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }

    err = cudaMemcpy(d_B, h_B, row_D * col_D * sizeof(double), cudaMemcpyHostToDevice);
    
    printf("数据传输至 GPU 完成\n");
    
    // 设置 MAGMA 参数
    magma_int_t ldda = row_bigF;  // A 的主维度
    magma_int_t lddb = row_D;     // B 的主维度
    
    // 查询最佳工作区大小
    double temp_work;
    magma_int_t lwork = -1;  // 查询最佳大小的标志
    magma_int_t info;
    
    magma_dgels3_gpu(
        MagmaNoTrans,      // 不转置 A
        row_bigF, col_bigF, col_D,  // m, n, nrhs
        d_A, ldda,         // A 矩阵及其主维度
        d_B, lddb,         // B 矩阵及其主维度
        &temp_work, lwork, // 工作区查询
        &info              // 错误信息
    );
    
    if (info != 0) {
        printf("MAGMA 工作区查询失败，错误码: %d\n", (int)info);
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }
    
    // 分配工作区内存
    lwork = static_cast<magma_int_t>(temp_work);
    double *h_work = (double *)malloc(lwork * sizeof(double));
    
    if (h_work == nullptr) {
        printf("工作区内存分配失败\n");
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }
    
    printf("正在求解最小二乘问题...\n");
    
    // 解决最小二乘问题
    // magma_dgels3_gpu(
    //     MagmaNoTrans,      // 不转置 A
    //     row_bigF, col_bigF, col_D,  // m, n, nrhs
    //     d_A, ldda,         // A 矩阵及其主维度
    //     d_B, lddb,         // B 矩阵及其主维度
    //     h_work, lwork,     // 工作区
    //     &info              // 错误信息
    // );

    // for(int i = 0 ; i < batch_size; ++i){
    //     magma_init();
    //     magma_dgels3_gpu(
    //         MagmaNoTrans,      // 不转置 A
    //         row_bigF, col_bigF, col_D,  // m, n, nrhs
    //         d_A, ldda,         // A 矩阵及其主维度
    //         d_D + i * row_D * col_D, lddb,         // B 矩阵及其主维度
    //         h_work, lwork,     // 工作区
    //         &info              // 错误信息
    //     );

    //     cudaMemcpy(h_x + i * N * control_dims, 
    //         d_D + i * N * state_dims, 
    //         N * control_dims * sizeof(double), 
    //         cudaMemcpyDeviceToHost);
    // }

    // Create a completely isolated environment for each batch

    for (int i = 0; i < batch_size; ++i) {
        
        // Make a fresh copy of A and the current batch's B
        double *d_A_batch = nullptr;
        // double *d_B_batch = nullptr;
        
        cudaMalloc((void **)&d_A_batch, row_bigF * col_bigF * sizeof(double));
        // cudaMalloc((void **)&d_B_batch, row_D * col_D * sizeof(double));
        
        cudaMemcpy(d_A_batch, h_bigF, row_bigF * col_bigF * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_B_batch, h_D + i * row_D * col_D, row_D * col_D * sizeof(double), cudaMemcpyHostToDevice);
        
        // Solve just this batch
        magma_dgels3_gpu(
            MagmaNoTrans,
            row_bigF, col_bigF, col_D,
            d_A_batch, ldda,
            d_D + i * row_D * col_D, lddb,
            h_work, lwork,
            &info
        );
        
        // Copy results
        cudaMemcpy(h_x + i * N * control_dims, d_D + i * row_D * col_D, N * control_dims * sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    // // 将解复制到主机（d_B 的前 col_bigF 个元素包含解）
    // err = cudaMemcpy(h_x, d_B, col_bigF * sizeof(double), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     printf("解向量传输失败: %s\n", cudaGetErrorString(err));
    //     free(h_work);
    //     cudaFreeHost(h_bigF);
    //     cudaFreeHost(h_D);
    //     cudaFreeHost(h_x);
    //     cudaFree(d_A);
    //     cudaFree(d_B);
    //     magma_queue_destroy(queue);
    //     magma_finalize();
    //     return 1;
    // }
    
    // 打印解
    printf("解向量:\n");
    for(int j = 0 ; j < batch_size; ++j){
        printf("batch:%d\n", j);
        for (int i = 0; i < col_bigF; ++i) {
            printf("%f ", h_x[j*col_bigF + i]);
        }
        printf("\n");
    }
    

    // // 计算残差向量和范数
    // // 首先，将原始的 bigF 矩阵和 h_x 解向量相乘
    // Eigen::Map<ColMatrix> xColMajorMat(h_x, col_bigF, 1);
    // ColMatrix residual = BigFColMajorMat * xColMajorMat - DColMajorMat;

    // // 计算残差的范数
    // double norm_diff = residual.norm();
    // printf("\nNorm of difference (should be close to zero): %f\n", norm_diff);
    
    // 清理资源
    free(h_work);
    cudaFreeHost(h_bigF);
    cudaFreeHost(h_D);
    cudaFreeHost(h_x);
    cudaFree(d_A);
    cudaFree(d_B);
    magma_queue_destroy(queue);
    magma_finalize();
    
    return 0;
}


// #include <magma_v2.h>
// #include <magma_operators.h>
// #include <iostream>
// #include <Eigen/Dense>
// #include <vector>
// #include <stdlib.h>

// // 全局常量
// const int state_dims = 5;
// const int control_dims = 3;
// const int N = 30;
// const int row_E = state_dims;
// const int col_E = state_dims;
// const int row_F = state_dims;
// const int col_F = control_dims;


// void PrintMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
//     std::cout << name << ":\n[";
//     for (int i = 0; i < matrix.rows(); ++i) {
//         for (int j = 0; j < matrix.cols(); ++j) {
//             std::cout << std::fixed << matrix(i, j) << " ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "]\n";
// }

// void ComputeBigEAndF_RowMajor(
//     const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& E,
//     const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& F,
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigE,
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigF) {
    
//     using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//     // 计算并存储E的幂
//     // 直接从E^1 = E开始
//     RowMatrix E_power = E;  
    
//     for (int k = 0; k < N; k++) {
//         // 将E的当前幂存入bigE (E^(k+1))
//         bigE.block(k * row_E, 0, row_E, col_E) = E_power;
        
//         // 下一次循环需要计算E^(k+2)
//         if (k < N-1) {
//             E_power = E * E_power;
//         }
//     }
    
//     // 填充bigF矩阵
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j <= i; j++) {
//             if (i == j) {
//                 // 对角线元素为F
//                 bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = F;
//             } else {
//                 // 计算E^(i-j)F
//                 int power = i - j;
//                 if (power == 1) {
//                     bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = E * F;
//                 } else {
//                     // 从bigE中获取已计算的E^power
//                     RowMatrix temp_E_power = bigE.block((power-1) * row_E, 0, row_E, col_E);
//                     bigF.block(i * state_dims, j * control_dims, state_dims, control_dims) = temp_E_power * F;
//                 }
//             }
//         }
//     }
// }

// int main(int argc, char** argv){
//     // int N = 5;
//     int state_dims = 5;
//     int control_dims = 3;
//     int row_bigF = N * state_dims;
//     int col_bigF = N * control_dims;
//     int row_u = N * control_dims;
//     int col_u = 1;
//     int row_D = N * state_dims;
//     int col_D = 1;

//     using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//     using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先

//     // 创建bigE和bigF矩阵
//     RowMatrix bigE_row(N * state_dims, state_dims);
//     RowMatrix bigF_row(N * state_dims, N * control_dims);

//     // 创建E和F矩阵，使用指定的值
//     RowMatrix E(state_dims, state_dims);
//     RowMatrix F(state_dims, control_dims);
    
//     // 设置E矩阵的值
//     E << 1, 0, 0.513166, 0, 0,
//          0, 1, 0, 0.513166, 0,
//          0, 0, 1.89298, 0, 0,
//          0, 0, 0, 1.89298, 0,
//          0, 0, 0, 0, 1;
    
//     // 设置F矩阵的值
//     F << -0.892976, -0, 0,
//          -0, -0.892976, 0,
//          -5.03416, -0, 0,
//          -0, -5.03416, 0,
//          0, 0, 1;
    
//     bigE_row.setZero();
//     bigF_row.setZero();

//     double D[] = {
//         0.287450, 0.282148, -0.022592, -0.033596, 1.584323,
//               0.275856, 0.264908, -0.042766, -0.063596, 1.584323,
//               0.253910, 0.232273, -0.080955, -0.120385, 1.584323,
//               0.212367, 0.170495, -0.153246, -0.227887, 1.584323,
//               0.133726, 0.053552, -0.290091, -0.431384, 1.584323,
//               -0.015138, -0.167820, -0.549135, -0.816599, 1.584323,
//               -0.296936, -0.586870, -1.039499, -1.545802, 1.584323,
//               -0.830371, -1.380123, -1.967746, -2.926166, 1.584323,
//               -1.840151, -2.881731, -3.724896, -5.539161, 1.584323,
//               -3.751641, -5.724240, -7.051139, -10.485497, 1.584323,
//               -7.370045, -11.105039, -13.347636, -19.848795, 1.584323,
//               -14.219594, -21.290760, -25.266750, -37.573288, 1.584323,
//               -27.185627, -40.572086, -47.829350, -71.125320, 1.584323,
//               -51.730019, -77.071182, -90.539803, -134.638519, 1.584323,
//               -98.191956, -146.163055, -171.389648, -254.867462, 1.584323,
//               -186.143250, -276.952332, -324.436462, -482.457947, 1.584323,
//               -352.632965, -524.533264, -614.150391, -913.281189, 1.584323,
//               -667.793945, -993.197998, -1162.571777, -1728.819092, 1.584323,
//               -1264.386108, -1880.368896, -2200.720215, -3272.612793, 1.584323,
//               -2393.720459, -3559.761719, -4165.910156, -6194.976562, 1.584323,
//               -4531.522949, -6738.812500, -7885.967285, -11726.941406, 1.584323,
//               -8578.332031, -12756.678711, -14927.945312, -22198.816406, 1.584323,
//               -16238.842773, -24148.351562, -28258.238281, -42021.820312, 1.584323,
//               -30740.007812, -45712.511719, -53492.160156, -79546.289062, 1.584323,
//               -58190.355469, -86532.953125, -101259.367188, -150579.203125, 1.584323,
//               -110153.195312, -163805.046875, -191681.531250, -285042.781250, 1.584323,
//               -208517.625000, -310079.281250, -362848.500000, -539579.125000, 1.584323,
//               -394719.093750, -586972.812500, -686863.437500, -1021410.125000, 1.584323,
//               -747193.875000, -1111125.625000, -1300215.875000, -1933504.750000, 1.584323,
//               -1414420.250000, -2103334.250000, -2461277.250000, -3660077.750000, 1.584323
//     };

//     ComputeBigEAndF_RowMajor(E, F, bigE_row, bigF_row);

//     PrintMatrix(bigF_row, "bigF_row");

//     // 使用 Eigen 映射原始数据 - 修复映射问题
//     // 修正: 使用适当的Map类型, 明确指定矩阵维度和存储顺序
//     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
//         DRowMajorMat(D, row_D, col_D);

//     // 转换为列主序格式，以便与 MAGMA 兼容
//     ColMatrix BigFColMajorMat = bigF_row;
//     ColMatrix DColMajorMat = DRowMajorMat;
    
//     // 设置截断阈值，用于处理小奇异值
//     double rcond = 1e-10;  // 条件数倒数的阈值

//     // 使用Eigen的SVD求解最小二乘问题
//     Eigen::BDCSVD<ColMatrix> svd(BigFColMajorMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
//     // 获取奇异值并打印
//     auto singularValues = svd.singularValues();
//     std::cout << "前10个奇异值:" << std::endl;
//     for (int i = 0; i < std::min(10, (int)singularValues.size()); ++i) {
//         std::cout << singularValues(i) << std::endl;
//     }
    
//     // 计算条件数
//     if (singularValues.size() > 0) {
//         double cond = singularValues(0) / singularValues(singularValues.size() - 1);
//         std::cout << "矩阵条件数: " << cond << std::endl;
//     }

//     // 使用SVD求解线性最小二乘问题
//     // 设置小于阈值的奇异值为0，以提高数值稳定性
//     ColMatrix solution = svd.solve(DColMajorMat);
    
//     // 打印解向量
//     std::cout << "解向量:" << std::endl;
//     for (int i = 0; i < col_bigF; ++i) {
//         std::cout << solution(i) << std::endl;
//     }

//     // 计算残差向量和范数
//     ColMatrix residual = BigFColMajorMat * solution - DColMajorMat;
//     double norm_diff = residual.norm();
//     std::cout << "\nNorm of difference (should be close to zero): " << norm_diff << std::endl;

//     // 对比解决方案：使用 MAGMA 实现
//     // 初始化 MAGMA
//     magma_init();
    
//     // 创建 MAGMA 队列
//     int device = 0;
//     magma_queue_t queue;
//     magma_getdevice(&device);
//     magma_queue_create(device, &queue);
    
//     // 分配设备内存和主机内存
//     double *d_A = nullptr;
//     double *d_B = nullptr;
//     magma_int_t m = row_bigF;
//     magma_int_t n = col_bigF;
//     magma_int_t nrhs = 1;  // 右侧向量的数量
//     magma_int_t ldda = m;  // A 的主维度
//     magma_int_t lddb = m;  // B 的主维度
    
//     // 将Eigen矩阵转换为数组
//     double *h_A = (double*)malloc(m * n * sizeof(double));
//     double *h_B = (double*)malloc(m * nrhs * sizeof(double));
    
//     // 复制数据
//     memcpy(h_A, BigFColMajorMat.data(), m * n * sizeof(double));
//     memcpy(h_B, DColMajorMat.data(), m * nrhs * sizeof(double));
    
//     // 分配设备内存
//     magma_dmalloc(&d_A, m * n);
//     magma_dmalloc(&d_B, m * nrhs);
    
//     // 将数据传输到设备 - 修正：添加队列参数
//     magma_dsetmatrix(m, n, h_A, m, d_A, ldda, queue);
//     magma_dsetmatrix(m, nrhs, h_B, m, d_B, lddb, queue);
    
//     // 设置工作区
//     magma_int_t lwork = -1;
//     double tmp_work[1];
//     magma_int_t info;
    
//     // 查询工作区大小
//     magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, d_A, ldda, d_B, lddb, tmp_work, lwork, &info);
    
//     lwork = (magma_int_t)tmp_work[0];
//     double *h_work = (double*)malloc(lwork * sizeof(double));
    
//     // 解决最小二乘问题
//     magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, d_A, ldda, d_B, lddb, h_work, lwork, &info);
    
//     if (info != 0) {
//         std::cerr << "MAGMA dgels_gpu 失败，错误码: " << info << std::endl;
//     } else {
//         // 获取解向量 - 修正：添加队列参数
//         double *h_X = (double*)malloc(n * sizeof(double));
//         magma_dgetmatrix(n, nrhs, d_B, lddb, h_X, n, queue);
        
//         // 打印解向量
//         std::cout << "\nMAGMA 解向量:" << std::endl;
//         for (int i = 0; i < n; ++i) {
//             std::cout << h_X[i] << std::endl;
//         }
        
//         // 计算残差向量和范数
//         // 修正: 使用更明确的Map类型
//         Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> magmaSolution(h_X, n);
//         ColMatrix magmaResidual = BigFColMajorMat * magmaSolution - DColMajorMat;
//         double magmaNormDiff = magmaResidual.norm();
//         std::cout << "\nMAGMA Norm of difference: " << magmaNormDiff << std::endl;
        
//         free(h_X);
//     }
    
//     // 清理资源
//     free(h_work);
//     free(h_A);
//     free(h_B);
//     magma_free(d_A);
//     magma_free(d_B);
//     magma_queue_destroy(queue);
//     magma_finalize();
    
//     return 0;
// }