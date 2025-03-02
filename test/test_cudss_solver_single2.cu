// #include <magma_v2.h>
// #include <magma_operators.h>
// #include <iostream>
// #include <Eigen/Dense>
// #include <vector>

// int main(int argc, char** argv){
//     int N = 5;
//     int state_dims = 5;
//     int control_dims = 3;
//     int row_bigF = N * state_dims;
//     int col_bigF = N * control_dims;
//     int row_u = N * control_dims;
//     int col_u = 1;
//     int row_D = N * state_dims;
//     int col_D = 1;

//     using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//     using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先

//     float bigF[] = {
//         -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
//         0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f,
//         -35.147114f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f,
//         0.000000f, -35.147114f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f,
//         -64.640846f, 0.000000f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f,
//         0.000000f, -64.640846f, 0.000000f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f,
//         0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f
//     };

//     float D[] = {
//         -0.050405, -0.009804, -0.365249, -0.073890, -1.185696, 
//         -0.087216, 0.051210, 0.100456, 0.904777, -1.555607, 
//         0.087744, 0.519485, 1.767516, 3.947194, -1.578014, 
//         0.684432, 1.680078, 3.823951, 6.861454, -1.529942, 
//         1.366274, 2.746448, 1.040091, 0.431384, -1.584323
//     };

//     // RowMatrix A_row_major(rowMajorMat, )
//     Eigen::Map<RowMatrix> BigFRowMajorMat(bigF, row_bigF, col_bigF);

//     Eigen::Map<RowMatrix> DRowMajorMat(D, row_D, col_D);

//     ColMatrix BigFcolMajorMat = BigFRowMajorMat;

//     ColMatrix DcolMajorMat = DRowMajorMat;


//     magma_init();
//     // //在host端初始化数据集
//     // //...
//     float *d_A = nullptr;
//     float *d_X = nullptr;
//     float *d_D = nullptr;

//     float *h_x = nullptr;

//     cudaMalloc(&d_A, row_bigF*col_bigF * sizeof(float));

//     cudaMalloc(&d_X, row_u*col_u * sizeof(float));

//     cudaMalloc(&d_D, row_D * col_D * sizeof(float));

//     cudaMemcpy(&d_A, BigFcolMajorMat.data(), row_bigF*col_bigF * sizeof(float), cudaMemcpyHostToDevice);

//     cudaMemcpy(&d_D, DRowMajorMat.data(), row_D * col_D * sizeof(float), cudaMemcpyHostToDevice);

//     cudaHostAlloc(&h_x, row_u * col_u * sizeof(float), cudaHostAllocDefault);

//     printf("correct");

//     //根据设备(device)创建队列(queue)
//     int device = 0;
//     magma_queue_t queue;
//     magma_getdevice( &device );
//     magma_queue_create( device, &queue );
//     // //初始化数据，并将数据拷贝到设备端
//     // float *dA, *dX;
//     // magma_dmalloc( &dA, ldda*n ); magma_dmalloc( &dX, lddx*nrhs );
//     // magma_dsetmatrix( n, n, A, lda, dA, ldda, queue );    //magma_dsetmatrix实现host->device copy
//     // magma_dsetmatrix( n, nrhs, X, ldx,dX, lddx, queue );
//     // //使用magma计算接口实现计算
//     magma_int_t ldda = row_bigF;  // A 的 leading dimension
//     magma_int_t lddb = row_D;     // B 的 leading dimension
//     float *dwork;
//     magma_int_t lwork = -1; // 先用 lwork=-1 查询最佳大小
//     cudaMalloc(&dwork, sizeof(float) * lwork);
//     int info;
//     // magma_dgels3_gpu(MagaNoTrans, row_bigF, col_bigF, d_A, row_bigF, d_D, row_D, dwork, , &info );
//     // 调用 MAGMA 计算
//     magma_dgels3_gpu(
//         MagmaNoTrans,    // 矩阵 A 不转置
//         row_bigF, col_bigF, col_D,  // m, n, nrhs
//         d_A, ldda,       // A 矩阵
//         d_D, lddb,       // B 矩阵
//         dwork, lwork,    // 工作区
//         &info
//     );
//     if (info != 0) {
//         std::cerr << "MAGMA dgels3_gpu failed with error code: " << info << std::endl;
//     }
//     // if(info!=0) throw std::exception();

//     cudaMemcpy(&h_x, d_X, row_u * col_u * sizeof(float), cudaMemcpyDeviceToHost);

//     for(int i = 0; i < row_u; ++i){
//         printf("%lf ", h_x[i]);
//     }

//     // //将计算结果拷贝到host端
//     // magma_dgetmatrix( n, nrhs,dX, lddx,X, ldx, queue );   //magma_dgetmatrix实现device->host copy
//     // //此处可进行结果校验等步骤...
//     // //资源释放销毁
//     // magma_queue_destroy( queue );
//     // magma_free( dA );
//     // magma_free( dX );
//     // magma_free_cpu( ipiv );
// }

#include <magma_v2.h>
#include <magma_operators.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <stdlib.h>

int main(int argc, char** argv){
    int N = 5;
    int state_dims = 5;
    int control_dims = 3;
    int row_bigF = N * state_dims;
    int col_bigF = N * control_dims;
    int row_u = N * control_dims;
    int col_u = 1;
    int row_D = N * state_dims;
    int col_D = 1;

    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先

    float bigF[] = {
        -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f,
        -35.147114f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, 0.000000f,
        0.000000f, -35.147114f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, 0.000000f, -0.892976f, 0.000000f,
        -64.640846f, 0.000000f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f, 0.000000f,
        0.000000f, -64.640846f, 0.000000f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -5.034157f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f
    };

    float D[] = {
        -0.050405f, -0.009804f, -0.365249f, -0.073890f, -1.185696f, 
        -0.087216f, 0.051210f, 0.100456f, 0.904777f, -1.555607f, 
        0.087744f, 0.519485f, 1.767516f, 3.947194f, -1.578014f, 
        0.684432f, 1.680078f, 3.823951f, 6.861454f, -1.529942f, 
        1.366274f, 2.746448f, 1.040091f, 0.431384f, -1.584323f
    };

    // 使用 Eigen 映射原始数据
    Eigen::Map<RowMatrix> BigFRowMajorMat(bigF, row_bigF, col_bigF);
    Eigen::Map<RowMatrix> DRowMajorMat(D, row_D, col_D);

    // 转换为列主序格式，以便与 MAGMA 兼容
    ColMatrix BigFColMajorMat = BigFRowMajorMat;
    ColMatrix DColMajorMat = DRowMajorMat;
    
    // 初始化 MAGMA
    magma_init();
    
    // 分配主机内存
    float *h_bigF = nullptr;
    float *h_D = nullptr;
    float *h_x = nullptr;
    
    // 使用 cudaHostAlloc 分配固定内存
    cudaError_t err;
    err = cudaHostAlloc(&h_bigF, row_bigF * col_bigF * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_bigF 内存分配失败: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaHostAlloc(&h_D, row_D * col_D * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_D 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        return 1;
    }
    
    err = cudaHostAlloc(&h_x, col_bigF * col_u * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("h_x 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        return 1;
    }
    
    // 从 Eigen 矩阵复制数据到 host 内存
    memcpy(h_bigF, BigFColMajorMat.data(), row_bigF * col_bigF * sizeof(float));
    memcpy(h_D, DColMajorMat.data(), row_D * col_D * sizeof(float));
    
    printf("数据准备完成\n");
    
    // 分配设备内存
    float *d_A = nullptr;
    float *d_B = nullptr;
    
    err = cudaMalloc((void **)&d_A, row_bigF * col_bigF * sizeof(float));
    if (err != cudaSuccess) {
        printf("d_A 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        return 1;
    }
    
    err = cudaMalloc((void **)&d_B, row_D * col_D * sizeof(float));
    if (err != cudaSuccess) {
        printf("d_B 内存分配失败: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        return 1;
    }
    
    // 创建 MAGMA 队列
    int device = 0;
    magma_queue_t queue;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);
    
    // 将数据传输到 GPU
    err = cudaMemcpy(d_A, h_bigF, row_bigF * col_bigF * sizeof(float), cudaMemcpyHostToDevice);
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
    
    err = cudaMemcpy(d_B, h_D, row_D * col_D * sizeof(float), cudaMemcpyHostToDevice);
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
    
    printf("数据传输至 GPU 完成\n");
    
    // 设置 MAGMA 参数
    magma_int_t ldda = row_bigF;  // A 的主维度
    magma_int_t lddb = row_D;     // B 的主维度
    
    // 查询最佳工作区大小
    float temp_work;
    magma_int_t lwork = -1;  // 查询最佳大小的标志
    magma_int_t info;
    
    magma_sgels_gpu(
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
    float *h_work = (float *)malloc(lwork * sizeof(float));
    
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
    magma_sgels_gpu(
        MagmaNoTrans,      // 不转置 A
        row_bigF, col_bigF, col_D,  // m, n, nrhs
        d_A, ldda,         // A 矩阵及其主维度
        d_B, lddb,         // B 矩阵及其主维度
        h_work, lwork,     // 工作区
        &info              // 错误信息
    );
    
    if (info != 0) {
        printf("MAGMA sgels_gpu 失败，错误码: %d\n", (int)info);
        free(h_work);
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }
    
    printf("线性系统求解成功\n");
    
    // 将解复制到主机（d_B 的前 col_bigF 个元素包含解）
    err = cudaMemcpy(h_x, d_B, col_bigF * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("解向量传输失败: %s\n", cudaGetErrorString(err));
        free(h_work);
        cudaFreeHost(h_bigF);
        cudaFreeHost(h_D);
        cudaFreeHost(h_x);
        cudaFree(d_A);
        cudaFree(d_B);
        magma_queue_destroy(queue);
        magma_finalize();
        return 1;
    }
    
    // 打印解
    printf("解向量:\n");
    for (int i = 0; i < col_bigF; ++i) {
        printf("%f\n", h_x[i]);
    }
    
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