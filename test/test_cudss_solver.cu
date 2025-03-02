#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

#define CUDSS_EXAMPLE_FREE \
    do { \
        for (int i = 0; i < batchCount; i++) { \
            free(csr_offsets_h[i]); \
            free(csr_columns_h[i]); \
            free(csr_values_h[i]); \
            free(x_values_h[i]); \
            free(b_values_h[i]); \
            cudaFree(batch_csr_offsets_h[i]); \
            cudaFree(batch_csr_columns_h[i]); \
            cudaFree(batch_csr_values_h[i]); \
            cudaFree(batch_x_values_h[i]); \
            cudaFree(batch_b_values_h[i]); \
        } \
        cudaFree(batch_csr_offsets_d); \
        cudaFree(batch_csr_columns_d); \
        cudaFree(batch_csr_values_d); \
        cudaFree(batch_b_values_d); \
        cudaFree(batch_x_values_d); \
    } while(0);

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);

    void print_matrix_A(int* csr_offsets, int* csr_columns, float* csr_values, int rows, int cols) {
        printf("\n----- 矩阵A的完整内容 (%dx%d) -----\n", rows, cols);
        
        /* 为完整矩阵分配内存 */
        float* full_matrix = (float*)calloc(rows * cols, sizeof(float));
        if (!full_matrix) {
            printf("为完整矩阵分配内存失败\n");
            return;
        }
        
        /* 将CSR格式转换为完整矩阵 */
        for (int i = 0; i < rows; i++) {
            int start = csr_offsets[i];
            int end = csr_offsets[i + 1];
            
            for (int j = start; j < end; j++) {
                int col = csr_columns[j];
                float val = csr_values[j];
                
                if (col < cols) {
                    full_matrix[i * cols + col] = val;
                }
            }
        }
        
        /* 打印完整矩阵 */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%9.4f ", full_matrix[i * cols + j]);
            }
            printf("\n");
        }
        
        /* 释放完整矩阵内存 */
        free(full_matrix);
    }

int main (int argc, char *argv[]) {
    printf("----------------------------------------------------------\n");
    printf("cuDSS example: solving two real linear systems of size 5x5 and 6x6\n"
        "with symmetric positive-definite matrices \n");
    printf("----------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    int batchCount = 2;
    int N = 5;
    int state_dims = 5;
    int control_dims = 3;
    int row_bigF = N * state_dims;
    int col_bigF = N * control_dims;
    int row_u = N * control_dims;
    int col_u = 1;
    int row_D = N * state_dims;
    int col_D = 1;

    // int batch_row_bigF[batchCount] = {row_bigF};
    // int batch_row_bigF[batchCount] = {row_bigF};
    int batch_row_bigF[batchCount];
    int batch_col_bigF[batchCount];
    int batch_row_u[batchCount];
    int batch_col_u[batchCount];
    int batch_row_D[batchCount];
    int batch_col_D[batchCount];


    int csr_offsets[] = {0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75};
    int csr_columns[] = {0, 1, 0, 1, 2, 0, 3, 1, 4, 0, 3, 1, 4, 2, 5, 0, 3, 6, 1, 4, 7, 0, 3, 6, 1, 4, 7, 2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14};
    float csr_values[] = {-0.892976, -0.892976, -5.034157, -5.034157, 1.000000, -3.476333, -0.892976, -3.476333, -0.892976, -9.529538, -5.034157, -9.529538, -5.034157, 1.000000, 1.000000, -8.366567, -3.476333, -0.892976, -8.366567, -3.476333, -0.892976, -18.039185, -9.529538, -5.034157, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, -17.623661, -8.366567, -3.476333, -0.892976, -17.623661, -8.366567, -3.476333, -0.892976, -34.147739, -18.039185, -9.529538, -5.034157, -34.147739, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, 1.000000, -35.147114, -17.623661, -8.366567, -3.476333, -0.892976, -35.147114, -17.623661, -8.366567, -3.476333, -0.892976, -64.640846, -34.147739, -18.039185, -9.529538, -5.034157, -64.640846, -34.147739, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
    
    float D[] = {
        -0.050405, -0.009804, -0.365249, -0.073890, -1.185696, 
        -0.087216, 0.051210, 0.100456, 0.904777, -1.555607, 
        0.087744, 0.519485, 1.767516, 3.947194, -1.578014, 
        0.684432, 1.680078, 3.823951, 6.861454, -1.529942, 
        1.366274, 2.746448, 1.040091, 0.431384, -1.584323
    };
    int csr_offsets_len = sizeof(csr_offsets) / sizeof(csr_offsets[0]);
    int csr_values_len = sizeof(csr_values) / sizeof(csr_values[0]);
    int csr_columns_len = sizeof(csr_columns) / sizeof(csr_columns[0]);
    printf("nnz:%d csr_columns_len:%d\n", csr_values_len, csr_columns_len);

    
    int n[2]    = {5, 6};
    int nnz[2]  = {csr_values_len, csr_values_len};
    int nrhs[2] = {1, 1};

    int *csr_offsets_h[2] = { NULL };
    int *csr_columns_h[2] = { NULL };
    float *csr_values_h[2] = { NULL };
    float *x_values_h[2] = { NULL }, *b_values_h[2] = { NULL };

    // (intermediate) host arrays with device pointers for the batch
    int *batch_csr_offsets_h[2] = { NULL };
    int *batch_csr_columns_h[2] = { NULL };
    float *batch_csr_values_h[2] = { NULL };
    float *batch_x_values_h[2] = { NULL }, *batch_b_values_h[2] = { NULL };

    // int *batch_row

    void **batch_csr_offsets_d = NULL;
    void **batch_csr_columns_d = NULL;
    void **batch_csr_values_d = NULL;
    void **batch_x_values_d = NULL, **batch_b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
    right-hand side x and solution b*/
    for (int i = 0; i < batchCount; i++) {
        csr_offsets_h[i] = (int*)malloc(sizeof(csr_offsets));
        csr_columns_h[i] = (int*)malloc(sizeof(csr_columns));
        csr_values_h[i] = (float*)malloc(sizeof(csr_values));
        x_values_h[i] = (float*)malloc(row_u * col_u * sizeof(float));
        b_values_h[i] = (float*)malloc(row_D * col_D * sizeof(float));

        memcpy(csr_offsets_h[i], csr_offsets, sizeof(csr_offsets));
        memcpy(csr_columns_h[i], csr_columns, sizeof(csr_columns));
        memcpy(csr_values_h[i], csr_values, sizeof(csr_values));
        memcpy(b_values_h[i], D, sizeof(D));
        memset(x_values_h[i], 0, col_bigF * sizeof(float)); // 初始化解向量为零

        if (!csr_offsets_h[i] || ! csr_columns_h[i] || !csr_values_h[i] ||
            !x_values_h[i] || !b_values_h[i]) {
            printf("Error: host memory allocation failed\n");
            return -1;
        }
    }

    for (int i = 0; i < batchCount; i++) {
        batch_row_bigF[i] = row_bigF;
        batch_col_bigF[i] = col_bigF;
        batch_row_u[i] = row_u;
        batch_col_u[i] = col_u;
        batch_row_D[i] = row_D;
        batch_col_D[i] = col_D;
    }

    for (int i = 0; i < batchCount; i++) {
        /* Allocate device memory for A, x and b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_h[i], csr_offsets_len * sizeof(int)),
            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_h[i], csr_columns_len * sizeof(int)),
            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_h[i], csr_values_len * sizeof(float)),
            "cudaMalloc for csr_values");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_h[i], row_D * col_D * sizeof(float)),
            "cudaMalloc for b_values");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_x_values_h[i], row_u * col_u * sizeof(float)),
            "cudaMalloc for x_values");

         /* Copy host memory to device for A and b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_h[i], csr_offsets, csr_offsets_len * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_h[i], csr_columns, csr_columns_len * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_h[i], csr_values, csr_values_len * sizeof(float),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_h[i], D, row_D * col_D * sizeof(float),
            cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
    }

    /* Allocate device memory for batch pointers of A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_d, batchCount * sizeof(float*)),
        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_d, batchCount * sizeof(float*)),
        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_x_values_d, batchCount * sizeof(float*)),
        "cudaMalloc for x_values");

    /* Copy host batch pointers to device */
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_d, batch_csr_offsets_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_d, batch_csr_columns_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_d, batch_csr_values_h, batchCount * sizeof(float*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_d, batch_b_values_h, batchCount * sizeof(float*),
        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_x_values_d, batch_x_values_h, batchCount * sizeof(float*),
        cudaMemcpyHostToDevice), "cudaMemcpy for x_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as batches of dense matrices). */
    cudssMatrix_t x, b;

    int *nrows = n, *ncols = n;
    int *ldb = ncols, *ldx = nrows;
    // printf("--------- Debug cudssMatrixCreateBatchDn for b ---------\n");
    // printf("batchCount: %d\n", batchCount);
    // for (int i = 0; i < batchCount; i++) {
    //     printf("Batch[%d]: row_D=%d, col_D=%d, ld=%d\n", 
    //         i, batch_row_D[i], batch_col_D[i], batch_row_D[i]);
        
    //     printf("Matrix b content for batch %d:\n", i);
    //     for (int j = 0; j < batch_row_D[i]; j++) {
    //         printf("b[%d] = %f\n", j, b_values_h[i][j]);
    //     }
    // }
    // printf("batch_b_values_d address: %p\n", batch_b_values_d);
    // printf("Data type: CUDA_R_32I, CUDA_R_32F\n");
    // printf("Layout: CUDSS_LAYOUT_COL_MAJOR\n");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchDn(&b, batchCount, batch_row_D, batch_col_D, batch_row_D,
        batch_b_values_d, CUDA_R_32I, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDn for b");

    // printf("--------- Debug cudssMatrixCreateBatchDn for x ---------\n");
    // printf("batchCount: %d\n", batchCount);
    // for (int i = 0; i < batchCount; i++) {
    //     printf("Batch[%d]: row_u=%d, col_u=%d, ld=%d\n", 
    //             i, batch_row_u[i], batch_col_u[i], batch_row_u[i]);
    // }
    // printf("batch_x_values_d address: %p\n", batch_x_values_d);
    // printf("Data type: CUDA_R_32I, CUDA_R_32F\n");
    // printf("Layout: CUDSS_LAYOUT_COL_MAJOR\n");
    // printf("-----------------------------------------------------\n");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchDn(&x, batchCount, batch_row_u, batch_col_u, batch_row_u,
        batch_x_values_d, CUDA_R_32I, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDn for x");

    /* Create a matrix object for the batch of sparse input matrices. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_LOWER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    // printf("--------- Debug cudssMatrixCreateBatchCsr for A ---------\n");
    // printf("batchCount: %d\n", batchCount);
    // for (int i = 0; i < batchCount; i++) {
    //     printf("Batch[%d]: row_bigF=%d, col_bigF=%d, ld=%d\n", 
    //         i, batch_row_bigF[i], batch_col_bigF[i], batch_row_bigF[i]);
    // }
    // printf("batch_csr_offsets_d address: %p\n", batch_csr_offsets_d);
    // printf("batch_csr_columns_d address: %p\n", batch_csr_columns_d);
    // printf("batch_csr_values_d address: %p\n", batch_csr_values_d);
    // printf("Matrix type: %d\n", mtype);
    // printf("Matrix view: %d\n", mview);
    // printf("Index base: %d\n", base);
    // printf("Data type: CUDA_R_32I, CUDA_R_32F\n");

    // printf("\n========= 打印矩阵A的内容 =========\n");
    // for (int i = 0; i < batchCount; i++) {
    //     printf("\n矩阵A (批次 %d):\n", i);
    //     print_matrix_A(csr_offsets_h[i], csr_columns_h[i], csr_values_h[i], 
    //                 batch_row_bigF[i], batch_col_bigF[i]);
    // }
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchCsr(&A, batchCount, batch_row_bigF, batch_col_bigF, nnz,
        batch_csr_offsets_d, NULL, batch_csr_columns_d, batch_csr_values_d,
        CUDA_R_32I, CUDA_R_32F, mtype, mview, base), status, "cudssMatrixCreateBatchCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                        solverData, A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                        solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                        solverData, A, x, b), status, "cudssExecute for solve");

    // /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    // CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    // CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    // CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    // CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    // CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    // CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    int passed = 1;
    printf("\nSolution Vector x:\n");
    for (int j = 0; j < batchCount; j++) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h[j], batch_x_values_h[j],
            nrhs[j] * n[j] * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy for x_values");

        printf("Batch %d:\n", j);
        for (int i = 0; i < row_u; i++) {
            printf("x[%d] = %f\n", i, x_values_h[j][i]);
        }
        printf("\n");
    }

    /* Release the data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}