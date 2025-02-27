#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates usage of cuDSS APIs for solving
    a linear least squares problem with a rectangular matrix E:
                               E^T*E*x = E^T*d,
    where:
        E is the rectangular input matrix (50x5),
        d is the right-hand side vector (50x1),
        x is the solution vector (5x1).
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(d_values_h); \
        free(E_values_h); \
        free(ETd_values_h); \
        free(ETE_values_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(d_values_d); \
        cudaFree(E_values_d); \
        cudaFree(ETd_values_d); \
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

// Function to compute E^T * E
void computeETE(const double* E, int rows, int cols, double* ETE) {
    // Simple CPU implementation to compute E^T * E
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < rows; k++) {
                sum += E[k * cols + i] * E[k * cols + j];
            }
            ETE[i * cols + j] = sum;
        }
    }
}

// Function to compute E^T * d
void computeETd(const double* E, int rows, int cols, const double* d, double* ETd) {
    // Simple CPU implementation to compute E^T * d
    for (int i = 0; i < cols; i++) {
        double sum = 0.0;
        for (int k = 0; k < rows; k++) {
            sum += E[k * cols + i] * d[k];
        }
        ETd[i] = sum;
    }
}

// Function to convert dense matrix to CSR format
void denseToCSR(const double* dense, int rows, int cols, int* offsets, int* columns, double* values, int* nnz) {
    int count = 0;
    offsets[0] = 0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = dense[i * cols + j];
            if (fabs(val) > 1e-10) {
                columns[count] = j;
                values[count] = val;
                count++;
            }
        }
        offsets[i + 1] = count;
    }
    *nnz = count;
}

int main(int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("Linear least squares solver for a 50x5 rectangular matrix\n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    // Matrix dimensions
    int rows = 50;
    int cols = 5;
    int nrhs = 1;
    
    // Initialize matrix E (50x5)
    double E_data[50][5] = {
        {1.000000, 0.000000, 0.513166, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 0.513166, 0.000000},
        {0.000000, 0.000000, 1.892976, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 1.892976, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 1.484576, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 1.484576, 0.000000},
        {0.000000, 0.000000, 3.583357, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 3.583357, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 3.323433, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 3.323433, 0.000000},
        {0.000000, 0.000000, 6.783209, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 6.783209, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 6.804344, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 6.804344, 0.000000},
        {0.000000, 0.000000, 12.840450, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 12.840450, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 13.393624, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 13.393624, 0.000000},
        {0.000000, 0.000000, 24.306662, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 24.306662, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 25.866974, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 25.866974, 0.000000},
        {0.000000, 0.000000, 46.011921, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 46.011921, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 49.478722, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 49.478722, 0.000000},
        {0.000000, 0.000000, 87.099457, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 87.099457, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 94.175186, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 94.175186, 0.000000},
        {0.000000, 0.000000, 164.877167, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 164.877167, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 178.784515, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 178.784515, 0.000000},
        {0.000000, 0.000000, 312.108490, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 312.108490, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000},
        {1.000000, 0.000000, 338.947937, 0.000000, 0.000000},
        {0.000000, 1.000000, 0.000000, 338.947937, 0.000000},
        {0.000000, 0.000000, 590.813843, 0.000000, 0.000000},
        {0.000000, 0.000000, 0.000000, 590.813843, 0.000000},
        {0.000000, 0.000000, 0.000000, 0.000000, 1.000000}
    };

    // Initialize vector D (50x1)
    double d_data[50] = {
        0.518370, 0.585255, 3.901861, 5.020326, 0.845929, 0.967649, 1.156522,
        4.637585, 5.882935, 0.874081, 1.375974, 1.688110, 3.280005, 4.520342,
        0.886917, 1.594215, 2.038698, 1.036706, 2.486008, 0.907142, 1.591695,
        2.200290, -0.964606, 0.884318, 0.944536, 1.438071, 2.250700, -1.878772,
        0.318334, 1.002233, 1.264941, 2.300803, -1.345858, 0.837540, 1.068120,
        1.207192, 2.436567, 0.306157, 1.885578, 1.110797, 1.324073, 2.655854,
        1.858718, 2.247999, 1.108310, 1.500000, 2.800000, 1.000000, 0.000000,
        1.078987
    };

    // Host memory pointers
    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL;
    double *d_values_h = NULL;
    double *E_values_h = NULL;
    double *ETd_values_h = NULL;
    double *ETE_values_h = NULL;

    // Device memory pointers
    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL;
    double *d_values_d = NULL;
    double *E_values_d = NULL;
    double *ETd_values_d = NULL;

    // Allocate host memory
    E_values_h = (double*)malloc(rows * cols * sizeof(double));
    d_values_h = (double*)malloc(rows * sizeof(double));
    ETd_values_h = (double*)malloc(cols * sizeof(double));
    ETE_values_h = (double*)malloc(cols * cols * sizeof(double));
    x_values_h = (double*)malloc(cols * sizeof(double));
    
    // Maximum number of non-zeros for a 5x5 matrix is 25, but we'll allocate more
    int max_nnz = cols * cols;
    csr_offsets_h = (int*)malloc((cols + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(max_nnz * sizeof(int));
    csr_values_h = (double*)malloc(max_nnz * sizeof(double));

    if (!E_values_h || !d_values_h || !ETd_values_h || !ETE_values_h || 
        !x_values_h || !csr_offsets_h || !csr_columns_h || !csr_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    // Copy matrix E and vector d data to flattened arrays
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            E_values_h[i * cols + j] = E_data[i][j];
        }
        d_values_h[i] = d_data[i];
    }

    // Compute E^T * E
    computeETE(E_values_h, rows, cols, ETE_values_h);
    
    // Compute E^T * d
    computeETd(E_values_h, rows, cols, d_values_h, ETd_values_h);

    // Convert E^T * E to CSR format
    int nnz = 0;
    denseToCSR(ETE_values_h, cols, cols, csr_offsets_h, csr_columns_h, csr_values_h, &nnz);

    printf("Matrix E^T * E has %d non-zero elements\n", nnz);

    // Allocate device memory
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (cols + 1) * sizeof(int)),
                       "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                       "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                       "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&ETd_values_d, cols * sizeof(double)),
                       "cudaMalloc for ETd_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, cols * sizeof(double)),
                       "cudaMalloc for x_values");

    // Copy data from host to device
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (cols + 1) * sizeof(int),
                       cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                       cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                       cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(ETd_values_d, ETd_values_h, cols * sizeof(double),
                       cudaMemcpyHostToDevice), "cudaMemcpy for ETd_values");

    // Create a CUDA stream
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Create cuDSS library handle
    cudssHandle_t handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    // Create cuDSS solver configuration and data objects
    cudssConfig_t solverConfig;
    cudssData_t solverData;
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    // Create matrix objects for x and E^T*d
    cudssMatrix_t x, b;
    int64_t nrows_sys = cols, ncols_sys = cols;
    int ldb = ncols_sys, ldx = nrows_sys;
    
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols_sys, nrhs, ldb, ETd_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows_sys, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    // Create matrix object for E^T*E (which is SPD)
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows_sys, ncols_sys, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    // Symbolic factorization
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    // Factorization
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    // Solving
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    // Destroy cuDSS resources
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    // Copy solution from device to host
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, cols * sizeof(double),
                       cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    // Print the solution
    printf("Solution vector x:\n");
    for (int i = 0; i < cols; i++) {
        printf("x[%d] = %1.8f\n", i, x_values_h[i]);
    }

    // Free resources
    CUDSS_EXAMPLE_FREE;

    return 0;
}