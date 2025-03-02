#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    Modified example to solve a specific rectangular linear system using cuDSS APIs.
    The system is of the form Ax = b, where:
    - A is a 25x15 sparse rectangular matrix in CSR format
    - b is the right-hand side vector of size 25
    - x is the solution vector of size 15 to be computed
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(b_values_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d); \
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


int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 25x15 rectangular system\n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    // Problem dimensions
    int nrows = 25;          // Number of rows in A
    int ncols = 15;          // Number of columns in A
    int nnz = 75;            // Number of non-zero elements
    int nrhs = 1;            // Number of right-hand sides

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    float *csr_values_h = NULL;
    float *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    float *csr_values_d = NULL;
    float *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side b and solution x */
    csr_offsets_h = (int*)malloc((nrows + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (float*)malloc(nnz * sizeof(float));
    x_values_h = (float*)malloc(ncols * sizeof(float));
    b_values_h = (float*)malloc(nrows * sizeof(float));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for matrix A in CSR format */
    int csr_offsets[] = {0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75};
    int csr_columns[] = {0, 1, 0, 1, 2, 0, 3, 1, 4, 0, 3, 1, 4, 2, 5, 0, 3, 6, 1, 4, 7, 0, 3, 6, 1, 4, 7, 2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14};
    float csr_values[] = {-0.892976, -0.892976, -5.034157, -5.034157, 1.000000, -3.476333, -0.892976, -3.476333, -0.892976, -9.529538, -5.034157, -9.529538, -5.034157, 1.000000, 1.000000, -8.366567, -3.476333, -0.892976, -8.366567, -3.476333, -0.892976, -18.039185, -9.529538, -5.034157, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, -17.623661, -8.366567, -3.476333, -0.892976, -17.623661, -8.366567, -3.476333, -0.892976, -34.147739, -18.039185, -9.529538, -5.034157, -34.147739, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, 1.000000, -35.147114, -17.623661, -8.366567, -3.476333, -0.892976, -35.147114, -17.623661, -8.366567, -3.476333, -0.892976, -64.640846, -34.147739, -18.039185, -9.529538, -5.034157, -64.640846, -34.147739, -18.039185, -9.529538, -5.034157, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
    
    /* Initialize right-hand side vector b */
    float b_values[] = {-0.050405, -0.009804, -0.365249, -0.073890, -1.185696, 
                         -0.087216, 0.051210, 0.100456, 0.904777, -1.555607, 
                         0.087744, 0.519485, 1.767516, 3.947194, -1.578014, 
                         0.684432, 1.680078, 3.823951, 6.861454, -1.529942, 
                         1.366274, 2.746448, 1.040091, 0.431384, -1.584323};

    float A[] = {
        -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -35.147114f, 0.000000f, -64.640846f, 0.000000f, 0.000000f,
        0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, -34.147739f, 0.000000f, 0.000000f, -35.147114f, 0.000000f, -64.640846f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, -34.147739f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f, 0.000000f, -17.623661f, 0.000000f, -34.147739f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f, -8.366567f, 0.000000f, -18.039185f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f, -3.476333f, 0.000000f, -9.529538f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.892976f, 0.000000f, -5.034157f, 0.000000f,
        0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f
    };

    float ATA[] = {
        7414.868652f, 0.000000f, 0.000000f, 3842.274170f, 0.000000f, 0.000000f, 1945.089844f, 0.000000f, 0.000000f, 925.823120f, 0.000000f, 0.000000f, 356.797729f, 0.000000f, 0.000000f,
        0.000000f, 7414.868652f, 0.000000f, 0.000000f, 3842.274170f, 0.000000f, 0.000000f, 1945.089844f, 0.000000f, 0.000000f, 925.823120f, 0.000000f, 0.000000f, 356.797729f, 0.000000f,
        0.000000f, 0.000000f, 5.000000f, 0.000000f, 0.000000f, 4.000000f, 0.000000f, 0.000000f, 3.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 1.000000f,
        3842.274170f, 0.000000f, 0.000000f, 2001.110229f, 0.000000f, 0.000000f, 1015.514465f, 0.000000f, 0.000000f, 484.961121f, 0.000000f, 0.000000f, 187.642593f, 0.000000f, 0.000000f,
        0.000000f, 3842.274170f, 0.000000f, 0.000000f, 2001.110229f, 0.000000f, 0.000000f, 1015.514465f, 0.000000f, 0.000000f, 484.961121f, 0.000000f, 0.000000f, 187.642593f, 0.000000f,
        0.000000f, 0.000000f, 4.000000f, 0.000000f, 0.000000f, 4.000000f, 0.000000f, 0.000000f, 3.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 1.000000f,
        1945.089844f, 0.000000f, 0.000000f, 1015.514465f, 0.000000f, 0.000000f, 524.448730f, 0.000000f, 0.000000f, 252.067535f, 0.000000f, 0.000000f, 98.283234f, 0.000000f, 0.000000f,
        0.000000f, 1945.089844f, 0.000000f, 0.000000f, 1015.514465f, 0.000000f, 0.000000f, 524.448730f, 0.000000f, 0.000000f, 252.067535f, 0.000000f, 0.000000f, 98.283234f, 0.000000f,
        0.000000f, 0.000000f, 3.000000f, 0.000000f, 0.000000f, 3.000000f, 0.000000f, 0.000000f, 3.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 1.000000f,
        925.823120f, 0.000000f, 0.000000f, 484.961121f, 0.000000f, 0.000000f, 252.067535f, 0.000000f, 0.000000f, 129.037140f, 0.000000f, 0.000000f, 51.077477f, 0.000000f, 0.000000f,
        0.000000f, 925.823120f, 0.000000f, 0.000000f, 484.961121f, 0.000000f, 0.000000f, 252.067535f, 0.000000f, 0.000000f, 129.037140f, 0.000000f, 0.000000f, 51.077477f, 0.000000f,
        0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 2.000000f, 0.000000f, 0.000000f, 1.000000f,
        356.797729f, 0.000000f, 0.000000f, 187.642593f, 0.000000f, 0.000000f, 98.283234f, 0.000000f, 0.000000f, 51.077477f, 0.000000f, 0.000000f, 26.140144f, 0.000000f, 0.000000f,
        0.000000f, 356.797729f, 0.000000f, 0.000000f, 187.642593f, 0.000000f, 0.000000f, 98.283234f, 0.000000f, 0.000000f, 51.077477f, 0.000000f, 0.000000f, 26.140144f, 0.000000f,
        0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f
    };

    float AT[] = {
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

    /* Copy the CSR data to the host arrays */
    memcpy(csr_offsets_h, csr_offsets, (nrows + 1) * sizeof(int));
    memcpy(csr_columns_h, csr_columns, nnz * sizeof(int));
    memcpy(csr_values_h, csr_values, nnz * sizeof(float));
    memcpy(b_values_h, b_values, nrows * sizeof(float));

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (nrows + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(float)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrows * sizeof(float)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, ncols * sizeof(float)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (nrows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(float),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrows * sizeof(float),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    /* Print parameters for matrix creation */
    int ldb = nrows, ldx = ncols;
    printf("\n--- cudssMatrixCreateDn Parameters for b ---\n");
    printf("nrows = %d\n", nrows);
    printf("nrhs = %d\n", nrhs);
    printf("ldb = %d\n", ldb);
    printf("dataType = CUDA_R_32F\n");
    printf("layout = CUDSS_LAYOUT_COL_MAJOR\n");
    
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, nrows, nrhs, ldb, b_values_d, CUDA_R_32F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    
    printf("\n--- cudssMatrixCreateDn Parameters for x ---\n");
    printf("ncols = %d\n", ncols);
    printf("nrhs = %d\n", nrhs);
    printf("ldx = %d\n", ldx);
    printf("dataType = CUDA_R_32F\n");
    printf("layout = CUDSS_LAYOUT_COL_MAJOR\n");
    
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, ncols, nrhs, ldx, x_values_d, CUDA_R_32F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_GENERAL;  // General matrix type
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;   // Entire matrix view
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    
    printf("\n--- cudssMatrixCreateCsr Parameters for A ---\n");
    printf("nrows = %d\n", nrows);
    printf("ncols = %d\n", ncols);
    printf("nnz = %d\n", nnz);
    printf("matrixType = CUDSS_MTYPE_GENERAL\n");
    printf("matrixViewType = CUDSS_MVIEW_LOWER\n");
    printf("indexBase = CUDSS_BASE_ZERO\n");
    printf("indexType = CUDA_R_32I\n");
    printf("valueType = CUDA_R_32F\n");
    
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_32F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* Print configurations before solving */
    printf("\n--- Before Execute ---\n");
    printf("Phase 1: CUDSS_PHASE_ANALYSIS\n");
    printf("Phase 2: CUDSS_PHASE_FACTORIZATION\n");
    printf("Phase 3: CUDSS_PHASE_SOLVE\n");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Copy solution from device to host and print it */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, ncols * sizeof(float),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    printf("\nSolution vector x:\n");
    for (int i = 0; i < ncols; i++) {
        printf("x[%d] = %f\n", i, x_values_h[i]);
    }

    /* Release the data allocated on the user side */
    CUDSS_EXAMPLE_FREE;

    printf("\nComputation completed successfully\n");
    return 0;
}