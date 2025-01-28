#include "footstep/footstep_utils.cuh"

namespace footstep{
    __constant__ float E[25] = {
        1.0f, 0.0f, __sinf(omega * T)/omega,        0.0f,               0.0f,
        0.0f, 1.0f,                0.0f,     __sinf(omega * T)/omega,   0.0f,
        0.0f, 0.0f,      __cosf(omega * T),         0.0f,               0.0f,
        0.0f, 0.0f,                0.0f,    __cosf(omega * T),          0.0f,
        0.0f, 0.0f,                0.0f,            0.0f,               1.0f
    };

    __constant__ float F[15] = {
        1 - __cosf(omega * T),    0.0f,                       0.0f,
        0.0f,                     1 - __cosf(omega * T),      0.0f,
        -omega * __sinf(omega * T), 0.0f,                     0.0f,
        0.0f,                     -omega * __sinf(omega * T), 0.0f,
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

    // __device__ float H3[350] = {
    //     -lam_max,      0.0f,
    //         0.0f, -lam_max,
    //     d_max,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,    d_max,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f,
    //         0.0f,      0.0f
    // };

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

    // __constant__ float Inx[16] = {
    //     1.0f, 0.0f, 0.0f, 0.0f,
    //     0.0f, 1.0f, 0.0f, 0.0f,
    //     0.0f, 0.0f, 1.0f, 0.0f,
    //     0.0f, 0.0f, 0.0f, 1.0f
    // };

    __constant__ float4 current_state = {0,0,0,0};
    __constant__ float2 current_wall_pos = {0,0};
}
