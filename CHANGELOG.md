# Changelog
All notable changes to this project will be documented in this file.


## [0.1.1] - 2024-12-13
### Changed
- Yandong Luo: First submit

## [0.1.2] - 2024-12-15
### Changed
- Yandong Luo: The framework from the main function to the warmstart part can run.

## [0.1.3] - 2024-12-16
### Changed
- Yandong Luo: Add the solver_center for paralleling multi-differential evolution solvers.

## [0.1.4] - 2024-12-18
### Changed
- Yandong Luo: Fixed and verified random generation of solutions in constraint space. Completed conversion of data in warm start. But still missing evaluation of warm start.

## [0.1.5] - 2024-12-21
### Changed
- Yandong Luo: Complete the evaluation part, get the best fitness and put it first in the population, and prepare for implementation and evolution

## [0.1.6] - 2024-12-22
### Changed
- Yandong Luo: finish CudaEvolveProcess() function

## [0.1.7] - 2024-12-23
### Changed
- Yandong Luo: finish update parameter part. Still need to sort the parameter when evolve is done.

## [0.1.8] - 2024-12-24
### Changed
- Yandong Luo: finish SortParamBasedBitonic() and BitonicWarpCompare() for sorting the parameter based on fitness.

## [0.1.9] - 2024-12-25
### Changed
- Yandong Luo: Fix blocking issue in update parameter. Add test unit. Fix error in BitonicWarpCompare. Reorganize the whole process and adjust warm start.

## [0.1.10] - 2024-12-25
### Changed
- Yandong Luo: Complete the sort of old param (0-127) and complete the sort test.

## [0.1.11] - 2024-12-25
### Changed
- Yandong Luo: Some issues in random center

## [0.1.12] - 2024-12-25
### Changed
- Yandong Luo: Remove the compilation flag in CMakeList to solve the random center failure problem. Sucessfully verify the parameter matrix.

## [0.1.13] - 2024-12-26
### Changed
- Yandong Luo: The matrix calculation and verification of objective function based on cublas has been completed. It is worth noting that the matrix used to receive the result must be cleared to zero. Otherwise, the result will continue to accumulate.

## [0.1.14] - 2024-12-26
### Changed
- Yandong Luo: Complete and verify all the contents of evaluate calculations, and perform floor() on the integer part.

## [0.1.15] - 2024-12-27
### Changed
- Yandong Luo: Completed a test of a MILP problem. The overall process is correct and the result is correct.

## [0.1.16] - 2024-12-28
### Changed
- Yandong Luo: Early termination is implemented by comparing the fitness values of the top 8 elite individuals with the best fitness from the previous generation.

## [0.1.17] - 2024-12-29
### Analysis
- Yandong Luo: Added nvtx analysis to the solver part and init_solver part.

## [0.1.18] - 2024-12-29
### Changed
- Yandong Luo: Remove all unnecessary implementations and selectively allocate memory space based on debug mode or not. And stop tracking existing qdrep files.

## [0.1.19] - 2024-12-29
### Changed
- Yandong Luo: To optimize the efficiency of host->device, evolve_data was used for memory alignment and multi-stream transmission. However, there was still no significant efficiency improvement. Currently, Nsight shows that the process of host->device is too slow when comparing to the solution of the differential evolution algorithm.

## [0.1.20] - 2024-12-30
### Changed
- Yandong Luo: Configure optimization problem parameters via YAML

## [0.1.21] - 2024-12-30
### Changed
- Yandong Luo: Fixed the error when running multiple tasks. Currently, multiple solving tasks can be automatically generated according to YAML and the solving can be completed.

## [0.1.22] - 2025-1-1
### Changed
- Yandong Luo: Adjust the expression of the matrix for evaluation. Now fill in the yaml in the form of rows.

## [0.1.23] - 2025-1-1
### Changed
- Yandong Luo: Supports solving QP problems

## [0.1.24] - 2025-1-2
### Changed
- Yandong Luo: Based on cuRand to generate the random number. (random_manager)

## [0.1.25] - 2025-1-4
### Changed
- Yandong Luo: Update CMakeList and Add README.md

## [0.1.26] - 2025-1-4
### Changed
- Yandong Luo: Update CMakeList for static CUDA Runtime

## [0.1.27] - 2025-1-7
### Changed
- Yandong Luo: QP and MILP problem solver branch

## [0.2.1] - 2025-1-8
### Changed
- Yandong Luo: Start cart pole system

## [0.2.2] - 2025-1-8
### Changed
- Yandong Luo: Add Cartpole environment and add constant variable/matrix for cuda

## [0.2.3] - 2025-1-9
### Changed
- Yandong Luo: Construct A and C matrix for MPC model. Adjust constant variable and matrix.

## [0.2.4] - 2025-1-11
### Changed
- Yandong Luo: Cart_pole.py file can run cuda based on .so file. CMakeList.txt file has been modified so that can support pybind11

## [0.2.5] - 2025-1-12
### Changed
- Yandong Luo: Add none-linear model for cart pole.

## [0.2.6] - 2025-1-13
### Changed
- Yandong Luo: Fixed "nvlink error   : Undefined reference to"

## [0.2.7] - 2025-1-13
### Changed
- Yandong Luo: Remove duplicate nvtx

## [0.2.8] - 2025-1-13
### Changed
- Yandong Luo: Finished the evaluation of state part. (Unverified)

## [0.2.9] - 2025-1-14
### Changed
- Yandong Luo: Finished the evaluation of state and control part. The entire cart pole can be run. Currently only the dynamics model is available. The corresponding states have not been added yet.

## [0.2.10] - 2025-1-15
### Changed
- Yandong Luo: Add constraint for evaluation.

## [0.2.11] - 2025-1-16
### Changed
- Yandong Luo: Modify some error about the state. Now, the performance almost correct. (Missing warm start)

## [0.2.11] - 2025-1-16
### Changed
- Yandong Luo: Modify the error from cart pole model.

## [0.2.12] - 2025-1-17
### Changed
- Yandong Luo: Increase the population size for the test module. Now it can support (64 and 128)

## [0.2.13] - 2025-1-17
### Changed
- Yandong Luo: Cart pole system using 128 population.

## [0.2.14] - 2025-1-19
### Changed
- Yandong Luo: Adjust some parameter

## [0.2.15] - 2025-1-20
### Changed
- Yandong Luo: Current version support 64, 128, 256 population. Fixed some serious bugs: the boundary of speed

## [0.2.16] - 2025-1-21
### Changed
- Yandong Luo: Update cart_pole_model.md

## [0.2.17] - 2025-1-21
### Changed
- Yandong Luo: insert youtube video to cart_pole_model.md. Set cart_pole.md as README

## [0.2.18] - 2025-1-22
### Changed
- Yandong Luo: Predicts 10 u instead of 1 u. The performance has been significantly improved. Modified the calculation of the contact force

## [0.2.18] - 2025-1-24
### Changed
- Yandong Luo: Rewrite the warm start. Warm start based previous solution and expected control input from model to generative more heuristic guesses. Current version support 512 population.

## [0.3.1] - 2025-1-27
### Changed
- Yandong Luo: Add cublasdx in CMakeList and set up E, F, H1, H2, h for footstep planner

## [0.3.2] - 2025-1-29
### Changed
- Yandong Luo: Finish bigE and bigF matrix. Adjust lots of parameter in E, F. Looks like I can't use __constant__ for E, F anymore. Because cublasSgemm doesn't support __constant__.

## [0.3.3] - 2025-1-31
### Changed
- Yandong Luo: Finish all state = bigE times init_state + bigF times all_u (Unverified)

## [0.3.4] - 2025-2-3
### Changed
- Yandong Luo: Complete the parallel update of different individual states. The main method is to update the states of multiple individuals in parallel, and for each individual state update, cublasdx is also used to complete the parallel matrix multiplication at the kernel function level.

## [0.3.5] - 2025-2-3
### Changed
- Yandong Luo: Almost finish all the evaluation part of footstep planner. Still missing the sum up and verification.

## [0.3.6] - 2025-2-4
### Changed
- Yandong Luo: Finish all the evaluation part of footstep planner. And fixed some error when compile (Unverified).

## [0.3.7] - 2025-2-4
### Changed
- Yandong Luo: All the required pipelines are completed to run, but currently it seems that evaluateModel has significant errors. Replace the warp thread sharing of parameters in evolve with shared memory. This is because the dimension height of the variable is now 90, which exceeds 32.

## [0.3.8] - 2025-2-7
### Changed
- Yandong Luo: All limitations on increasing population and increasing parameter dimensions have been fixed and validated. But now when N=30, the state update in footstep causes the precision of float to be exceeded. This problem has not been solved yet.

## [0.3.9] - 2025-2-7
### Changed
- Yandong Luo: Updated the evaluation of model (each step consider the distance between goal and current position). But now when N=30, the score issue hasn't been solved.

## [0.3.10] - 2025-2-9
### Changed
- Yandong Luo: Fixed the issue from old param sorting. Updated the evaluation of model (Add all constraint into the evalution). I also found the issue from SortOldParamBasedBitonic. The root cause may be that the threads are competing to modify the data, causing param to be modified randomly and out of order during large-scale processing. This can be observed in test/sort_param.cu. This part will be re-implemented based on the thrust library in the future.

## [0.3.11] - 2025-2-10
### Changed
- Yandong Luo: Use thrust to complete the sorting (verified). When N>=25, T=512, there is still a race condition error.

## [0.3.12] - 2025-2-11
### Changed
- Yandong Luo: Modified UpdateParameter2 and DuplicateBestAndReorganize2 to support 1024 population size.

## [0.3.13] - 2025-2-13
### Changed
- Yandong Luo: Modified the error from foothold constraint. Added diversity and param_reset module for the solver to avoid local optimality.

## [0.3.14] - 2025-2-14
### Changed
- Yandong Luo: Adjust the parameter of objective function to reach out a better performance.

## [0.3.15] - 2025-2-18
### Changed
- Yandong Luo: Fixed some race in bitonic sorting method.

## [0.3.16] - 2025-2-22
### Changed
- Yandong Luo: Implement Bezier curve to decode the whole trajectory. Still testing velocity of bezier curve. The velocity part stills wrong. But the position is correct.

## [0.3.17] - 2025-2-22
### Changed
- Yandong Luo: The Bezier curve in the Cartesian coordinate system has been fully verified. Continous to verify the bezier curve in polar coordinate system.

## [0.3.18] - 2025-2-23
### Changed
- Yandong Luo: The Bezier curve in the Polar coordinate system has been fully verified. Continous to finish CUDA-DE + Bazier curve

## [0.3.19] - 2025-2-24
### Changed
- Yandong Luo: Finish and verify the decode function DecodeParameters2State().

## [0.3.20] - 2025-2-25
### Changed
- Yandong Luo: Use Eigen to create column layout bigE and bigF for future batch cudss. Based on cublas_dx to finish matrix.

## [0.3.21] - 2025-2-26
### Changed
- Yandong Luo: The solution part of cudss is completed. But there is an error when running cudss.

## [0.3.22] - 2025-2-27
### Changed
- Yandong Luo: cudss can successfully solve HugeF * U = D

## [0.3.23] - 2025-3-2
### Changed
- Yandong Luo: Using MAGMA solve bigF * u = D. I found that CUDSS requires that the A matrix must be a square matrix, so I gave up using CUDSS.

## [0.3.24] - 2025-3-3
### Changed
- Yandong Luo: Fixed serious errors in the magma batch solver, including errors in boundary conditions.

## [0.3.25] - 2025-3-4
### Changed
- Yandong Luo: The solution of D matrix and u is completed through the diagonal matrix E and F_inv. It is also necessary to verify the D matrix in the future.

## [0.3.26] - 2025-3-4
### Changed
- Yandong Luo: Fixed the error on ConstructMatrixD and lots of issue.

## [0.3.27] - 2025-3-5
### Changed
- Yandong Luo: Start arc-length parametrization for bezier curve. Finished and verify the calculation of arch length. Note: Bezier curves based on lookup cannot be used for arc length calculations because PrepareBinomialandFixedPoint only builds a table for TIMESTEP. However, for arc length, it needs to be built based on ARC_LENGTH_SAMPLES. This can be improved later.

## [0.3.27] - 2025-3-6
### Changed
- Yandong Luo: Finish arc-length parametrization of bezier curve. Now each step is evenly distributed according to the arc length of the curve.

## [0.3.28] - 2025-3-11
### Changed
- Yandong Luo: Totally finish the footstep planner based on bezier curve. just need 0.66s to get a good result. It seems that it is not necessary to reparameterize the arc length of the Bezier curve to get good results. I also keep the arc length parameterization for future complex trajectory planning.