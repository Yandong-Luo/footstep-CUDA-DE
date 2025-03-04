// src/python_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "diff_evolution_solver/solver.cuh"
#include "footstep/footstep_utils.cuh"

namespace py = pybind11;

PYBIND11_MODULE(DE_cuda_solver, m) {
    // 为模块添加文档
    m.doc() = "CUDA-based Differential Evolution Solver"; 

    // 绑定CudaDiffEvolveSolver类
    py::class_<cudaprocess::CudaDiffEvolveSolver>(m, "Create")
        // 构造函数
        .def(py::init<>())
        
        // 初始化求解器
        .def("init_solver", &cudaprocess::CudaDiffEvolveSolver::InitSolver,
             "Initialize the CUDA solver with specified GPU device",
             py::arg("gpu_device"))
             
        // // 更新状态
        // .def("UpdateCartPoleSystem", [](cudaprocess::CudaDiffEvolveSolver& self, 
        //         py::array_t<float> state, py::array_t<float> wall_state) {
        //     // 确保输入是正确的numpy数组
        //     auto buf = state.request();
        //     auto buf2 = wall_state.request();
        //     if (buf.ndim != 1 || buf.shape[0] != 4) {
        //         throw std::runtime_error("State must be a 1D array with 4 elements");
        //     }
        //     if (buf2.ndim != 1 || buf2.shape[0] != 2) {
        //         throw std::runtime_error("wall position must be a 1D array with 2 elements");
        //     }
        //     self.UpdateCartPoleSystem(static_cast<float*>(buf.ptr), static_cast<float*>(buf2.ptr));
        // }, "Update cart pole state with new state vector")
        
        // // 主求解函数
        .def("Solve", [](cudaprocess::CudaDiffEvolveSolver& self) {
            auto result = self.Solver();
            
            // 创建返回字典
            py::dict solution;
            solution["fitness"] = result.fitness;
            // solution["objective_score"] = result.objective_score;
            // solution["constraint_score"] = result.constraint_score;
            
            // control input (u)转换参数到numpy数组
            auto param_array = py::array_t<float>(footstep::N * footstep::control_dims);
            auto buf = param_array.request();
            float* ptr = static_cast<float*>(buf.ptr);
            std::memcpy(ptr, result.N_control, footstep::N * footstep::control_dims * sizeof(float));
            
            solution["param"] = param_array;

            // state转换参数到numpy数组
            auto state_array = py::array_t<float>((footstep::N + 1)* footstep::state_dims);
            auto state_buf = state_array.request();
            float* state_ptr = static_cast<float*>(state_buf.ptr);
            std::memcpy(state_ptr, result.N_states, (footstep::N + 1)* footstep::state_dims * sizeof(float));
            
            solution["state"] = state_array;
            return solution;
        }, "Run the differential evolution solver and return the solution");
}