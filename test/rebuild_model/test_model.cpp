#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

class MLDSystem {
private:
    // 系统参数
    double omega;
    double T;
    
    // 状态向量
    Eigen::Vector3d state;  // [x, y, theta]
    
    // 初始速度
    Eigen::Vector2d v0;  // [v_x0, v_y0]
    
    // 控制输入历史
    std::vector<Eigen::Vector2d> control_history;  // [u_x, u_y] 历史
    
    // 预计算的矩阵
    Eigen::Matrix3d A;  // 状态转移矩阵
    Eigen::Matrix<double, 3, 3> B;  // 控制输入矩阵
    
    // 预计算的常量
    double cosh_omega_T;
    double sinh_omega_T;
    double one_minus_cosh;
    double sinh_over_omega;
    
public:
    // 构造函数
    MLDSystem(double omega_val, double T_val, const Eigen::Vector3d& initial_state, const Eigen::Vector2d& initial_velocity)
        : omega(omega_val), T(T_val), state(initial_state), v0(initial_velocity) {
        
        // 预计算常量
        cosh_omega_T = std::cosh(omega * T);
        sinh_omega_T = std::sinh(omega * T);
        one_minus_cosh = 1.0 - cosh_omega_T;
        sinh_over_omega = sinh_omega_T / omega;
        
        // 设置状态转移矩阵 A (Identity matrix)
        A.setIdentity();
        
        // 设置控制输入矩阵 B
        B.setZero();
        B(0, 0) = one_minus_cosh;
        B(1, 1) = one_minus_cosh;
        B(2, 2) = 1.0;
    }
    
    // 更新状态函数 - 使用矩阵形式
    void updateState(const Eigen::Vector3d& control) {
        // 添加当前控制到历史记录
        Eigen::Vector2d u_xy(control(0), control(1));
        control_history.push_back(u_xy);
        
        // 计算历史项
        Eigen::Vector3d history_term = Eigen::Vector3d::Zero();
        
        int k = control_history.size() - 1;
        
        // 计算初始速度的影响: sinh(ωT)/ω * cosh^k(ωT) * v0
        double cosh_power_k = std::pow(cosh_omega_T, k);
        Eigen::Vector2d v0_term = cosh_power_k * v0;
        
        // 计算历史控制输入的累积影响
        Eigen::Vector2d sum_term = Eigen::Vector2d::Zero();
        for (int j = 0; j < k; ++j) {
            double cosh_power = std::pow(cosh_omega_T, j);
            sum_term += cosh_power * control_history[k - 1 - j];
        }
        
        // 计算完整的历史项: sinh(ωT)/ω * (cosh^k(ωT) * v0 - ω*sinh(ωT) * Σ...)
        history_term(0) = sinh_over_omega * (v0_term(0) - omega * sinh_omega_T * sum_term(0));
        history_term(1) = sinh_over_omega * (v0_term(1) - omega * sinh_omega_T * sum_term(1));
        // theta没有历史项
        
        // 使用完整的矩阵形式更新状态: x[k+1] = A*x[k] + B*u[k] + history_term
        state = A * state + B * control + history_term;
    }
    
    // 获取当前状态
    Eigen::Vector3d getState() const {
        return state;
    }
    
    // 重置系统
    void reset(const Eigen::Vector3d& initial_state, const Eigen::Vector2d& initial_velocity) {
        state = initial_state;
        v0 = initial_velocity;
        control_history.clear();
    }
    
    // 调试信息
    void printSystemInfo() const {
        std::cout << "System Parameters:\n";
        std::cout << "  omega = " << omega << ", T = " << T << "\n";
        std::cout << "  cosh(ωT) = " << cosh_omega_T << "\n";
        std::cout << "  sinh(ωT) = " << sinh_omega_T << "\n";
        std::cout << "  sinh(ωT)/ω = " << sinh_over_omega << "\n";
        std::cout << "  1 - cosh(ωT) = " << one_minus_cosh << "\n\n";
        
        std::cout << "State Transition Matrix A:\n" << A << "\n\n";
        std::cout << "Control Input Matrix B:\n" << B << "\n\n";
        
        std::cout << "Current State [x, y, θ]:\n" << state << "\n\n";
        std::cout << "Initial Velocity [v_x0, v_y0]:\n" << v0 << "\n\n";
        std::cout << "Control History Size: " << control_history.size() << "\n";
    }
};

// 示例用法
int main() {
    // 系统参数
    // 物理系统参数
    const double g = 9.81; // 重力加速度
    const double legLength = 1.0; // 腿长
    double T = 0.4; // 时间步长
    double omega = std::sqrt(g / legLength); // 特征频率
    
    // 初始状态 [x, y, theta]
    Eigen::Vector3d initial_state(0.29357406, 0.29125562, 1.58432257);
    
    // 初始速度 [v_x0, v_y0]
    Eigen::Vector2d initial_velocity(-0.01193462, -0.01774755);
    
    // 创建MLD系统
    MLDSystem system(omega, T, initial_state, initial_velocity);
    
    // 打印初始系统信息
    std::cout << "=== Initial System Information ===\n";
    system.printSystemInfo();
    
    // 应用一系列控制输入
    std::cout << "\n=== Applying Control Inputs ===\n";
    
    // 控制输入 [u_x, u_y, u_theta]
    // 0.119504,-0.0237862,0.122125
    Eigen::Vector3d control1(0.119504,-0.0237862,0.122125);
    Eigen::Vector3d control2(-0.135927,-0.0297207,0.121111);
    Eigen::Vector3d control3(0.110947,0.0465326,0.118862);
    
    // 应用控制并显示结果
    std::cout << "After control 1:\n";
    system.updateState(control1);
    std::cout << "State: " << system.getState().transpose() << "\n\n";
    
    std::cout << "After control 2:\n";
    system.updateState(control2);
    std::cout << "State: " << system.getState().transpose() << "\n\n";
    
    std::cout << "After control 3:\n";
    system.updateState(control3);
    std::cout << "State: " << system.getState().transpose() << "\n\n";
    
    // 打印最终系统信息
    std::cout << "=== Final System Information ===\n";
    system.printSystemInfo();
    
    return 0;
}