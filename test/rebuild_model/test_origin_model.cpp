#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

class OriginalMLDSystem {
private:
    // 系统参数
    double omega;
    double T;
    
    // 状态向量
    Eigen::VectorXd state;  // [x, y, dx, dy, theta]
    
    // 系统矩阵
    Eigen::MatrixXd E;  // 状态转移矩阵
    Eigen::MatrixXd F;  // 控制矩阵
    Eigen::MatrixXd G;  // 二进制变量矩阵
    
    // 约束矩阵
    Eigen::MatrixXd H1;
    Eigen::MatrixXd H2;
    Eigen::MatrixXd H3;
    Eigen::VectorXd h;
    
    // 维度
    int n_x;
    int n_u;
    int n_delta;
    int n_c;  // 约束数量
    
    // 预计算的常量
    double cosh_omega_T;
    double sinh_omega_T;
    double one_minus_cosh;
    double sinh_over_omega;
    
public:
    // // 构造函数 - 带约束
    // OriginalMLDSystem(double omega_val, double T_val, const Eigen::VectorXd& initial_state,
    //                   const Eigen::MatrixXd& H1_val, const Eigen::MatrixXd& H2_val, 
    //                   const Eigen::MatrixXd& H3_val, const Eigen::VectorXd& h_val)
    //     : omega(omega_val), T(T_val), state(initial_state), 
    //       H1(H1_val), H2(H2_val), H3(H3_val), h(h_val) {
        
    //     // 设置维度
    //     n_x = 5;  // [x, y, dx, dy, theta]
    //     n_u = 3;  // [u_x, u_y, u_theta]
    //     n_delta = H3.cols();  // 基于H3矩阵
    //     n_c = h.size();       // 约束数量
        
    //     // 预计算常量
    //     cosh_omega_T = std::cosh(omega * T);
    //     sinh_omega_T = std::sinh(omega * T);
    //     one_minus_cosh = 1.0 - cosh_omega_T;
    //     sinh_over_omega = sinh_omega_T / omega;
        
    //     // 初始化系统矩阵
    //     initializeMatrices();
    // }
    
    // 简化构造函数 - 无约束
    OriginalMLDSystem(double omega_val, double T_val, const Eigen::VectorXd& initial_state)
        : omega(omega_val), T(T_val), state(initial_state) {
        
        // 设置维度
        n_x = 5;  // [x, y, dx, dy, theta]
        n_u = 3;  // [u_x, u_y, u_theta]
        n_delta = 20; // 问题中给出的默认值
        n_c = 0;      // 无约束
        
        // 预计算常量
        cosh_omega_T = std::cosh(omega * T);
        sinh_omega_T = std::sinh(omega * T);
        one_minus_cosh = 1.0 - cosh_omega_T;
        sinh_over_omega = sinh_omega_T / omega;
        
        // 初始化系统矩阵
        initializeMatrices();
        
        // 初始化空约束矩阵
        H1 = Eigen::MatrixXd(0, n_x);
        H2 = Eigen::MatrixXd(0, n_u);
        H3 = Eigen::MatrixXd(0, n_delta);
        h = Eigen::VectorXd(0);
    }
    
    // 初始化系统矩阵
    void initializeMatrices() {
        // 初始化状态转移矩阵 E
        E = Eigen::MatrixXd::Zero(n_x, n_x);
        E(0, 0) = 1.0;
        E(0, 2) = sinh_over_omega;
        E(1, 1) = 1.0;
        E(1, 3) = sinh_over_omega;
        E(2, 2) = cosh_omega_T;
        E(3, 3) = cosh_omega_T;
        E(4, 4) = 1.0;
        
        // 初始化控制矩阵 F
        F = Eigen::MatrixXd::Zero(n_x, n_u);
        F(0, 0) = one_minus_cosh;
        F(1, 1) = one_minus_cosh;
        F(2, 0) = -omega * sinh_omega_T;
        F(3, 1) = -omega * sinh_omega_T;
        F(4, 2) = 1.0;
        
        // 初始化二进制变量矩阵 G (全零矩阵)
        G = Eigen::MatrixXd::Zero(n_x, n_delta);
    }
    
    // 更新状态函数
    void updateState(const Eigen::VectorXd& control, const Eigen::VectorXd& delta = Eigen::VectorXd()) {
        // 更新状态: x[k+1] = E*x[k] + F*u[k] + G*delta[k]
        state = E * state + F * control;
    }
    
    // 获取当前状态
    Eigen::VectorXd getState() const {
        return state;
    }
    
    // 重置系统
    void reset(const Eigen::VectorXd& initial_state) {
        state = initial_state;
    }
    
    // 获取系统矩阵
    Eigen::MatrixXd getE() const { return E; }
    Eigen::MatrixXd getF() const { return F; }
    Eigen::MatrixXd getG() const { return G; }
    
    // 调试信息
    void printSystemInfo() const {
        std::cout << "原始MLD系统参数:\n";
        std::cout << "  omega = " << omega << ", T = " << T << "\n";
        std::cout << "  cosh(ωT) = " << cosh_omega_T << "\n";
        std::cout << "  sinh(ωT) = " << sinh_omega_T << "\n";
        std::cout << "  sinh(ωT)/ω = " << sinh_over_omega << "\n\n";
        
        std::cout << "状态转移矩阵 E:\n" << E << "\n\n";
        std::cout << "控制矩阵 F:\n" << F << "\n\n";
        std::cout << "二进制变量矩阵 G:\n";
        std::cout << "  [大小为 " << n_x << "x" << n_delta << " 的零矩阵]\n\n";
        
        std::cout << "当前状态 [x, y, dx, dy, θ]:\n" << state << "\n\n";
        
        if (n_c > 0) {
            std::cout << "约束矩阵:\n";
            std::cout << "  H1 (" << H1.rows() << "x" << H1.cols() << ")\n";
            std::cout << "  H2 (" << H2.rows() << "x" << H2.cols() << ")\n";
            std::cout << "  H3 (" << H3.rows() << "x" << H3.cols() << ")\n";
            std::cout << "  h (" << h.size() << "x1)\n\n";
        } else {
            std::cout << "无约束定义。\n\n";
        }
    }
};

// 示例用法
int main() {
    // 物理系统参数
    const double g = 9.81; // 重力加速度
    const double legLength = 1.0; // 腿长
    double T = 0.4; // 时间步长
    double omega = std::sqrt(g / legLength); // 特征频率
    
    // 初始状态 [x, y, dx, dy, theta]
    Eigen::VectorXd initial_state(5);
    initial_state << 0.29357406, 0.29125562, -0.01193462, -0.01774755, 1.58432257;
    
    // 创建无约束的MLD系统
    OriginalMLDSystem system(omega, T, initial_state);
    
    // 打印初始系统信息
    std::cout << "=== 初始系统信息 ===\n";
    system.printSystemInfo();
    
    // 应用一系列控制输入
    std::cout << "\n=== 应用控制输入 ===\n";
    
    // 控制输入 [u_x, u_y, u_theta]
    Eigen::Vector3d control1(0.119504,-0.0237862,0.122125);
    Eigen::Vector3d control2(-0.135927,-0.0297207,0.121111);
    Eigen::Vector3d control3(0.110947,0.0465326,0.118862);
    
    // 应用控制并显示结果
    std::cout << "应用控制1后:\n";
    system.updateState(control1);
    std::cout << "状态: " << system.getState().transpose() << "\n\n";
    
    std::cout << "应用控制2后:\n";
    system.updateState(control2);
    std::cout << "状态: " << system.getState().transpose() << "\n\n";
    
    std::cout << "应用控制3后:\n";
    system.updateState(control3);
    std::cout << "状态: " << system.getState().transpose() << "\n\n";
    
    // 打印最终系统信息
    std::cout << "=== 最终系统信息 ===\n";
    system.printSystemInfo();
    
    return 0;
}