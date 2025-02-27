#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <vector>

// 全局常量
const int state_dims = 5;
const int control_dims = 3;
const int N = 3;
const int row_E = state_dims;
const int col_E = state_dims;
const int row_F = state_dims;
const int col_F = control_dims;

// 打印函数
void PrintMatrix(const Eigen::MatrixXf& matrix, const std::string& name) {
    std::cout << name << ":\n[";
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << matrix(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

// 打印内存布局
void PrintMemoryLayout(const float* data, int size, const std::string& name) {
    std::cout << name << " memory layout: ";
    for (int i = 0; i < size; i++) { // 仅打印前20个元素
        std::cout << data[i] << " ";
    }
    // if (size > 20) std::cout << "...";
    std::cout << std::endl;
}

// 打印CSR格式数据 - 完整版
void PrintCSR(const std::vector<int>& offsets, const std::vector<int>& columns, 
    const std::vector<float>& values, const std::string& name) {
    std::cout << "=== CSR Format for " << name << " ===\n";

    std::cout << "Row offsets (" << offsets.size() << " elements): [";
    for (size_t i = 0; i < offsets.size(); ++i) {
    std::cout << offsets[i];
    if (i < offsets.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Column indices (" << columns.size() << " elements): [";
    for (size_t i = 0; i < columns.size(); ++i) {
    std::cout << columns[i];
    if (i < columns.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Values (" << values.size() << " elements): [";
    for (size_t i = 0; i < values.size(); ++i) {
    std::cout << std::fixed << std::setprecision(6) << values[i];
    if (i < values.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// 构建CSR格式
void BuildCSRFromMatrix(const Eigen::MatrixXf& matrix, 
                        std::vector<int>& row_offsets, 
                        std::vector<int>& column_indices, 
                        std::vector<float>& values) {
    // 清空输入向量
    row_offsets.clear();
    column_indices.clear();
    values.clear();
    
    // 为row_offsets预留空间（行数+1）
    row_offsets.reserve(matrix.rows() + 1);
    row_offsets.push_back(0); // 第一个元素总是0
    
    // 计算非零元素数量和CSR格式
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            float val = matrix(i, j);
            if (val != 0.0f) {
                column_indices.push_back(j);
                values.push_back(val);
            }
        }
        row_offsets.push_back(values.size()); // 每行结束后的累计非零元素数
    }
}

void ConstructBigEAndF(
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& E,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& F,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigE,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& bigF) {
    
    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
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

int main() {
    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; // 默认列优先
    
    // 创建E和F矩阵，使用指定的值
    RowMatrix E(row_E, col_E);
    RowMatrix F(row_F, col_F);
    
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
    
    // 创建行优先格式的bigE和bigF矩阵
    RowMatrix bigE_row(N * row_E, col_E);
    RowMatrix bigF_row(N * state_dims, N * control_dims);
    bigE_row.setZero();
    bigF_row.setZero();
    
    // 构建行优先格式的bigE和bigF
    ConstructBigEAndF(E, F, bigE_row, bigF_row);
    
    // 创建列优先格式的bigE和bigF
    // 方法1: 直接类型转换（改变内存布局）
    ColMatrix bigE_col = bigE_row;
    ColMatrix bigF_col = bigF_row;
    
    // 打印结果
    std::cout << "=== Row Major Format (行优先) ===\n";
    PrintMatrix(bigE_row, "bigE_row");
    PrintMatrix(bigF_row, "bigF_row");
    
    std::cout << "\n=== Column Major Format (列优先) ===\n";
    PrintMatrix(bigE_col, "bigE_col");
    PrintMatrix(bigF_col, "bigF_col");
    
    // 验证内存布局差异
    std::cout << "\n=== Memory Layout Comparison (内存布局比较) ===\n";
    PrintMemoryLayout(bigE_row.data(), bigE_row.size(), "bigE_row");
    PrintMemoryLayout(bigE_col.data(), bigE_col.size(), "bigE_col");
    PrintMemoryLayout(bigF_row.data(), bigF_row.size(), "bigF_row");
    PrintMemoryLayout(bigF_col.data(), bigF_col.size(), "bigF_col");
    
    // 为bigF_col构建CSR格式
    std::vector<int> csr_row_offsets;
    std::vector<int> csr_column_indices;
    std::vector<float> csr_values;
    
    BuildCSRFromMatrix(bigF_col, csr_row_offsets, csr_column_indices, csr_values);
    
    // 打印CSR格式信息
    PrintCSR(csr_row_offsets, csr_column_indices, csr_values, "bigF_col");
    
    // 计算非零元素数量
    std::cout << "\n总非零元素数量: " << csr_values.size() << std::endl;
    std::cout << "矩阵维度: " << bigF_col.rows() << " x " << bigF_col.cols() << std::endl;
    std::cout << "稀疏度: " << 100.0 * (1.0 - static_cast<double>(csr_values.size()) / (bigF_col.rows() * bigF_col.cols())) << "%" << std::endl;
    
    return 0;
}