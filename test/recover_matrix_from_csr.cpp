#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <iomanip>

// 定义矩阵大小
const int ROWS = 15;  // 矩阵行数
const int COLS = 9;   // 矩阵列数

void ConstructSparseMatrix(Eigen::SparseMatrix<float>& mat,
                           const std::vector<int>& rowStart,
                           const std::vector<int>& colIndices,
                           const std::vector<float>& values) {
    std::vector<Eigen::Triplet<float>> triplets;

    // 解析 CSR 数据并插入非零元素
    for (size_t row = 0; row < rowStart.size() - 1; ++row) {
        for (int j = rowStart[row]; j < rowStart[row + 1]; ++j) {
            triplets.emplace_back(row, colIndices[j], values[j]);
        }
    }

    // 设定稀疏矩阵尺寸
    mat.resize(ROWS, COLS);

    // 构造稀疏矩阵
    mat.setFromTriplets(triplets.begin(), triplets.end());
}

// **打印稀疏矩阵**
void PrintSparseMatrix(const Eigen::SparseMatrix<float>& mat) {
    std::cout << "=== Sparse Matrix (Restored) ===\n";
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << mat.coeff(i, j) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // **提供的 CSR 数据**
    std::vector<int> rowStart = {0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30};
    std::vector<int> colIndices = {0, 1, 0, 1, 2, 0, 3, 1, 4, 0, 3, 1, 4, 2, 5, 0, 3, 6, 1, 4, 7, 0, 3, 6, 1, 4, 7, 2, 5, 8};
    std::vector<float> values = {-0.892976, -0.892976, -5.034160, -5.034160, 1.000000, -3.476336, 
                                  -0.892976, -3.476336, -0.892976, -9.529564, -5.034160, -9.529564, 
                                  -5.034160, 1.000000, 1.000000, -8.366585, -3.476336, -0.892976, 
                                  -8.366585, -3.476336, -0.892976, -18.039274, -9.529564, -5.034160, 
                                  -18.039274, -9.529564, -5.034160, 1.000000, 1.000000, 1.000000};

    // **创建 Eigen::SparseMatrix**
    Eigen::SparseMatrix<float> sparseMatrix;

    // **从 CSR 数据构造稀疏矩阵**
    ConstructSparseMatrix(sparseMatrix, rowStart, colIndices, values);

    // **打印结果**
    PrintSparseMatrix(sparseMatrix);

    return 0;
}
