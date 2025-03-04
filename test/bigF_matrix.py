import numpy as np

def construct_big_F(E, F, N):
    """
    计算 bigF 矩阵：
    bigF = [
        [F, 0, 0, ..., 0],
        [E F, F, 0, ..., 0],
        [E^2 F, E F, F, ..., 0],
        ...
        [E^(N-1) F, E^(N-2) F, ..., F]
    ]
    :param E: (5x5) 状态转移矩阵
    :param F: (5x3) 控制矩阵
    :param N: 预测步长
    :return: big_F 矩阵 (5N x 3N)
    """
    state_dim, control_dim = F.shape
    big_F = np.zeros((state_dim * N, control_dim * N))  # 初始化 bigF 矩阵

    for i in range(N):
        for j in range(i + 1):
            E_power = np.linalg.matrix_power(E, i - j) if i != j else np.eye(state_dim)
            big_F[i * state_dim:(i + 1) * state_dim, j * control_dim:(j + 1) * control_dim] = E_power @ F

    return big_F

# 定义矩阵 E 和 F
E = np.array([
    [1, 0, 0.513166, 0, 0],
    [0, 1, 0, 0.513166, 0],
    [0, 0, 1.89298, 0, 0],
    [0, 0, 0, 1.89298, 0],
    [0, 0, 0, 0, 1]
])

F = np.array([
    [-0.892976, -0, 0],
    [-0, -0.892976, 0],
    [-5.03416, -0, 0],
    [-0, -5.03416, 0],
    [0, 0, 1]
])

F_pseudo_inverse = np.linalg.pinv(F)

print("F 矩阵:")
print(F)
print("\nF 矩阵的伪逆:")
print(F_pseudo_inverse)

N = 10  # 计算前 10 步
big_F_python = construct_big_F(E, F, N)

# 保存 big_F 矩阵到 CSV 文件
csv_filename = "big_F_python_output.csv"
np.savetxt(csv_filename, big_F_python, delimiter=",", fmt="%.6f")

print(f"big_F 矩阵已保存到 {csv_filename}")
