import numpy as np

def construct_E_matrix(E, N):
    """
    生成堆叠矩阵 [E, E^2, ..., E^N]，并打印每一步的矩阵
    :param E: (5x5) 矩阵
    :param N: 最高阶次
    :return: 5N x 5 矩阵
    """
    state_dim = E.shape[0]  # 5
    big_E = np.zeros((state_dim * N, state_dim))  # 初始化大矩阵

    for i in range(N):
        E_power = np.linalg.matrix_power(E, i + 1)  # 计算 E^(i+1)
        big_E[i * state_dim:(i + 1) * state_dim, :] = E_power  # 存入大矩阵

        # 打印当前次方矩阵
        print(f"\nE^{i+1}:")
        print(E_power)

    return big_E

# 定义矩阵 E
E = np.array([
    [1, 0, 0.513166, 0, 0],
    [0, 1, 0, 0.513166, 0],
    [0, 0, 1.89298, 0, 0],
    [0, 0, 0, 1.89298, 0],
    [0, 0, 0, 0, 1]
])

N = 10  # 计算前 10 个步长
big_E = construct_E_matrix(E, N)