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
        # print(f"\nE^{i+1}:")
        # print(E_power)

    return big_E

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

N = 10  # 计算前 10 步
big_F_python = construct_big_F(E, F, N)

# N = 10  # 计算前 10 个步长
big_E_python = construct_E_matrix(E, N)

# print(big_E_python)
print(big_F_python)

x0_MLD = np.array([ 0.29357406,  0.29125562, -0.0, -0.0,  1.58432257])

print(np.dot(E, x0_MLD))

# u = np.array([
#     [-0.05876903, -0.11210947,  0.10821661],
#     [ 0.03893148,  0.05902505,  0.13331123],
#     [ 0.05507302,  0.17836939,  0.03928813],
#     [-0.05754665,  0.03220638, -0.0808465 ],
#     [ 0.11146203,  0.07931738, -0.19619025],
#     [ 0.01171406,  0.16665864,  0.14603116],
#     [-0.07849471,  0.08309314,  0.00424856],
#     [-0.07917739,  0.21530892,  0.12213598],
#     [-0.00427725,  0.16955128, -0.08483312],
#     [ 0.03235876, -0.16870178, -0.19042563]
# ])
# u = np.array([
#     0.014494, 0.011752, 0.106959, -0.016315, -0.118803, -0.048978, -0.043494, 
#     0.113994, 0.205796, 0.030347, 0.061138, -0.111982, 0.007480, 0.023217, 
#     0.105595, -0.041712, -0.028752, -0.013986, -0.142971, 0.014229, 0.052919, 
#     0.065996, 0.027053, -0.201575, -0.014427, -0.030627, -0.131165, 0.001109, 
#     0.102250, -0.091096
# ])
# u = u.reshape(N, 3)
# u = u.reshape(-1)
# print(big_E)
# print(big_F_python)
# print("result")
# print(big_E @ x0_MLD + big_F_python @ u)