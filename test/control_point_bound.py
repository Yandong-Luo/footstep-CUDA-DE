import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 已知控制点
P0 = np.array([0.293574, 0.291256])
P1 = np.array([0.291585, 0.288298])
P6 = np.array([1.45, 2.8])
P7 = np.array([1.5, 2.8])

# 时间点
t_values = np.linspace(0, 1, 31)  # 0, 1/30, 2/30, ..., 1

# 设置速度限制
vx_min, vx_max = -0.5, 0.5
vy_min, vy_max = -0.5, 0.5

# 贝塞尔导数系数矩阵
def bernstein_derivative_coef(t, n=7):
    coefs = np.zeros((len(t), n))
    for i in range(len(t)):
        ti = t[i]
        # 计算每个基函数在时间ti的值
        b0 = (1-ti)**6
        b1 = 6*ti*(1-ti)**5
        b2 = 15*ti**2*(1-ti)**4
        b3 = 20*ti**3*(1-ti)**3
        b4 = 15*ti**4*(1-ti)**2
        b5 = 6*ti**5*(1-ti)
        b6 = ti**6
        coefs[i] = [b0, b1, b2, b3, b4, b5, b6]
    return 7 * coefs

# 生成约束条件
def generate_constraints(t_values):
    coefs = bernstein_derivative_coef(t_values)
    num_t = len(t_values)
    
    # 为x和y各设置2*num_t个约束（上下界）
    A_ub = np.zeros((4*num_t, 8))  # 4个未知控制点，每个点2个坐标
    b_ub = np.zeros(4*num_t)
    
    # 填充x方向的约束
    for i in range(num_t):
        # 计算已知点的贡献
        known_contrib_x = 7*((P1[0] - P0[0])*coefs[i,0] + (P7[0] - P6[0])*coefs[i,6])
        
        # vx ≤ vx_max 约束
        A_ub[i, 0:8:2] = [coefs[i,1], coefs[i,2], coefs[i,3], coefs[i,4]]  # P2-P5的x坐标
        b_ub[i] = vx_max - known_contrib_x
        
        # vx ≥ vx_min 约束
        A_ub[num_t+i, 0:8:2] = [-coefs[i,1], -coefs[i,2], -coefs[i,3], -coefs[i,4]]
        b_ub[num_t+i] = -vx_min + known_contrib_x
    
    # 填充y方向的约束（类似于x方向）
    for i in range(num_t):
        known_contrib_y = 7*((P1[1] - P0[1])*coefs[i,0] + (P7[1] - P6[1])*coefs[i,6])
        
        # vy ≤ vy_max 约束
        A_ub[2*num_t+i, 1:8:2] = [coefs[i,1], coefs[i,2], coefs[i,3], coefs[i,4]]  # P2-P5的y坐标
        b_ub[2*num_t+i] = vy_max - known_contrib_y
        
        # vy ≥ vy_min 约束
        A_ub[3*num_t+i, 1:8:2] = [-coefs[i,1], -coefs[i,2], -coefs[i,3], -coefs[i,4]]
        b_ub[3*num_t+i] = -vy_min + known_contrib_y
    
    return A_ub, b_ub

# 目标函数（可以根据需要修改，例如最小化控制点的总距离）
c = np.ones(8)

# 求解线性规划问题
A_ub, b_ub = generate_constraints(t_values)
result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

if result.success:
    # 提取解
    P2 = np.array([result.x[0], result.x[1]])
    P3 = np.array([result.x[2], result.x[3]])
    P4 = np.array([result.x[4], result.x[5]])
    P5 = np.array([result.x[6], result.x[7]])
    
    print("解得的控制点:")
    print(f"P2 = {P2}")
    print(f"P3 = {P3}")
    print(f"P4 = {P4}")
    print(f"P5 = {P5}")
else:
    print("未找到满足约束的解")