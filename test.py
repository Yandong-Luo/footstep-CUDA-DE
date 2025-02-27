import numpy as np

np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

# 定义矩阵 D (50x1)
D = np.array([0.518370, 0.585255, 3.901861, 5.020326, 0.845929, 0.967649, 1.156522, 
             4.637585, 5.882935, 0.874081, 1.375974, 1.688110, 3.280005, 4.520342, 
             0.886917, 1.594215, 2.038698, 1.036706, 2.486008, 0.907142, 1.591695, 
             2.200290, -0.964606, 0.884318, 0.944536, 1.438071, 2.250700, -1.878772, 
             0.318334, 1.002233, 1.264941, 2.300803, -1.345858, 0.837540, 1.068120, 
             1.207192, 2.436567, 0.306157, 1.885578, 1.110797, 1.324073, 2.655854, 
             1.858718, 2.247999, 1.108310, 1.500000, 2.800000, 1.000000, 0.000000, 
             1.078987]).reshape(50, 1)

# 定义矩阵 E (50x5)
E = np.zeros((50, 5))
E_data = [
    [1.000000, 0.000000, 0.513166, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 0.513166, 0.000000],
    [0.000000, 0.000000, 1.892976, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 1.892976, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 1.484576, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 1.484576, 0.000000],
    [0.000000, 0.000000, 3.583357, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 3.583357, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 3.323433, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 3.323433, 0.000000],
    [0.000000, 0.000000, 6.783209, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 6.783209, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 6.804344, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 6.804344, 0.000000],
    [0.000000, 0.000000, 12.840450, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 12.840450, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 13.393624, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 13.393624, 0.000000],
    [0.000000, 0.000000, 24.306662, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 24.306662, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 25.866974, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 25.866974, 0.000000],
    [0.000000, 0.000000, 46.011921, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 46.011921, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 49.478722, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 49.478722, 0.000000],
    [0.000000, 0.000000, 87.099457, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 87.099457, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 94.175186, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 94.175186, 0.000000],
    [0.000000, 0.000000, 164.877167, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 164.877167, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 178.784515, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 178.784515, 0.000000],
    [0.000000, 0.000000, 312.108490, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 312.108490, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [1.000000, 0.000000, 338.947937, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000, 338.947937, 0.000000],
    [0.000000, 0.000000, 590.813843, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 590.813843, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 1.000000]
]

for i in range(50):
    E[i] = E_data[i]

# Define x_0 (make sure it's shaped correctly)
x_0 = np.array([0.29357406, 0.29125562, -0.01193462, -0.01774755, 1.58432257]).reshape(5, 1)

# Calculate E@x_0
E_x_0 = E @ x_0  # This should be 50x1

# Print shapes to verify
print("D shape:", D.shape)
print("E shape:", E.shape)
print("x_0 shape:", x_0.shape)
print("E@x_0 shape:", E_x_0.shape)

# Calculate D - E@x_0
result = D - E_x_0  # This should be 50x1

print("Result shape:", result.shape)
print("\nD - E@x_0 result:")
print(result)

# # Print row by row
# print("\nRow-by-row results:")
# for i in range(50):
#     print(f"[{i}]: {result[i, 0]:.8f}")