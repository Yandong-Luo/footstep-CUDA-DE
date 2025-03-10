import numpy as np

# Define matrix E (5x5)
E = np.array([
    [1, 0, 0.513166, 0, 0],
    [0, 1, 0, 0.513166, 0],
    [0, 0, 1.89298, 0, 0],
    [0, 0, 0, 1.89298, 0],
    [0, 0, 0, 0, 1]
])

# Define matrix F (5x3)
F = np.array([
    [-0.892976, 0, 0],
    [0, -0.892976, 0],
    [-5.03416, 0, 0],
    [0, -5.03416, 0],
    [0, 0, 1]
])

# Print the matrices to verify
print("Matrix E:")
print(E)
print("\nMatrix F:")
print(F)

x0_MLD = np.array([ 0.29357406,  0.29125562, -0.01193462, -0.01774755,  1.58432257])

u_row = np.array([0.11950412229874177, -0.023786243102569576, 0.12212541796864013])

print(np.dot(E, x0_MLD)+np.dot(F, u_row))