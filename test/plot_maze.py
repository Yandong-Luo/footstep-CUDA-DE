import matplotlib.pyplot as plt
import numpy as np

# Define scale factor
scale = 1.0  # 可以随时调整这个值来改变缩放比例

# Define original regions and scale them
x_lim = [
    # Left section (L shape)
    {'xl': 0.0*scale, 'xu': 0.2*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},    # Left vertical bar
    {'xl': 0.25*scale, 'xu': 1.0*scale, 'yl': 0*scale, 'yu': 0.2*scale, 'color': 'gray'},     # Bottom horizontal
    {'xl': 1.05*scale, 'xu': 1.25*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},     # Right vertical
    {'xl': 0.25*scale, 'xu': 1.0*scale, 'yl': 0.8*scale, 'yu': 1*scale, 'color': 'gray'},     # Top horizontal

    {'xl': 0.25*scale, 'xu': 0.9*scale, 'yl': 0.7*scale, 'yu': 0.8*scale, 'color': 'gray'},     # left center horizontal
    # {'xl': 1.0*scale, 'xu': 1.1*scale, 'yl': 0.2*scale, 'yu': 0.8*scale, 'color': 'gray'},     # left center vertical

    # {'xl': 1.4*scale, 'xu': 1.5*scale, 'yl': 0.6*scale, 'yu': 0.8*scale, 'color': 'gray'},     # center vertical
    # {'xl': 1.4*scale, 'xu': 1.5*scale, 'yl': 0.3*scale, 'yu': 0.5*scale, 'color': 'gray'},     # center vertical

    # {'xl': 2.8*scale, 'xu': 2.9*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},    # Right vertical bar
    # {'xl': 1.8*scale, 'xu': 2.7*scale, 'yl': 0*scale, 'yu': 0.1*scale, 'color': 'gray'},     # Bottom horizontal
    # {'xl': 1.6*scale, 'xu': 1.7*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},     # Left vertical
    # {'xl': 1.8*scale, 'xu': 2.7*scale, 'yl': 0.9*scale, 'yu': 1*scale, 'color': 'gray'},     # Top horizontal
    # {'xl': 2.0*scale, 'xu': 2.7*scale, 'yl': 0.7*scale, 'yu': 0.8*scale, 'color': 'gray'},     # right center horizontal
    # {'xl': 1.8*scale, 'xu': 1.9*scale, 'yl': 0.2*scale, 'yu': 0.8*scale, 'color': 'gray'},     # right center vertical
]

# Create plot
plt.figure(figsize=(15, 8))
ax = plt.gca()
ax.set_aspect('equal')

# Add all regions
for item in x_lim:
    rectangle = plt.Rectangle((item['xl'], item['yl']), 
                            (item['xu']-item['xl']), 
                            (item['yu']-item['yl']), 
                            fc=item['color'])
    ax.add_patch(rectangle)

# Set plot limits and grid
plt.xlim(-0.1*scale, 3.1*scale)
plt.ylim(-0.1*scale, 1.1*scale)
plt.grid(True)

# Add title and labels
plt.title(f'Region Layout (Scale: {scale}x)')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Define scale factor
# scale = 1.0  # 可以随时调整这个值来改变缩放比例

# # Define original regions and scale them
# x_lim = [
#     # Left section (L shape)
#     {'xl': 0.0*scale, 'xu': 0.4*scale, 'yl': 0*scale, 'yu': 1.5*scale, 'color': 'gray'},    # Left vertical bar
#     {'xl': 0.5*scale, 'xu': 1.5*scale, 'yl': 0*scale, 'yu': 0.4*scale, 'color': 'gray'},     # Bottom horizontal
#     # {'xl': 1.2*scale, 'xu': 1.3*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},     # Right vertical
#     # {'xl': 0.2*scale, 'xu': 1.1*scale, 'yl': 0.9*scale, 'yu': 1*scale, 'color': 'gray'},     # Top horizontal

#     # {'xl': 0.2*scale, 'xu': 0.9*scale, 'yl': 0.7*scale, 'yu': 0.8*scale, 'color': 'gray'},     # left center horizontal
#     # {'xl': 1.0*scale, 'xu': 1.1*scale, 'yl': 0.2*scale, 'yu': 0.8*scale, 'color': 'gray'},     # left center vertical

#     # {'xl': 1.4*scale, 'xu': 1.5*scale, 'yl': 0.6*scale, 'yu': 0.8*scale, 'color': 'gray'},     # center vertical
#     # {'xl': 1.4*scale, 'xu': 1.5*scale, 'yl': 0.3*scale, 'yu': 0.5*scale, 'color': 'gray'},     # center vertical

#     # {'xl': 2.8*scale, 'xu': 2.9*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},    # Right vertical bar
#     # {'xl': 1.8*scale, 'xu': 2.7*scale, 'yl': 0*scale, 'yu': 0.1*scale, 'color': 'gray'},     # Bottom horizontal
#     # {'xl': 1.6*scale, 'xu': 1.7*scale, 'yl': 0*scale, 'yu': 1*scale, 'color': 'gray'},     # Left vertical
#     # {'xl': 1.8*scale, 'xu': 2.7*scale, 'yl': 0.9*scale, 'yu': 1*scale, 'color': 'gray'},     # Top horizontal
#     # {'xl': 2.0*scale, 'xu': 2.7*scale, 'yl': 0.7*scale, 'yu': 0.8*scale, 'color': 'gray'},     # right center horizontal
#     # {'xl': 1.8*scale, 'xu': 1.9*scale, 'yl': 0.2*scale, 'yu': 0.8*scale, 'color': 'gray'},     # right center vertical
# ]

# # Create plot
# plt.figure(figsize=(15, 8))
# ax = plt.gca()
# ax.set_aspect('equal')

# # Add all regions
# for item in x_lim:
#     rectangle = plt.Rectangle((item['xl'], item['yl']), 
#                             (item['xu']-item['xl']), 
#                             (item['yu']-item['yl']), 
#                             fc=item['color'])
#     ax.add_patch(rectangle)

# # Set plot limits and grid
# plt.xlim(-0.1*scale, 3.1*scale)
# plt.ylim(-0.1*scale, 2.1*scale)
# plt.grid(True)

# # Add title and labels
# plt.title(f'Region Layout (Scale: {scale}x)')
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.show()