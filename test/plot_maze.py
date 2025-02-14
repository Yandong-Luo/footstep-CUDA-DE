import matplotlib.pyplot as plt
import numpy as np

# Define regions - aligned with reference image
x_lim = [
    # Left section (L shape)
    {'xl': 0.0, 'xu': 0.1, 'yl': 0, 'yu': 1, 'color': 'gray'},    # Left vertical bar
    {'xl': 0.2, 'xu': 1.1, 'yl': 0, 'yu': 0.1, 'color': 'gray'},     # Bottom horizontal
    {'xl': 1.2, 'xu': 1.3, 'yl': 0, 'yu': 1, 'color': 'gray'},     # Right vertical
    {'xl': 0.2, 'xu': 1.1, 'yl': 0.9, 'yu': 1, 'color': 'gray'},     # Top horizontal

    {'xl': 0.2, 'xu': 0.9, 'yl': 0.7, 'yu': 0.8, 'color': 'gray'},     # left center horizontal
    {'xl': 1.0, 'xu': 1.1, 'yl': 0.2, 'yu': 0.8, 'color': 'gray'},     # left center vertical

    {'xl': 1.4, 'xu': 1.5, 'yl': 0.6, 'yu': 0.8, 'color': 'gray'},     # center vertical
    {'xl': 1.4, 'xu': 1.5, 'yl': 0.3, 'yu': 0.5, 'color': 'gray'},     # center vertical

    {'xl': 2.8, 'xu': 2.9, 'yl': 0, 'yu': 1, 'color': 'gray'},    # Right vertical bar (symmetric to left)
    {'xl': 1.8, 'xu': 2.7, 'yl': 0, 'yu': 0.1, 'color': 'gray'},     # Bottom horizontal
    {'xl': 1.6, 'xu': 1.7, 'yl': 0, 'yu': 1, 'color': 'gray'},     # Left vertical
    {'xl': 1.8, 'xu': 2.7, 'yl': 0.9, 'yu': 1, 'color': 'gray'},     # Top horizontal
    {'xl': 2.0, 'xu': 2.7, 'yl': 0.7, 'yu': 0.8, 'color': 'gray'},     # right center horizontal
    {'xl': 1.8, 'xu': 1.9, 'yl': 0.2, 'yu': 0.8, 'color': 'gray'},     # right center vertical
]

# Create plot
plt.figure(figsize=(12, 6))
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
plt.xlim(-0.1, 3.1)
plt.ylim(-0.1, 1.1)
plt.grid(True)

# Add title and labels
plt.title('Region Layout')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()