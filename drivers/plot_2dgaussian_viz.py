import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma_x, sigma_y = 0.24, 0.14
# Define grid limits and resolution
lim = 3
n_points = 1000
x = np.linspace(-lim, lim, n_points)
y = np.linspace(-lim, lim, n_points)
X, Y = np.meshgrid(x, y)

# Compute the 2D Gaussian distribution
I = np.exp(-((X**2)/(2*sigma_x**2) + (Y**2)/(2*sigma_y**2)))

# Plot the Gaussian image
plt.figure(figsize=(8, 6))
plt.imshow(I, extent=[-lim,lim,-lim,lim], cmap='gray_r', origin='lower')
plt.colorbar(label='Intensity')

# Overplot the circumference of a circle with radius 1
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), color='k', linewidth=2)

plt.title('2D Gaussian Distribution with Circle')
plt.xlabel('x')
plt.ylabel('z')

# Calculate the fraction of the circle (radius 1) blocked by the Gaussian numerically
dx = x[1] - x[0]  # grid spacing in x
dy = y[1] - y[0]  # grid spacing in y
mask = (X**2 + Y**2) <= 1  # mask for points inside the circle of radius 1
integral = np.sum(I[mask]) * dx * dy  # numerical integration over the circle
circle_area = np.pi  # area of circle with radius 1
fraction = integral / circle_area
print("Blocked fraction:", fraction)

# Ensure output directory exists and save the image
output_dir = os.path.join('results', '2dgaussian_viz')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, '2dgaussian.png'), dpi=300)
plt.close()
