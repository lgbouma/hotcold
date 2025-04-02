import pandas as pd
import numpy as np
from aesthetic.plot import set_style
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Set the plot style to 'science'
set_style('science')

# Load CSV data with pandas
df = pd.read_csv('profile_check_inner.csv')

x, y = df['xval'], df['yval']

def single_gauss(x, a, mu, sigma):
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))

def double_gauss(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return a1*np.exp(-((x-mu1)**2) / (2*sigma1**2)) + a2*np.exp(-((x-mu2)**2) / (2*sigma2**2))

# Initial parameter guesses:
# Amplitudes are 0.5, means are 2 and 2.1 (ensuring at least 0.1 apart), and sigmas are 0.2.
p0 = [0.5, 2.0, 0.2, 0.5, 2.1, 0.2]

# Optionally, set bounds to ensure physically meaningful values:
lower_bounds = [0, -5, 0.01, 0, -5, 0.01]
upper_bounds = [1, 5, 0.4, 1, 5, 0.4]

params, cov = curve_fit(double_gauss, x, y, p0=p0, bounds=(lower_bounds, upper_bounds))

# Generate fitted curve data
x_fit = np.linspace(np.min(x), np.max(x), 500)
y_fit = double_gauss(x_fit, *params)

# Plot the fitted curve with the original data
y_component1 = single_gauss(x_fit, params[0], params[1], params[2])
y_component2 = single_gauss(x_fit, params[3], params[4], params[5])

plt.plot(x_fit, y_fit, label='Fitted Sum of Two Gaussians', alpha=0.5)
plt.plot(x_fit, y_component1, '--', label='Gaussian Component 1', alpha=0.5)
plt.plot(x_fit, y_component2, '--', label='Gaussian Component 2', alpha=0.5)
plt.legend()

print("Fitted parameters:")
print(f"Amplitude1 = {params[0]:.4f}, Mean1 = {params[1]:.4f}, Sigma1 = {params[2]:.4f}")
print(f"Amplitude2 = {params[3]:.4f}, Mean2 = {params[4]:.4f}, Sigma2 = {params[5]:.4f}")
uncertainty_mean1 = np.sqrt(cov[1, 1])
uncertainty_mean2 = np.sqrt(cov[4, 4])
print(f"Uncertainty in Mean1: {uncertainty_mean1:.4f}")
print(f"Uncertainty in Mean2: {uncertainty_mean2:.4f}")

# Plot the DataFrame
ax = plt.gca()
ax.plot(x, y, c='k', lw=2, zorder=-1)
plt.title('Profile Check Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()