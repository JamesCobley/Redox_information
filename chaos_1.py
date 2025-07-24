import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bifurcation diagram (logistic map as redox proxy model)
mu_values = np.linspace(2.5, 4.0, 10000)
iterations = 1000
last = 100
x0 = 0.1

bifurcation_x = []
bifurcation_y = []

for mu in mu_values:
    x = x0
    for i in range(iterations):
        x = mu * x * (1 - x)
        if i >= (iterations - last):
            bifurcation_x.append(mu)
            bifurcation_y.append(x)

# Recurrence plot from synthetic redox signal
t = np.linspace(0, 20 * np.pi, 300)
signal = np.sin(t) + 0.1 * np.random.randn(300)
signal[150:] += np.sin(3 * t[150:]) * np.sin(0.2 * t[150:])  # inject chaos

def recurrence_plot(x, epsilon):
    N = len(x)
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            R[i, j] = np.abs(x[i] - x[j]) < epsilon
    return R

epsilon = 0.3 * np.std(signal)
R = recurrence_plot(signal, epsilon)

# Plot both
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Bifurcation
axs[0].plot(bifurcation_x, bifurcation_y, ',k', alpha=0.25)
axs[0].set_title('Bifurcation Diagram (Redox Proxy)')
axs[0].set_xlabel('Control Parameter μ (e.g., ROS flux)')
axs[0].set_ylabel('Oxidation State x')

# Recurrence
sns.heatmap(R, cmap='viridis', cbar=False, ax=axs[1])
axs[1].set_title('Recurrence Plot of Synthetic Redox Signal')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Time')

plt.tight_layout()
plt.savefig('bifurcation_recurrence.png', dpi=300)
plt.show()
