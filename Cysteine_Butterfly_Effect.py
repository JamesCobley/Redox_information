import numpy as np
import matplotlib.pyplot as plt

# Generate a small perturbation (butterfly wing flap)
def generate_initial_perturbation(n_points=300):
    t = np.linspace(0, 10, n_points)
    signal = 0.5 + 0.01 * np.sin(10 * t)  # small oscillation near baseline
    signal[50] += 0.05  # symbolic 'butterfly wing flap'
    return t, signal

# Generate divergent redox trajectories due to the initial perturbation
def generate_divergent_trajectories(t, perturbation):
    np.random.seed(0)
    base_trajectory = 0.5 + 0.01 * np.sin(2 * np.pi * t)
    chaos = np.zeros((5, len(t)))
    for i in range(5):
        noise = np.random.normal(0, 0.001, len(t))
        amplification = np.exp(0.3 * t)  # symbolic chaos amplification
        chaos[i] = base_trajectory + noise + amplification * (perturbation - 0.5)
    return chaos

# Plot
t, perturbation = generate_initial_perturbation()
chaotic_paths = generate_divergent_trajectories(t, perturbation)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Left Panel: Initial small perturbation
axs[0].plot(t, perturbation, color='teal', linewidth=2)
axs[0].set_title("Initial Redox Perturbation\n(Small Change, e.g., ROS burst)")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Oxidation Level")
axs[0].grid(True)

# Right Panel: Divergent outcomes
for i, traj in enumerate(chaotic_paths):
    axs[1].plot(t, traj, label=f'Trajectory {i+1}', alpha=0.8)

axs[1].set_title("Redox Butterfly Effect\n(Divergent Proteoform Dynamics)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Oxidation Level")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('butterfly_effect.png', dpi=300)
plt.show()
