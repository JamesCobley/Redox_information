# Step 0: Install required library
!pip install pyts

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

# Step 2: Generate a synthetic redox signal with fractal-like, recursive bursts
np.random.seed(0)
n = 300
t = np.linspace(0, 8 * np.pi, n)
base = 0.5 + 0.05 * np.sin(t)

signal = base.copy()
for scale in [1, 2, 4]:
    burst_indices = np.arange(0, n, 30 * scale)
    signal[burst_indices] += 0.05 * scale * np.sin(5 * t[burst_indices])
signal = np.clip(signal, 0, 1)

# Step 3: Define imaginary component as velocity (derivative)
imaginary = np.gradient(signal)

# Step 4: Create complex signal Z(t)
z_t = signal + 1j * imaginary

# Step 5: Generate recurrence plot on magnitude
rp = RecurrencePlot(threshold='point', percentage=20)
X_rp = rp.fit_transform([np.abs(z_t)])

# Step 6: Plot results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(np.real(z_t), color='navy', label='Real: Oxidation Level')
axs[0].plot(np.imag(z_t), color='crimson', linestyle='--', label='Imaginary: Velocity')
axs[0].set_title('Fractal Redox Signal in $\mathbb{C}_{Redox}$')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Signal')
axs[0].legend()

axs[1].imshow(X_rp[0], cmap='inferno', origin='lower')
axs[1].set_title('Recurrence Plot of Complex Redox Signal')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Time')

plt.tight_layout()
plt.savefig('complex_recurrence.png', dpi=300)
plt.show()
