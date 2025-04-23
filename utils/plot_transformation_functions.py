"""Plot transformation functions of Pearson values as used in LNM.

The script is not part of the main analysis pipeline.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from utils import power_transform

x = np.linspace(-0.999, 0.999, 500)

identity = x
fisher_x = np.arctanh(x)
inverse_fisher_x = np.tanh(x)
power_transformed_x = power_transform(x)

# Plot all functions
plt.figure(figsize=(10, 6))

plt.plot(x, identity, label="Diagonal", linestyle="--", color="gray")
plt.plot(x, fisher_x, label="Fisher transform (artanh)", color="blue")
plt.plot(x, inverse_fisher_x, label="Inverse Fisher (tanh)", color="green")
plt.plot(x, power_transformed_x, label="Power-law: sign(x)·|x|^0.5", color="red")

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.title("Transformations of Correlation Values")
plt.xlabel("x")
plt.ylabel("Transformed value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
