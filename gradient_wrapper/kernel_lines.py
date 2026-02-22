import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================
# save helper
# ================================
SAVE_DIR = Path("figures")
SAVE_DIR.mkdir(exist_ok=True)

def savefig(name):
    path = SAVE_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")


# ================================
# Kernel definition
# ================================
def phi(z, beta=5.0, k=2.0):
    return (1.0 + beta * (1.0 - z)) ** (-k)


# ============================================================
# 1) Plot kernel φ(z) for different parameters
# ============================================================
def plot_kernel_family(save=True):
    zs = np.linspace(-1, 1, 2000)

    plt.figure(figsize=(7,5))

    # vary beta
    for beta in [1, 2, 5, 10]:
        plt.plot(zs, phi(zs, beta=beta, k=2),
                 label=f"beta={beta}, k=2")

    # vary k
    for k in [1, 2, 4]:
        plt.plot(zs, phi(zs, beta=5, k=k),
                 linestyle="--",
                 label=f"beta=5, k={k}")

    plt.xlabel("z = cos(theta) = <v, g_hat>")
    plt.ylabel("phi(z)")
    plt.title("Heavy-tailed directional kernel")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        savefig("kernel_family.png")

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    plot_kernel_family()
