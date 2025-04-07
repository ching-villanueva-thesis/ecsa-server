import numpy as np
import matplotlib.pyplot as plt
import io

from src.ecs.sop.sop3 import CosineAnnealingWithWarmRestarts

def plot_cosine_annealing(n_min, n_max, t_max, p, c_max=4):
    scheduler = CosineAnnealingWithWarmRestarts(n_min, n_max, t_max, c_max=c_max)
    print(c_max)
    values = [scheduler.step() for _ in range(t_max)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values, color="royalblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(f"Cosine Annealing with Warm Restarts ({p})")
    ax.legend()
    ax.grid(True)
    
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    img_bytes.seek(0)
    plt.close(fig)
    
    return img_bytes

def plot_fixed_parameters(p, iterations, parameter):
    values = [p for _ in range(iterations)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values, color="royalblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(f"Fixed Values for ({parameter})")
    ax.legend()
    ax.grid(True)
    
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    img_bytes.seek(0)
    plt.close(fig)
    
    return img_bytes