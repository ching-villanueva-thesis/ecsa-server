import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_convergence(csa_fmin, ecsa_fmin, csa_best, ecsa_best, function):
    # Create the figure and axis
    plt.figure(figsize=(12, 7))

    csa_iterations = np.arange(len(csa_fmin))
    ecsa_iterations = np.arange(len(ecsa_fmin))

    # Plot both convergence curves
    plt.plot(csa_iterations, csa_fmin, linewidth=2, label='Cuckoo Search Algorithm')
    plt.plot(ecsa_iterations, ecsa_fmin, linewidth=2, label='Enhanced Cuckoo Search Algorithm')

    # Add semi-transparent fills
    plt.fill_between(csa_iterations, csa_fmin, alpha=0.15, color='blue')
    plt.fill_between(ecsa_iterations, ecsa_fmin, alpha=0.15, color='red')

    # Customize the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Values of fmin', fontsize=12)
    plt.title(f"{function} Convergence Graph", fontsize=14)
    plt.legend(fontsize=10)

    # Add minor grid lines
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)

    info_box = plt.text(0.01, 0.98, f"CSA fmin = {csa_best}\nECSA fmin = {ecsa_best}",
                       transform=plt.gca().transAxes,
                       bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                       verticalalignment='top',
                       fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()