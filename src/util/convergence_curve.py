import numpy as np
import matplotlib.pyplot as plt

def show_convergence(csa_fmin, ecsa_fmin, function):
    # Create the figure and axis
    plt.figure(figsize=(12, 7))

    csa_iterations = np.arange(len(csa_fmin))
    ecsa_iterations = np.arange(len(ecsa_fmin))

    # Plot both convergence curves
    plt.plot(csa_iterations, csa_fmin, linewidth=2, label='Cuckoo Search Algorithm')
    plt.plot(ecsa_iterations, ecsa_fmin, linewidth=2, label='Enhanced Cuckoo Search Algorithm')

    # Customize the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlabel('Iterations', fontsize=12)
    # plt.ylabel('FEVs', fontsize=12)
    # plt.title(f"{function} Convergence Graph", fontsize=14)
    # plt.legend(fontsize=10)

    # Set the y-axis to a base 10 log scale
    ax = plt.gca()
    ax.set_yscale('log')

    # Add minor grid lines
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)

    # Adjust layout
    plt.tight_layout()

    plt.savefig('viz.png', format='png', dpi=600, transparent=True)
    # Show the plot
    plt.show()