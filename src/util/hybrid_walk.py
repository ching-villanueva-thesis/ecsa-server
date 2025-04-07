import numpy as np
import matplotlib.pyplot as plt
import math
import io

def levy_flight(Lambda, size=1):
    """Generates Lévy flight step sizes."""
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    sigma_v = 1

    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)

    return u / (np.abs(v)**(1 / Lambda))

def gaussian_random_walk(best_solution, current_solution, t, lower_bound, upper_bound):
    """Gaussian random walk with boundary constraints."""
    r1, r2 = np.random.rand(), np.random.rand()
    sigma = (np.log(t + 2) / (t + 1)) * np.linalg.norm(best_solution - current_solution)

    gaussian_noise = np.random.normal(best_solution, sigma, size=len(best_solution))
    direction_term = r1 * best_solution - r2 * current_solution
    new_solution = np.clip(gaussian_noise + direction_term, lower_bound, upper_bound)

    return new_solution

def plt_hybrid_walk(n_steps=100, Lambda=1.5, lower_bound=-10, upper_bound=20):
    """
    Simulates a hybrid stochastic walk.
    - First half: Lévy Flight (Exploration)
    - Second half: Gaussian Random Walk (Exploitation)
    """
    best_solution = np.random.uniform(lower_bound, upper_bound, size=2)
    current_solution = np.random.uniform(lower_bound, upper_bound, size=2)

    x_positions, y_positions = [current_solution[0]], [current_solution[1]]

    for t in range(1, n_steps + 1):
        if t <= n_steps // 2:
            # First half: Lévy Flight
            step_x, step_y = levy_flight(Lambda, size=2)
            current_solution += np.array([step_x, step_y])
        else:
            # Second half: Gaussian Random Walk
            current_solution = gaussian_random_walk(best_solution, current_solution, t, lower_bound, upper_bound)

        current_solution = np.clip(current_solution, lower_bound, upper_bound)
        x_positions.append(current_solution[0])
        y_positions.append(current_solution[1])

    # Create plot and save to image bytes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_positions, y_positions, color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7, label="Path")
    ax.scatter(x_positions[0], y_positions[0], color="green", marker="o", label="Start")
    ax.scatter(x_positions[n_steps // 2], y_positions[n_steps // 2], color="orange", marker="s", label="Switch Point")
    ax.scatter(x_positions[-1], y_positions[-1], color="red", marker="x", label="End")
    ax.set_title("Hybrid Stochastic Walk: Lévy (Exploration) → Gaussian (Exploitation)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    img_bytes.seek(0)

    plt.close(fig)
    return img_bytes

def plt_levy_walk(n_steps=100, Lambda=1.5):
    """
    Simulates a Lévy flight in a 2D space and returns the plot as image bytes.

    :param n_steps: Number of steps
    :param Lambda: Lévy distribution parameter
    """
    # Initialize position
    x, y = [0], [0]

    for _ in range(n_steps):
        step_x = levy_flight(Lambda)
        step_y = levy_flight(Lambda)

        new_x = x[-1] + step_x[0]
        new_y = y[-1] + step_y[0]

        x.append(new_x)
        y.append(new_y)

    # Create plot and save to image bytes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7)
    ax.scatter(x[0], y[0], color="green", marker="o", label="Start")
    ax.scatter(x[-1], y[-1], color="red", marker="x", label="End")
    ax.set_title("Lévy Flight Simulation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    img_bytes.seek(0)

    plt.close(fig)
    return img_bytes