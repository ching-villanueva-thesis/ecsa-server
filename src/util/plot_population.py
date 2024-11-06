import matplotlib.pyplot as plt

def plot_population(initial_pop, min_values, max_values):
    """
    Creates a scatter plot of the initial population for Cuckoo Search.
    
    Parameters:
    -----------
    initial_pop : numpy.ndarray
        Initial population array where each row contains [x, y, fitness]
    min_values : tuple
        Minimum values for x and y coordinates
    max_values : tuple
        Maximum values for x and y coordinates
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(initial_pop[:,0], initial_pop[:,1], c='blue', alpha=0.6)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cuckoo Search Initial Population Distribution')
    plt.grid(True)
    plt.axis('equal')

    plt.xlim(min_values[0], max_values[0])
    plt.ylim(min_values[1], max_values[1])

    plt.show()