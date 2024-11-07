import matplotlib.pyplot as plt

def create_line_graph(x, y, title="Line Graph", xlabel="X-Axis", ylabel="Y-Axis",
                      line_color="blue", line_style="-", marker="o", linewidth=2,
                      figsize=(10, 6), grid=True):
    """
    Creates a line graph with the specified parameters.

    Parameters:
        x (list or array-like): Values for the x-axis.
        y (list or array-like): Values for the y-axis.
        title (str): Title of the graph.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        line_color (str): Color of the line.
        line_style (str): Line style (e.g., '-', '--', '-.', ':').
        marker (str): Marker style (e.g., 'o', 's', '^', None).
        linewidth (int): Width of the line.
        figsize (tuple): Figure size, default is (10, 6).
        grid (bool): Whether to display a grid, default is True.

    Returns:
        fig, ax: The figure and axis of the created plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=line_color, linestyle=line_style, marker=marker, linewidth=linewidth)
    
    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Enable grid if specified
    if grid:
        ax.grid(True)

    plt.tight_layout()
    plt.show()