import numpy as np
import matplotlib
import io
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

matplotlib.use("Agg")

def compute_hamming_diversity(binary_population):
    """Computes the average Hamming distance as a diversity measure."""
    if binary_population.shape[0] < 2:
        return 0  # Not enough individuals for diversity computation
    
    # Compute pairwise Hamming distances
    hamming_distances = pdist(binary_population, metric="hamming")
    
    # Compute average Hamming distance
    return np.mean(hamming_distances)

def plot_scatter(binary_population, title, method="t-SNE", n_clusters=3):
    import matplotlib.pyplot as plt

    binary_population = np.array(binary_population)
    reducer = TSNE(n_components=2, random_state=42)
    reduced_data = reducer.fit_transform(binary_population)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        reduced_data[:, 0], reduced_data[:, 1], color="royalblue", alpha=0.6
    )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{title} ({method})")
    ax.grid(True)

    # Add legend with Hamming Diversity Score
    legend_text = method
    ax.legend([legend_text], loc="upper right", fontsize=10, frameon=True)

    # Save plot to image bytes
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    img_bytes.seek(0)

    plt.close(fig)
    return img_bytes
