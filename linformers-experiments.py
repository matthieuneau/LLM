import numpy as np
import matplotlib.pyplot as plt


def generate_random_matrix(size: int, distribution: str, **kwargs) -> np.ndarray:
    """
    Generate a random square matrix of specified size and distribution.

    Args:
        size (int): The size of the square matrix (size x size).
        distribution (str): The name of the distribution to use. Supported distributions:
            - "uniform": Uniform distribution.
            - "normal": Normal (Gaussian) distribution.
            - "poisson": Poisson distribution.
            - "exponential": Exponential distribution.
        **kwargs: Additional parameters to pass to the distribution function.

    Returns:
        np.ndarray: A randomly generated square matrix.

    Raises:
        ValueError: If the specified distribution is not supported.
    """
    if distribution == "uniform":
        return np.random.uniform(size=(size, size), **kwargs)
    elif distribution == "normal":
        return np.random.normal(size=(size, size), **kwargs)
    elif distribution == "poisson":
        return np.random.poisson(size=(size, size), **kwargs)
    elif distribution == "exponential":
        return np.random.exponential(size=(size, size), **kwargs)
    else:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. Supported distributions are: "
            "'uniform', 'normal', 'poisson', 'exponential'."
        )


def compute_svd_explained_variance(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative explained variance of the singular values for a matrix.

    Args:
        matrix (np.ndarray): Input matrix for SVD decomposition.

    Returns:
        np.ndarray: Cumulative explained variance.
    """
    # Compute the SVD
    _, S, _ = np.linalg.svd(matrix, full_matrices=False)

    # Compute explained variance
    singular_values_squared = S**2
    total_variance = np.sum(singular_values_squared)
    explained_variance = singular_values_squared / total_variance
    return np.cumsum(explained_variance)


def plot_svd_with_error_bars(
    size: int, distribution: str, num_experiments: int, **kwargs
):
    """
    Plot the cumulative explained variance with 1-sigma bounds from repeated experiments.

    Args:
        size (int): Size of the square matrix.
        distribution (str): Distribution used for generating the matrix.
        num_experiments (int): Number of repeated experiments.

    Returns:
        None
    """
    results = []

    for _ in range(num_experiments):
        matrix = generate_random_matrix(size=size, distribution=distribution, **kwargs)
        cumulative_variance = compute_svd_explained_variance(matrix)
        results.append(cumulative_variance)

    # Convert results to a numpy array for easier manipulation
    results = np.array(results)

    # Compute mean and standard deviation across experiments
    mean_cumulative_variance = np.mean(results, axis=0)
    std_cumulative_variance = np.std(results, axis=0)

    # Plot the mean cumulative explained variance
    plt.figure(figsize=(8, 5))
    x = range(1, len(mean_cumulative_variance) + 1)
    plt.plot(
        x,
        mean_cumulative_variance,
        marker="o",
        linestyle="-",
        label="Mean Cumulative Explained Variance",
    )
    # Add 1-sigma bounds
    plt.fill_between(
        x,
        mean_cumulative_variance - std_cumulative_variance,
        mean_cumulative_variance + std_cumulative_variance,
        color="red",
        alpha=0.3,
        label="1-Sigma Bounds",
    )

    plt.title("Cumulative Explained Variance by Singular Values", fontsize=14)
    plt.xlabel("Number of Singular Values", fontsize=12)
    plt.ylabel("Cumulative Explained Variance", fontsize=12)
    plt.grid(True)
    plt.xticks(range(0, len(mean_cumulative_variance) + 1, 20))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_svd_with_error_bars(
        size=512, distribution="uniform", num_experiments=10, low=0, high=1
    )

