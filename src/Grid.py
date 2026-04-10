import numpy as np

def Grid(L, N):
    """
    Generates a 1D spatial grid for the numerical simulation.

    Args:
        L (float): Total length of the spatial domain.
        N (int): Number of grid points.

    Returns:
        tuple: A tuple containing:
            - xi (np.ndarray): The 1D array of spatial coordinates.
            - dx (float): The uniform grid spacing.
    """
    xi = np.linspace(-L/2, L/2, N)
    dx = xi[1] - xi[0]
    return xi, dx