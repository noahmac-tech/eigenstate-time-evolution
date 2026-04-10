import numpy as np
from numpy.linalg import eigh

def q_parameter(p):
    """Calculates the q parameter for the Pöschl-Teller potential well."""
    return 1 / np.sqrt(p)

def potential_energy(x, q):
    """Calculates the stationary localized potential energy at position x."""
    return -1 / (np.cosh(q * x) ** 2)

def number_of_localized_states(p):
    """Calculates the theoretical number of bound (localized) states for a given p."""
    s = 0.5 * (np.sqrt(1 + 4 * p) - 1)
    return int(s)

def localized_analytical_energies(p, n):
    """Returns the analytical energy eigenvalues for comparison with numerical results."""
    s = number_of_localized_states(p)
    energies_analytical = []
    for n in range(int(s)):
        E_n = -(1/(4*p)) * (-(1 + 2*n) + np.sqrt(1 + 4*p))**2
        if E_n < 0: 
            energies_analytical.append(E_n)
    return energies_analytical

def stationary_hamiltonian(N, L, p, dx, xi):
    """Constructs the stationary Hamiltonian matrix using finite difference methods."""
    main_diag = 2 / dx**2 + potential_energy(xi, q_parameter(p))
    off_diag = -1 / dx**2 * np.ones(N-1)
    H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return H

def eigenvalues_and_vectors(H):
    """Solves the eigenvalue problem for the given Hamiltonian."""
    energies, wavefuncs = eigh(H)
    return energies, wavefuncs

def Bound_energies(energies):
    """Filters and returns only the negative (bound state) energies."""
    bound = np.where(energies < 0)[0]
    return energies[bound]

def Bound_wavefuncs(wavefuncs, energies):
    """Filters and returns the wavefunctions corresponding to bound states."""
    bound = np.where(energies < 0)[0]
    return wavefuncs[:, bound] 

def Normalising_wavefunction(bound_wavefunctions, n, dx):
    """Normalizes a specific bound wavefunction using trapezoidal integration."""
    u = bound_wavefunctions[:, n]
    u_norm = u / np.sqrt(np.trapezoid(np.abs(u)**2, dx=dx))
    return u_norm

def Selecting_energy(bound_energies, n):
    """Selects the energy of the n-th bound state."""
    epsilon_n = bound_energies[n]
    return epsilon_n

def Crank_Nicholson_Matrices(N, dt, H):
    """Generates the A inverse and B matrices required for the Crank-Nicholson step."""
    I = np.identity(N)
    A = I + 0.5j * dt * H
    B = I - 0.5j * dt * H
    A_inv = np.linalg.inv(A)
    return A_inv, B

def Crank_Nicholson_Step(psi_t, A_inv, B):
    """Advances the wavefunction by one time step (dt) using Crank-Nicholson."""
    psi_t_dt = A_inv @ (B @ psi_t)
    return psi_t_dt

def Modulation_Frequency(epsilon_0, epsilon_2):
    """Calculates the modulation frequency between two energy states."""
    omega = np.abs(epsilon_2 - epsilon_0)
    return omega

def Time_Evolving_Potential(xi, p, t, eta, omega):
    """Calculates the time-dependent modulated potential well."""
    V_t = -(1 + (eta * np.sin(omega * t))) / (np.cosh(q_parameter(p) * xi) ** 2)
    return V_t

def Time_Evolving_Hamiltonian(N, p, dx, xi, t, eta, omega):
    """Constructs the Hamiltonian matrix for a specific point in time."""
    V_t = Time_Evolving_Potential(xi, p, t, eta, omega)
    main_diag = 2 / dx**2 + V_t
    off_diag = -1 / dx**2 * np.ones(N-1)
    H_t = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return H_t