import os
import numpy as np
import matplotlib.pyplot as plt
import Functions as fn  # Explicitly importing your functions module
from Grid import Grid   # Explicitly importing the Grid function

# Ensure the assets directory exists so code doesn't crash when saving
os.makedirs('assets', exist_ok=True)

plt.rcParams.update({'font.size': 20})
plt.rc('legend', fontsize=14)

# ==========================================
# Plot 1: N = 100, L = 10, P = 30
# ==========================================
xi, dx = Grid(L=10, N=100)
H = fn.stationary_hamiltonian(N=100, L=10, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), xi))  
    ax1.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax1.set_xlabel('$\u03BE$')
ax1.set_ylabel('u($\u03BE$)')
ax1.set_title('N = 100, L = 10, P = 30')
ax1.legend()
plt.savefig('assets/Plot1_N100_L10_P30.png') # Saving directly to assets/
plt.close() # Closes the figure to save memory when running large scripts


# ==========================================
# Time Evolution with Modulated Potential
# ==========================================
xi, dx = Grid(L=100, N=1000)

H = fn.stationary_hamiltonian(N=1000, L=100, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)  

u_0_norm = fn.Normalising_wavefunction(bound_wavefunctions, 0, dx)
u_1_norm = fn.Normalising_wavefunction(bound_wavefunctions, 1, dx)
u_2_norm = fn.Normalising_wavefunction(bound_wavefunctions, 2, dx)

epsilon_0 = fn.Selecting_energy(bound_energies, 0)
epsilon_2 = fn.Selecting_energy(bound_energies, 2)

psi_0 = u_0_norm.copy()
psi_1 = u_1_norm.copy()
psi_2 = u_2_norm.copy()

for eta in [0.1, 0.5, 1.0]:
    omega = fn.Modulation_Frequency(epsilon_0, epsilon_2)
    TM = 2 * np.pi / omega
    t_max = 8 * np.pi / omega
    dt = 0.01 * TM
    num_steps = int(t_max / dt)
    
    time_array_2 = np.zeros(num_steps+1)

    c_0 = np.zeros(num_steps+1, dtype=complex)
    c_1 = np.zeros(num_steps+1, dtype=complex)
    c_2 = np.zeros(num_steps+1, dtype=complex) 

    V_n = fn.Time_Evolving_Potential(xi, p=30, t=0, eta=eta, omega=omega)
    t = 0.0

    for n in range(num_steps+1):
        time_array_2[n] = t

        c_0[n] = np.sum(np.conj(psi_0) * u_0_norm) * dx
        c_1[n] = np.sum(np.conj(psi_1) * u_1_norm) * dx
        c_2[n] = np.sum(np.conj(psi_2) * u_2_norm) * dx

        V_n_1 = fn.Time_Evolving_Potential(xi, p=30, t=t+dt, eta=eta, omega=omega)
        V_mid = 0.5 * (V_n + V_n_1)

        H_t = fn.Time_Evolving_Hamiltonian(N=1000, p=30, dx=dx, xi=xi, t=t, eta=eta, omega=omega)
        A_inv, B = fn.Crank_Nicholson_Matrices(N=1000, dt=dt, H=H_t)

        psi_0 = fn.Crank_Nicholson_Step(psi_0, A_inv, B)
        psi_1 = fn.Crank_Nicholson_Step(psi_1, A_inv, B)
        psi_2 = fn.Crank_Nicholson_Step(psi_2, A_inv, B)

        t += dt
        V_n = V_n_1
    
    Modulation_time = (time_array_2 * omega) / (2 * np.pi)

    fig10 = plt.figure(figsize=(10, 8))
    ax10 = fig10.add_subplot(1, 1, 1)
    ax10.plot(Modulation_time, np.abs(c_0), label='|c_0(t)|')
    ax10.plot(Modulation_time, np.abs(c_1), label='|c_1(t)|')
    ax10.plot(Modulation_time, np.abs(c_2), label='|c_2(t)|')
    ax10.set_xlabel('t / $T_M$')
    ax10.legend()
    ax10.grid(True)
    fig10.savefig(f'assets/Plot10_Time_Evolution_Modulated_Potential_eta_{eta}.png', bbox_inches='tight')
    plt.close()