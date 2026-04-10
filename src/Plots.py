import os
import numpy as np
import matplotlib.pyplot as plt
import Functions as fn
from Grid import Grid

# Ensure the assets directory exists
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
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax1.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax1.set_xlabel('$\u03BE$')
ax1.set_ylabel('u($\u03BE$)')
ax1.set_title('N = 100, L = 10, P = 30')
ax1.legend()
plt.savefig('assets/Plot1_N100_L10_P30.png')
plt.close()

# ==========================================
# Plot 2: N = 100, L = 50, P = 30
# ==========================================
xi, dx = Grid(L=50, N=100)
H = fn.stationary_hamiltonian(N=100, L=50, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax2.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax2.set_xlabel('$\u03BE$')
ax2.set_ylabel('u($\u03BE$)')
ax2.set_title('N = 100, L = 50, P = 30')
ax2.legend()
plt.savefig('assets/Plot2_N100_L50_P30.png')
plt.close()

# ==========================================
# Plot 3 & 5: N = 100, L = 100, P = 30
# ==========================================
xi, dx = Grid(L=100, N=100)
H = fn.stationary_hamiltonian(N=100, L=100, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax3.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax3.set_xlabel('$\u03BE$')
ax3.set_ylabel('u($\u03BE$)')
ax3.set_title('N = 100, L = 100, P = 30')
ax3.legend()
plt.savefig('assets/Plot3_N100_L100_P30.png')
plt.savefig('assets/Plot5_N100_L100_P30.png') # Saving as both per your original script
plt.close()

# ==========================================
# Plot 4: N = 100, L = 150, P = 30
# ==========================================
xi, dx = Grid(L=150, N=100)
H = fn.stationary_hamiltonian(N=100, L=150, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax4.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax4.set_xlabel('$\u03BE$')
ax4.set_ylabel('u($\u03BE$)')
ax4.set_title('N = 100, L = 150, P = 30')
ax4.legend()
plt.savefig('assets/Plot4_N100_L150_P30.png')
plt.close()

# ==========================================
# Plot 6: N = 500, L = 100, P = 30
# ==========================================
xi, dx = Grid(L=100, N=500)
H = fn.stationary_hamiltonian(N=500, L=100, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig6 = plt.figure(figsize=(10, 8))
ax6 = fig6.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax6.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax6.set_xlabel('$\u03BE$')
ax6.set_ylabel('u($\u03BE$)')
ax6.set_title('N = 500, L = 100, P = 30')
ax6.legend()
plt.savefig('assets/Plot6_N500_L100_P30.png')
plt.close()

# ==========================================
# Plot 7: N = 1000, L = 100, P = 30
# ==========================================
xi, dx = Grid(L=100, N=1000)
H = fn.stationary_hamiltonian(N=1000, L=100, p=30, dx=dx, xi=xi)
energies, wavefunctions = fn.eigenvalues_and_vectors(H)
bound_energies = fn.Bound_energies(energies)
bound_wavefunctions = fn.Bound_wavefuncs(wavefunctions, energies)

fig7 = plt.figure(figsize=(10, 8))
ax7 = fig7.add_subplot(1, 1, 1)
for i, psi in enumerate(bound_wavefunctions.T):
    psi_norm = psi / np.sqrt(np.trapezoid((np.abs(psi)**2), dx=dx))  
    ax7.plot(xi, psi_norm, label=f'n={i}, $\u03B5$={bound_energies[i]:.3f}')
ax7.set_xlabel('$\u03BE$')
ax7.set_ylabel('u($\u03BE$)')
ax7.legend()
plt.savefig('assets/Plot7_N1000_L100_P30.png')
plt.close()

# ==========================================
# Plot 8: Comparing analytical and numerical energies
# ==========================================
fig8 = plt.figure(figsize=(10, 8))
ax8 = fig8.add_subplot(1, 1, 1)
for n in range(len(bound_energies)):
    E_analytical = fn.localized_analytical_energies(p=30, n=n)
    if len(E_analytical) > n:
        ax8.plot(n, bound_energies[n], 'bo')
        ax8.plot(n, E_analytical[n], 'rx')
ax8.set_xlabel('n')
ax8.set_ylabel('Energy')   
ax8.legend(['Numerical', 'Analytical'])
plt.savefig('assets/Plot8_Analytical_vs_Numerical_Energies.png')
plt.close()

# ==========================================
# Plot 9: Time Evolution of Coefficient c_0
# ==========================================
u_0_norm = fn.Normalising_wavefunction(bound_wavefunctions, 0, dx)
epsilon_0 = fn.Selecting_energy(bound_energies, 0)

T0 = 2 * np.pi / np.abs(epsilon_0)
dt = 0.1
t_max = 8 * np.pi / np.abs(epsilon_0)
num_steps = int(t_max / dt)

c_0 = np.zeros(num_steps, dtype=complex)
time_array_1 = np.zeros(num_steps)
psi = u_0_norm.copy()

for n in range(num_steps):
    c_0[n] = np.sum(np.conj(psi) * u_0_norm) * dx
    time_array_1[n] = n * dt
    A_inv, B = fn.Crank_Nicholson_Matrices(N=1000, dt=dt, H=H)
    psi = fn.Crank_Nicholson_Step(psi, A_inv, B)

fig9 = plt.figure(figsize=(10, 8))
ax9 = fig9.add_subplot(1, 1, 1)
ax9.plot(time_array_1 / T0, np.abs(c_0), label='|c_0(t)|')
ax9.plot(time_array_1 / T0, np.real(c_0), label='Re(c_0(t))')
ax9.plot(time_array_1 / T0, np.imag(c_0), label='Im(c_0(t))')
ax9.grid(True)
ax9.legend()
plt.savefig('assets/Plot9_Time_Evolution_c0.png', bbox_inches='tight')
plt.close()

# ==========================================
# Plot 10: Time Evolution with Modulated Potential
# ==========================================
u_1_norm = fn.Normalising_wavefunction(bound_wavefunctions, 1, dx)
u_2_norm = fn.Normalising_wavefunction(bound_wavefunctions, 2, dx)
epsilon_2 = fn.Selecting_energy(bound_energies, 2)

psi_0 = u_0_norm.copy()
psi_1 = u_1_norm.copy()
psi_2 = u_2_norm.copy()

for eta in [0.1, 0.5, 1.0]:
    omega = fn.Modulation_Frequency(epsilon_0, epsilon_2)
    TM = 2 * np.pi / omega
    t_max = 8 * np.pi / omega
    dt_mod = 0.01 * TM
    num_steps_mod = int(t_max / dt_mod)
    
    time_array_2 = np.zeros(num_steps_mod+1)
    c_0_mod = np.zeros(num_steps_mod+1, dtype=complex)
    c_1_mod = np.zeros(num_steps_mod+1, dtype=complex)
    c_2_mod = np.zeros(num_steps_mod+1, dtype=complex) 

    V_n = fn.Time_Evolving_Potential(xi, p=30, t=0, eta=eta, omega=omega)
    t = 0.0

    psi_0_temp = psi_0.copy()
    psi_1_temp = psi_1.copy()
    psi_2_temp = psi_2.copy()

    for n in range(num_steps_mod+1):
        time_array_2[n] = t

        c_0_mod[n] = np.sum(np.conj(psi_0_temp) * u_0_norm) * dx
        c_1_mod[n] = np.sum(np.conj(psi_1_temp) * u_1_norm) * dx
        c_2_mod[n] = np.sum(np.conj(psi_2_temp) * u_2_norm) * dx

        V_n_1 = fn.Time_Evolving_Potential(xi, p=30, t=t+dt_mod, eta=eta, omega=omega)
        H_t = fn.Time_Evolving_Hamiltonian(N=1000, p=30, dx=dx, xi=xi, t=t, eta=eta, omega=omega)
        A_inv_mod, B_mod = fn.Crank_Nicholson_Matrices(N=1000, dt=dt_mod, H=H_t)

        psi_0_temp = fn.Crank_Nicholson_Step(psi_0_temp, A_inv_mod, B_mod)
        psi_1_temp = fn.Crank_Nicholson_Step(psi_1_temp, A_inv_mod, B_mod)
        psi_2_temp = fn.Crank_Nicholson_Step(psi_2_temp, A_inv_mod, B_mod)

        t += dt_mod
        V_n = V_n_1
    
    Modulation_time = (time_array_2 * omega) / (2 * np.pi)

    fig10 = plt.figure(figsize=(10, 8))
    ax10 = fig10.add_subplot(1, 1, 1)
    ax10.plot(Modulation_time, np.abs(c_0_mod), label='|c_0(t)|')
    ax10.plot(Modulation_time, np.abs(c_1_mod), label='|c_1(t)|')
    ax10.plot(Modulation_time, np.abs(c_2_mod), label='|c_2(t)|')
    ax10.set_xlabel('t / $T_M$')
    ax10.legend()
    ax10.grid(True)
    plt.savefig(f'assets/Plot10_Time_Evolution_Modulated_Potential_eta_{eta}.png', bbox_inches='tight')
    plt.close()