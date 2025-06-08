import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

start_time = time.time()

# -----------------------------
# PARAMÈTRES GÉNÉRAUX
# -----------------------------
dt = 1E-7
dx = 0.001
nx = int(1/dx)*2
nt = 90000
nd = int(nt / 1000) + 1
n_frame = nd
s = dt / (dx**2)

# Paramètres du paquet initial
xc = 0.6
sigma = 0.1
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))

# Paramètres du potentiel
v0 = -4000
e = 5  # E/V0
E = e * v0
k = math.sqrt(2 * abs(E))

# -----------------------------
# ESPACE ET POTENTIEL
# -----------------------------
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o <= 0.9)]= v0  # Puits rectangulaire

# -----------------------------
# CONDITION INITIALE : paquet d'ondes gaussien
# -----------------------------

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite = np.zeros((nt, nx))
densite[0, :] = np.abs(cpt) ** 2
final_densite = np.zeros((n_frame, nx))
re = np.real(cpt)
im = np.imag(cpt)
b = np.zeros(nx)

# Zone de transmission
droite = o > 1.1
T_final = 0

# -----------------------------
# PROPAGATION NUMÉRIQUE DE L’ONDE
# -----------------------------
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)
    if (i - 1) % 1000 == 0:
        it += 1
        final_densite[it][:] = densite[i][:]
    if i == nt - 1:
        T_final = np.sum(re[droite]**2 + im[droite]**2) * dx

# -----------------------------
# ANIMATION
# -----------------------------
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j, :])
    return line,

fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(0, 13)
plt.xlim(0, 2)
plt.plot(o, V, label="Potentiel")
plt.title(f"Propagation d’un paquet d’onde — E/V₀ = {e}")
plt.xlabel("x")
plt.ylabel("Densité de probabilité")
plt.legend()
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frame, blit=False, interval=100, repeat=False)
plt.show()

# -----------------------------
# ÉTATS STATIONNAIRES
# -----------------------------
hbar = 1
m = 1
diag = np.full(nx, -2.0)
offdiag = np.full(nx - 1, 1.0)
laplacien = (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)) / dx**2
H = -(hbar**2 / (2 * m)) * laplacien + np.diag(V)

energies_stationnaires, vecteurs_stationnaires = np.linalg.eigh(H)
indices_bornes = np.where(energies_stationnaires < 0)[0][:50]
etats_stationnaires = []
for n in indices_bornes:
    psi_n = vecteurs_stationnaires[:, n]
    norm = np.sqrt(np.sum(np.abs(psi_n)**2) * dx)
    psi_n = psi_n / norm
    etats_stationnaires.append((energies_stationnaires[n], psi_n))

plt.figure()
for idx, (E_n, psi_n) in enumerate(etats_stationnaires):
    plt.plot(o, abs(psi_n**2) + E_n, label=f"État {idx+1}, E = {E_n:.1f}")
plt.plot(o, V, 'k--', label="Potentiel V(x)")
plt.xlabel("x")
plt.ylabel("Densité de probabilité + Énergie")
plt.title("États stationnaires")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# INFORMATIONS FINALES
# -----------------------------
T = T_final
R = 1 - T
end_time = time.time()
print(f"\nTemps d'exécution : {end_time - start_time:.2f} s")
print(f"• Énergie du paquet : E = {E:.2f}")
print(f"• Profondeur du puits : V₀ = {v0}")
print(f"\n• Énergies des états liés :")
for idx, (E_n, _) in enumerate(etats_stationnaires):
    print(f"  • État {idx+1} : E = {E_n:.2f}")
print(f"\n• Transmission T(E) = {T:.4f}")
print(f"• Réflexion R(E) = {R:.4f}")

