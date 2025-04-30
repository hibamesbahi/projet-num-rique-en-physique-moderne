import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
nx = 1000
x = np.linspace(0, (nx-1)*dx, nx)

# Potentiel en escalier par exemple
V = np.zeros(nx)
V[(x >= 0.8) & (x <= 0.9)] = -4000

# Construction de la matrice Hamiltonienne H
diag = np.zeros(nx)
offdiag = np.ones(nx - 1)

kinetic = (-1 / (dx**2)) * (np.diag(offdiag, -1) + np.diag(offdiag, 1) - 2 * np.diag(np.ones(nx)))
potential = np.diag(V)
H = -0.5 * kinetic + potential

# Résolution des valeurs propres
eigvals, eigvecs = np.linalg.eigh(H)

# Affichage des premiers états stationnaires
plt.figure()
for i in range(3):  # Affiche les 3 premiers états
    plt.plot(x, eigvecs[:, i]**2, label=f"E = {eigvals[i]:.2f}")
plt.plot(x, V / 5000, label="V(x)/5000", color='k')  # Potentiel rescalé pour visualisation
plt.legend()
plt.title("États stationnaires")
plt.xlabel("x")
plt.ylabel("|ψ(x)|²")
plt.grid()
plt.show()

