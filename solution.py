import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
import time

class RamsauerTownsendSimulation:
    def __init__(self):
        # Paramètres par défaut
        self.dt = 1E-7
        self.dx = 0.001
        self.nx = int(2/self.dx)  
        self.nt = 90000
        self.nd = int(self.nt/1000) + 1
        
        
        self.v0 = -4000  # Profondeur du puits 
        self.e = 5  # Rapport E/V0
        self.E = self.e * self.v0
        self.k = np.sqrt(2 * abs(self.E))
        
        
        self.xc = 0.6  
        self.sigma = 0.05  
        self.A = 1/(np.sqrt(self.sigma * np.sqrt(np.pi)))
        
        
        self.x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        self.V = np.zeros(self.nx)
        
        
        self.setup_potential()
        
    def setup_potential(self):
        """Configure le potentiel (puits carré vers le bas)"""
       
        self.V[(self.x >= 0.8) & (self.x <= 0.9)] = self.v0
        
    def initial_wavepacket(self):
        """Crée le paquet d'ondes initial (gaussien)"""
        return self.A * np.exp(1j * self.k * self.x - 
                              ((self.x - self.xc) ** 2) / (2 * (self.sigma ** 2)))
    
    def propagate_wavepacket(self):
        """
        Algorithme de résolution d'équation différentielle pour la propagation
        du paquet d'ondes (méthode des différences finies explicites)
        """
        print("Début de la propagation du paquet d'ondes...")
        start_time = time.time()
        
        
        s = self.dt / (self.dx ** 2)
        
        
        psi = self.initial_wavepacket()
        re = np.real(psi)
        im = np.imag(psi)
        
       
        densities = np.zeros((self.nd, self.nx))
        densities[0, :] = np.abs(psi) ** 2
        
        # Buffer temporaire
        b = np.zeros(self.nx)
        
       
        frame_idx = 1
        for i in range(1, self.nt):
            if i % 2 != 0: 
                b[1:-1] = im[1:-1]  # Sauvegarde pour le calcul de densité
                im[1:-1] = (im[1:-1] + s * (re[2:] + re[:-2]) - 
                           2 * re[1:-1] * (s + self.V[1:-1] * self.dt))
                
                # Sauvegarde de la densité tous les 1000 pas
                if (i - 1) % 1000 == 0 and frame_idx < self.nd:
                    densities[frame_idx, 1:-1] = (re[1:-1] ** 2 + 
                                                 im[1:-1] * b[1:-1])
                    frame_idx += 1
                    
            else:  
                re[1:-1] = (re[1:-1] - s * (im[2:] + im[:-2]) + 
                           2 * im[1:-1] * (s + self.V[1:-1] * self.dt))
        
        elapsed_time = time.time() - start_time
        print(f"Propagation terminée en {elapsed_time:.2f} secondes")
        
        return densities
    
    def find_stationary_states(self, n_states=10):
        """
        Algorithme pour trouver les états stationnaires
        (résolution du problème aux valeurs propres)
        """
        print("Recherche des états stationnaires...")
        start_time = time.time()
        
        # Construction de la matrice hamiltonienne discrétisée
        # H = -1/2 * d²/dx² + V(x)
        
        # Matrice de dérivée seconde (différences finies centrées)
        H = np.zeros((self.nx, self.nx))
        
       
        for i in range(1, self.nx - 1):
            H[i, i] = -2 / (self.dx ** 2) + self.V[i]
            
        
        for i in range(1, self.nx - 1):
            if i > 0:
                H[i, i-1] = 1 / (self.dx ** 2)
            if i < self.nx - 1:
                H[i, i+1] = 1 / (self.dx ** 2)
        
        # Multiplication par -1/2 (unités atomiques)
        H *= -0.5
        
        # Conditions aux bords 
        H[0, 0] = 1e10  # Grande valeur pour forcer psi(0) = 0
        H[-1, -1] = 1e10  # Grande valeur pour forcer psi(L) = 0
        
        # Résolution du problème aux valeurs propres
        eigenvalues, eigenvectors = eigh(H)
        
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
       
        for i in range(min(n_states, len(eigenvalues))):
            norm = np.trapz(eigenvectors[:, i] ** 2, self.x)
            eigenvectors[:, i] /= np.sqrt(norm)
            
            
            center_idx = len(self.x) // 2
            if eigenvectors[center_idx, i] < 0:
                eigenvectors[:, i] *= -1
        
        elapsed_time = time.time() - start_time
        print(f"États stationnaires calculés en {elapsed_time:.2f} secondes")
        print(f"Premières énergies propres: {eigenvalues[:5]}")
        
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    
    def animate_propagation(self, densities):
        """Animation de la propagation du paquet d'ondes"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(frame):
            line.set_data(self.x, densities[frame, :])
            ax.set_title(f'Propagation du paquet d\'ondes - Frame {frame}')
            return line,
        
        line, = ax.plot([], [], 'b-', linewidth=2, label='Densité de probabilité')
        
       
        # Normalisation pour affichage 
        V_plot = self.V / abs(np.min(self.V)) * np.max(densities) * 0.3
        ax.plot(self.x, V_plot, 'r-', linewidth=2, label='Potentiel (normalisé)')
        
        ax.set_xlim(0, 2)
        ax.set_ylim(np.min(V_plot) * 1.1, np.max(densities) * 1.1)  # Ajusté pour voir le puits
        ax.set_xlabel('Position x')
        ax.set_ylabel('Densité de probabilité')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                    frames=self.nd, interval=100, 
                                    blit=False, repeat=True)
        
        return fig, ani
    
    def plot_stationary_states(self, eigenvalues, eigenvectors, n_plot=5):
        """Affichage des états stationnaires"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique 1: Fonctions d'onde
        for i in range(min(n_plot, len(eigenvalues))):
            ax1.plot(self.x, eigenvectors[:, i], 
                    label=f'État {i+1}, E = {eigenvalues[i]:.2f}')
        
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Fonction d\'onde ψ(x)')
        ax1.set_title('États stationnaires dans le puits de potentiel')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Densités de probabilité
        for i in range(min(n_plot, len(eigenvalues))):
            ax2.plot(self.x, eigenvectors[:, i] ** 2, 
                    label=f'État {i+1}, E = {eigenvalues[i]:.2f}')
        
      
        V_plot = self.V / abs(np.min(self.V)) * np.max(eigenvectors ** 2) * 0.5
        ax2.plot(self.x, V_plot, 'k--', linewidth=2, label='Potentiel (normalisé)')
        
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Densité de probabilité |ψ(x)|²')
        ax2.set_title('Densités de probabilité des états stationnaires')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_transmission(self, eigenvalues):
        """Analyse de la transmission et recherche de l'effet Ramsauer-Townsend"""
        print("\nAnalyse de l'effet Ramsauer-Townsend:")
        print("=" * 50)
        
        # États liés (E < 0) et états de diffusion (E > 0)
        bound_states = eigenvalues[eigenvalues < 0]
        scattering_states = eigenvalues[eigenvalues > 0]
        
        print(f"Nombre d'états liés: {len(bound_states)}")
        print(f"Énergies des états liés: {bound_states}")
        print(f"Nombre d'états de diffusion calculés: {len(scattering_states)}")
        
        if len(scattering_states) > 0:
            print(f"Premières énergies de diffusion: {scattering_states[:5]}")
        
        return bound_states, scattering_states

def main():
    """Fonction principale"""
    print("Simulation de l'effet Ramsauer-Townsend")
    print("=" * 40)
    
    # Création de l'instance de simulation
    sim = RamsauerTownsendSimulation()
    
   
    print("\n1. PROPAGATION DU PAQUET D'ONDES")
    print("-" * 40)
    densities = sim.propagate_wavepacket()
    
    
    print("\n2. CALCUL DES ÉTATS STATIONNAIRES")
    print("-" * 40)
    eigenvalues, eigenvectors = sim.find_stationary_states(n_states=20)
    
   
    print("\n3. ANALYSE DES RÉSULTATS")
    print("-" * 40)
    bound_states, scattering_states = sim.analyze_transmission(eigenvalues)
    
    
    print("\n4. AFFICHAGE DES RÉSULTATS")
    print("-" * 40)
    
    # Graphiques des états stationnaires
    fig_states = sim.plot_stationary_states(eigenvalues, eigenvectors, n_plot=6)
    
    # Animation de la propagation
    print("Création de l'animation...")
    fig_anim, ani = sim.animate_propagation(densities)
    
    # Graphique du potentiel seul 
    fig_pot, ax_pot = plt.subplots(figsize=(10, 6))
    ax_pot.plot(sim.x, sim.V, 'r-', linewidth=3, label='Potentiel V(x)')
    ax_pot.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax_pot.set_xlabel('Position x')
    ax_pot.set_ylabel('Potentiel V(x)')
    ax_pot.set_title('Puits de potentiel carré (vers le bas)')
    ax_pot.legend()
    ax_pot.grid(True, alpha=0.3)
    
    plt.show()
    
    print("\nSimulation terminée!")
    print("Fermer les fenêtres graphiques pour continuer...")

if __name__ == "__main__":
    main()
