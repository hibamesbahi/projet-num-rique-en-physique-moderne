import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.optimize import minimize_scalar
import time

class QuantumSimulator:
    def __init__(self, nx=2000, L=2.0, dt=1e-7):
        """
        Simulateur quantique pour l'équation de Schrödinger
        
        Parameters:
        - nx: nombre de points spatiaux
        - L: longueur du domaine
        - dt: pas de temps
        """
        self.nx = nx
        self.L = L
        self.dx = L / nx
        self.dt = dt
        self.x = np.linspace(0, L, nx)
        self.s = dt / (self.dx**2)  # paramètre de stabilité
        
       
        if self.s > 0.5:
            print(f"Attention: s = {self.s:.4f} > 0.5, instabilité numérique possible")
    
    def create_potential(self, potential_type="barrier", **kwargs):
        """
        Crée différents types de potentiels
        """
        V = np.zeros(self.nx)
        
        if potential_type == "barrier":
            # Barrière de potentiel
            x_start = kwargs.get('x_start', 0.8)
            x_end = kwargs.get('x_end', 0.9)
            height = kwargs.get('height', -4000)
            V[(self.x >= x_start) & (self.x <= x_end)] = height
            
        elif potential_type == "well":
            # Puits de potentiel
            x_start = kwargs.get('x_start', 0.4)
            x_end = kwargs.get('x_end', 0.6)
            depth = kwargs.get('depth', -2000)
            V[(self.x >= x_start) & (self.x <= x_end)] = depth
            
        elif potential_type == "harmonic":
            # Oscillateur harmonique
            x0 = kwargs.get('x0', self.L/2)
            k = kwargs.get('k', 1000)
            V = 0.5 * k * (self.x - x0)**2
            
        elif potential_type == "double_well":
            # Double puits
            x1 = kwargs.get('x1', 0.3)
            x2 = kwargs.get('x2', 0.7)
            width = kwargs.get('width', 0.1)
            depth = kwargs.get('depth', -1000)
            barrier_height = kwargs.get('barrier_height', 500)
            
          
            V[(self.x >= x1-width/2) & (self.x <= x1+width/2)] = depth
          
            V[(self.x >= x2-width/2) & (self.x <= x2+width/2)] = depth
           
            V[(self.x >= x1+width/2) & (self.x <= x2-width/2)] = barrier_height
            
        return V
    
    def create_wave_packet(self, x0=0.3, sigma=0.05, k0=50):
        """
        Crée un paquet d'ondes gaussien
        """
        A = 1 / (sigma * np.sqrt(np.sqrt(np.pi)))
        psi = A * np.exp(1j * k0 * self.x - ((self.x - x0)**2) / (2 * sigma**2))
        return psi
    
    def propagate_wave_packet(self, psi_initial, V, nt=90000, save_every=1000):
        """
        Propage le paquet d'ondes en utilisant la méthode des différences finies
        """
        psi = psi_initial.copy()
        re_part = np.real(psi)
        im_part = np.imag(psi)
        
        # Stockage des densités de probabilité
        n_saves = nt // save_every + 1
        densities = np.zeros((n_saves, self.nx))
        densities[0, :] = np.abs(psi)**2
        
        save_index = 1
        
        print(f"Propagation du paquet d'ondes ({nt} itérations)...")
        start_time = time.time()
        
        for i in range(1, nt):
            if i % 2 != 0: 
                im_part[1:-1] = (im_part[1:-1] + 
                               self.s * (re_part[2:] + re_part[:-2]) - 
                               2 * re_part[1:-1] * (self.s + V[1:-1] * self.dt))
            else:  
                re_part[1:-1] = (re_part[1:-1] - 
                               self.s * (im_part[2:] + im_part[:-2]) + 
                               2 * im_part[1:-1] * (self.s + V[1:-1] * self.dt))
            
          
            if i % save_every == 0 and save_index < n_saves:
                psi_current = re_part + 1j * im_part
                densities[save_index, :] = np.abs(psi_current)**2
                save_index += 1
                
                if i % (save_every * 10) == 0:
                    progress = i / nt * 100
                    print(f"Progression: {progress:.1f}%")
        
        elapsed_time = time.time() - start_time
        print(f"Propagation terminée en {elapsed_time:.2f} secondes")
        
        return densities
    
    def find_stationary_states(self, V, n_states=10):
        """
        Trouve les états stationnaires en résolvant l'équation aux valeurs propres
        """
        print("Recherche des états stationnaires...")
        
       
        main_diag = -2 * np.ones(self.nx) / (self.dx**2)
        off_diag = np.ones(self.nx-1) / (self.dx**2)
        
        # Conditions aux limites nulles
        H = diags([off_diag, main_diag + V, off_diag], [-1, 0, 1]).toarray()
        H[0, :] = 0
        H[0, 0] = 1
        H[-1, :] = 0
        H[-1, -1] = 1
        
        # Résolution du problème aux valeurs propres
        eigenvalues, eigenvectors = eigh(H)
        
        # Tri par énergie croissante
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normalisation des fonctions d'onde
        for i in range(min(n_states, len(eigenvalues))):
            norm = np.trapz(eigenvectors[:, i]**2, self.x)
            eigenvectors[:, i] /= np.sqrt(norm)
            
            # Ajustement du signe pour plus de cohérence
            if eigenvectors[self.nx//4, i] < 0:
                eigenvectors[:, i] *= -1
        
        print(f"États trouvés. Énergies des {min(n_states, 5)} premiers états:")
        for i in range(min(n_states, 5)):
            print(f"  E_{i} = {eigenvalues[i]:.2f}")
        
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    
    def analyze_tunneling(self, E, V, barrier_start, barrier_end):
        """
        Analyse analytique de l'effet tunnel
        """
        V_barrier = np.max(V[(self.x >= barrier_start) & (self.x <= barrier_end)])
        
        if E >= V_barrier:
            print("Énergie supérieure à la barrière - pas d'effet tunnel")
            return None
        
       
        kappa = np.sqrt(2 * abs(E - V_barrier)) 
        width = barrier_end - barrier_start
        
        # Coefficient de transmission (approximation)
        T = np.exp(-2 * kappa * width)
        R = 1 - T
        
        print(f"Analyse de l'effet tunnel:")
        print(f"  Largeur de barrière: {width:.3f}")
        print(f"  Coefficient κ: {kappa:.2f}")
        print(f"  Transmission T: {T:.4f}")
        print(f"  Réflexion R: {R:.4f}")
        
        return {'T': T, 'R': R, 'kappa': kappa}
    
    def animate_propagation(self, densities, V, title="Propagation du paquet d'ondes"):
        """
        Crée une animation de la propagation
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Configuration du graphique
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, np.max(densities) * 1.1)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Densité de probabilité |ψ|²')
        ax.set_title(title)
        
        # Tracé du potentiel (normalisé pour l'affichage)
        V_normalized = V / np.max(np.abs(V)) * np.max(densities) * 0.2
        ax.plot(self.x, V_normalized, 'r-', label='Potentiel (normalisé)', alpha=0.7)
        ax.legend()
        
        line, = ax.plot([], [], 'b-', linewidth=2)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(frame):
            line.set_data(self.x, densities[frame, :])
            return line,
        
        # Création de l'animation
        ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                    frames=len(densities), blit=True, 
                                    interval=100, repeat=True)
        
        return fig, ani
    
    def plot_stationary_states(self, energies, wavefunctions, V, n_plot=5):
        """
        Trace les états stationnaires
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique du potentiel
        ax1.plot(self.x, V, 'r-', linewidth=2, label='Potentiel V(x)')
        
        # Graphique des énergies
        for i in range(min(n_plot, len(energies))):
            ax1.axhline(y=energies[i], color=f'C{i}', linestyle='--', alpha=0.7, 
                       label=f'E_{i} = {energies[i]:.1f}')
        
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Énergie')
        ax1.set_title('Potentiel et niveaux d\'énergie')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique des fonctions d'onde
        for i in range(min(n_plot, len(energies))):
            # Décalage vertical pour la visualisation
            psi_shifted = wavefunctions[:, i] + energies[i] * 0.001
            ax2.plot(self.x, psi_shifted, label=f'ψ_{i}', linewidth=2)
        
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Fonction d\'onde')
        ax2.set_title('États stationnaires')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main_simulation():
    """
    Fonction principale de simulation
    """
    print("=== SIMULATEUR QUANTIQUE ===")
    
   
    sim = QuantumSimulator(nx=2000, L=2.0, dt=1e-7)
    
    # 1. Simulation de propagation d'un paquet d'ondes
    print("\n1. Simulation de propagation d'un paquet d'ondes")
    
    # Création du potentiel (barrière)
    V = sim.create_potential("barrier", x_start=0.8, x_end=0.9, height=-4000)
    
    # Création du paquet d'ondes initial
    psi_0 = sim.create_wave_packet(x0=0.3, sigma=0.05, k0=50)
    
   
    E = 25**2 / 2  
    tunnel_data = sim.analyze_tunneling(E, V, 0.8, 0.9)
    
   
    densities = sim.propagate_wave_packet(psi_0, V, nt=30000, save_every=500)
    
    # 2. Recherche des états stationnaires
    print("\n2. Recherche des états stationnaires")
    
    # Potentiel puits pour les états liés
    V_well = sim.create_potential("well", x_start=0.4, x_end=0.6, depth=-2000)
    energies, wavefunctions = sim.find_stationary_states(V_well, n_states=10)
    
    # 3. Visualisations
    print("\n3. Génération des graphiques")
    
    
    fig_anim, ani = sim.animate_propagation(densities, V, 
                                           "Effet tunnel - Propagation du paquet d'ondes")
    
    # États stationnaires
    fig_states = sim.plot_stationary_states(energies, wavefunctions, V_well)
    
    # Graphique de comparaison finale
    fig_comp, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sim.x, densities[0, :], 'b-', label='État initial', linewidth=2)
    ax.plot(sim.x, densities[-1, :], 'g-', label='État final', linewidth=2)
    ax.plot(sim.x, V/np.max(np.abs(V)) * np.max(densities) * 0.2, 'r-', 
            label='Potentiel (normalisé)', alpha=0.7)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Densité de probabilité')
    ax.set_title('Comparaison états initial et final')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    return sim, densities, energies, wavefunctions

if __name__ == "__main__":
    
    simulator, wave_data, eigen_energies, eigen_states = main_simulation()
    
    print("\nSimulation terminée. Utilisez les variables suivantes pour l'analyse:")
    print("- simulator: instance du simulateur")
    print("- wave_data: données de propagation du paquet d'ondes")
    print("- eigen_energies: énergies des états stationnaires")
    print("- eigen_states: fonctions d'onde des états stationnaires")
