# Simulateur Quantique Simple pour Débutants
# Ce code simule le comportement d'une particule quantique

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================================
# PARTIE 1: CONFIGURATION DE BASE
# ============================================================================


longueur_totale = 2.0       
nombre_points = 1000         
pas_temps = 1e-6           


dx = longueur_totale / nombre_points    # Distance entre chaque point
dt = pas_temps                         
x = np.linspace(0, longueur_totale, nombre_points) 

print("=== SIMULATION QUANTIQUE POUR DÉBUTANTS ===")
print(f"Longueur du domaine: {longueur_totale}")
print(f"Nombre de points: {nombre_points}")
print(f"Résolution spatiale: {dx:.4f}")

# ============================================================================
# PARTIE 2: CRÉATION DU POTENTIEL (OBSTACLES POUR LA PARTICULE)
# ============================================================================

def creer_barriere_simple():
    """
    Crée une barrière de potentiel - comme un mur que la particule doit traverser
    """
    V = np.zeros(nombre_points) 
    
    # Position de la barrière
    debut_barriere = 0.8
    fin_barriere = 0.9
    hauteur_barriere = 2000  
    
    # On met la barrière aux bons endroits
    for i in range(nombre_points):
        if debut_barriere <= x[i] <= fin_barriere:
            V[i] = hauteur_barriere
    
    print(f"Barrière créée entre x={debut_barriere} et x={fin_barriere}")
    return V

def creer_puits_simple():
    """
    Crée un puits de potentiel - comme un trou où la particule peut être piégée
    """
    V = np.zeros(nombre_points)
    
    # Position du puits
    debut_puits = 0.4
    fin_puits = 0.6
    profondeur_puits = -1000  # Négatif = trou
    
    # On creuse le puits
    for i in range(nombre_points):
        if debut_puits <= x[i] <= fin_puits:
            V[i] = profondeur_puits
    
    print(f"Puits créé entre x={debut_puits} et x={fin_puits}")
    return V

# ============================================================================
# PARTIE 3: CRÉATION DE LA PARTICULE (PAQUET D'ONDES)
# ============================================================================

def creer_particule():
    """
    Crée notre particule quantique sous forme de paquet d'ondes
    """
    # Paramètres de la particule
    position_initiale = 0.3     
    largeur = 0.05              
    vitesse = 30               
    
    
    particule = np.zeros(nombre_points, dtype=complex)  
    
    for i in range(nombre_points):
       
        gaussienne = np.exp(-((x[i] - position_initiale)**2) / (2 * largeur**2))
        
        # Partie onde (oscillation)
        onde = np.exp(1j * vitesse * x[i])  # 1j = racine de -1
        
        # Notre particule = gaussienne × onde
        particule[i] = gaussienne * onde
    
    # Normalisation (pour que la probabilité totale = 1)
    norme = np.trapz(np.abs(particule)**2, x)
    particule = particule / np.sqrt(norme)
    
    print(f"Particule créée à la position {position_initiale}")
    return particule

# ============================================================================
# PARTIE 4: SIMULATION DE L'ÉVOLUTION DANS LE TEMPS
# ============================================================================

def simuler_evolution(particule_initiale, potentiel, duree=5000):
    """
    Fait évoluer notre particule dans le temps
    """
    print(f"Début de la simulation ({duree} étapes)...")
    
    # Préparation
    psi = particule_initiale.copy()
    partie_reelle = np.real(psi)      
    partie_imaginaire = np.imag(psi)  
    
    
    sauvegardes = []
    sauvegarde_tous_les = 100  
    
   
    s = dt / (dx**2)
    if s > 0.25:
        print(f"Attention! Paramètre s = {s:.4f} peut causer des instabilités")
    
    # BOUCLE PRINCIPALE - L'ÉVOLUTION DANS LE TEMPS
    for etape in range(duree):
        
       
        
        if etape % 2 == 0:  
            for i in range(1, nombre_points-1): 
                partie_imaginaire[i] = (partie_imaginaire[i] + 
                                      s * (partie_reelle[i+1] + partie_reelle[i-1]) - 
                                      2 * partie_reelle[i] * (s + potentiel[i] * dt))
        
        else:  
            for i in range(1, nombre_points-1):
                partie_reelle[i] = (partie_reelle[i] - 
                                  s * (partie_imaginaire[i+1] + partie_imaginaire[i-1]) + 
                                  2 * partie_imaginaire[i] * (s + potentiel[i] * dt))
        
       
        if etape % sauvegarde_tous_les == 0:
          
            psi_actuel = partie_reelle + 1j * partie_imaginaire
            densite = np.abs(psi_actuel)**2
            sauvegardes.append(densite.copy())
            
           
            if etape % (sauvegarde_tous_les * 10) == 0:
                pourcentage = (etape / duree) * 100
                print(f"Progression: {pourcentage:.1f}%")
    
    print("Simulation terminée!")
    return np.array(sauvegardes)

# ============================================================================
# PARTIE 5: RECHERCHE DES ÉTATS STATIONNAIRES (SIMPLE)
# ============================================================================

def trouver_etats_stationnaires_simple(potentiel):
    """
    Trouve les états où la particule peut rester stable
    """
    print("Recherche des états stationnaires...")
    
    # Construction de la matrice hamiltonienne 
    H = np.zeros((nombre_points, nombre_points))
    
    # Remplissage de la matrice
    for i in range(1, nombre_points-1):
        # Terme cinétique (dérivée seconde)
        H[i, i-1] = 1 / (dx**2)      # Voisin de gauche
        H[i, i] = -2 / (dx**2) + potentiel[i]  # Point central + potentiel
        H[i, i+1] = 1 / (dx**2)      # Voisin de droite
    
    # Conditions aux bords 
    H[0, 0] = 1
    H[-1, -1] = 1
    
    # Résolution du problème aux valeurs propres
    from scipy.linalg import eigh
    energies, fonctions_onde = eigh(H)
    
   
    n_etats = 5
    energies = energies[:n_etats]
    fonctions_onde = fonctions_onde[:, :n_etats]
    
    # Normalisation
    for i in range(n_etats):
        norme = np.trapz(fonctions_onde[:, i]**2, x)
        fonctions_onde[:, i] /= np.sqrt(norme)
    
    print(f"Premiers niveaux d'énergie trouvés:")
    for i in range(min(3, n_etats)):
        print(f"  E_{i} = {energies[i]:.2f}")
    
    return energies, fonctions_onde

# ============================================================================
# PARTIE 6: VISUALISATION
# ============================================================================

def dessiner_evolution(donnees_evolution, potentiel):
    """
    Crée une animation de l'évolution de la particule
    """
    print("Création de l'animation...")
    
    # Préparation de la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, longueur_totale)
    ax.set_ylim(0, np.max(donnees_evolution) * 1.2)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Probabilité de présence |ψ|²')
    ax.set_title('Évolution de la particule quantique')
    
  
    potentiel_affichage = potentiel / np.max(np.abs(potentiel)) * np.max(donnees_evolution) * 0.3
    ax.plot(x, potentiel_affichage, 'r-', label='Potentiel', alpha=0.7, linewidth=2)
    ax.legend()
    
   
    ligne, = ax.plot([], [], 'b-', linewidth=2, label='Particule')
    
    def init_animation():
        ligne.set_data([], [])
        return ligne,
    
    def animer(frame):
        ligne.set_data(x, donnees_evolution[frame])
        return ligne,
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animer, init_func=init_animation,
                                 frames=len(donnees_evolution), interval=50,
                                 blit=True, repeat=True)
    
    return fig, anim

def dessiner_etats_stationnaires(energies, fonctions_onde, potentiel):
    """
    Dessine les états stationnaires trouvés
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graphique du potentiel avec niveaux d'énergie
    ax1.plot(x, potentiel, 'r-', linewidth=2, label='Potentiel V(x)')
    
   
    couleurs = ['blue', 'green', 'orange', 'purple', 'brown']
    for i in range(len(energies)):
        couleur = couleurs[i % len(couleurs)]
        ax1.axhline(y=energies[i], color=couleur, linestyle='--', 
                   label=f'E_{i} = {energies[i]:.1f}')
    
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Énergie')
    ax1.set_title('Potentiel et niveaux d\'énergie')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique des fonctions d'onde
    for i in range(len(energies)):
        couleur = couleurs[i % len(couleurs)]
        ax2.plot(x, fonctions_onde[:, i], color=couleur, 
                label=f'État {i}', linewidth=2)
    
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Fonction d\'onde ψ(x)')
    ax2.set_title('États stationnaires')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# PARTIE 7: PROGRAMME PRINCIPAL
# ============================================================================

def simulation_complete():
    """
    Lance une simulation complète
    """
    print("Démarrage de la simulation complète...\n")
    
    # 1. Créer la particule et le potentiel
    print("1. Création de la particule et du potentiel")
    particule = creer_particule()
    potentiel_barriere = creer_barriere_simple()
    
    # 2. Simuler l'évolution
    print("\n2. Simulation de l'évolution temporelle")
    evolution = simuler_evolution(particule, potentiel_barriere, duree=3000)
    
    # 3. Trouver les états stationnaires avec un puits
    print("\n3. Recherche des états stationnaires")
    potentiel_puits = creer_puits_simple()
    energies, etats = trouver_etats_stationnaires_simple(potentiel_puits)
    
    # 4. Créer les graphiques
    print("\n4. Création des visualisations")
    
    # Animation de l'évolution
    fig1, anim = dessiner_evolution(evolution, potentiel_barriere)
    
    # États stationnaires
    fig2 = dessiner_etats_stationnaires(energies, etats, potentiel_puits)
    
    # Comparaison avant/après
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, evolution[0], 'b-', label='État initial', linewidth=2)
    ax.plot(x, evolution[-1], 'g-', label='État final', linewidth=2)
    
    # Potentiel normalisé
    pot_norm = potentiel_barriere / np.max(potentiel_barriere) * np.max(evolution) * 0.3
    ax.plot(x, pot_norm, 'r-', label='Barrière', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Densité de probabilité')
    ax.set_title('Comparaison: avant et après passage de la barrière')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    print("\n=== RÉSUMÉ ===")
    print("✓ Simulation de propagation terminée")
    print("✓ États stationnaires calculés")
    print("✓ Graphiques générés")
    print("\nLa particule quantique peut traverser la barrière même si")
    print("son énergie est plus faible - c'est l'effet tunnel!")
    
    return evolution, energies, etats

# ============================================================================
# LANCEMENT DU PROGRAMME
# ============================================================================

if __name__ == "__main__":
    # Lance la simulation si on exécute ce fichier directement
    resultats_evolution, niveaux_energie, fonctions_stationnaires = simulation_complete()
    
    print("\nVariables disponibles pour explorer:")
    print("- resultats_evolution: évolution de la particule dans le temps")
    print("- niveaux_energie: énergies des états stationnaires") 
    print("- fonctions_stationnaires: formes des états stationnaires")
    print("\nExpérimentez en changeant les paramètres au début du code!")
