"""
BEHAVIORS
Comportements du robot.
 
Rôle :
- Combiner les fonctions de navigation pour produire des actions
- Générer les commandes moteur (linear + angular)
 
Contient :
- navigate() : déplacement principal (évitement + cible)
- circle_object() : tourner autour de l’objet
- escape() : sortir d’un blocage
 
Entrées : scan, yaw
Sorties : cmd_vel = {"linear": x, "angular": y}
"""
# Ludo : À modifier les schémas de fonctions car peu clair et pas adapter
 
import numpy as np
 
def navigate(theta_goal):
    K = 1.0
    angular = np.clip(K * theta_goal, -1.0, 1.0)
 
    # FIX #8 — couplage vitesse linéaire / angulaire.
    # Vitesse constante à 0.5 m/s même en virage serré est problématique
    # (dérapage, imprécision). On réduit la vitesse linéaire proportionnellement
    # à la correction angulaire.
    linear = 0.5 * (1.0 - 0.5 * abs(angular))
 
    return {
        "linear": linear,
        "angular": angular
    }
 
 
def stop():
    return {"linear": 0.0, "angular": 0.0}
 
 
def escape(scan):
    distances = scan[:, 0]
    angles = scan[:, 1]
 
    left_mask = angles > 0
    right_mask = angles < 0
    # FIX #9b — vérification de l'espace arrière avant de reculer.
    # L'original reculait toujours à -0.2 m/s sans vérifier les obstacles derrière.
    rear_mask = np.abs(angles) > (2 * np.pi / 3)
 
    # FIX #9a — médiane au lieu de moyenne.
    # Un seul rayon aberrant (bruit, réflexion parasite) peut fortement biaiser
    # la moyenne et choisir la mauvaise direction d'évasion.
    left = np.median(distances[left_mask]) if np.any(left_mask) else 0.0
    right = np.median(distances[right_mask]) if np.any(right_mask) else 0.0
 
    rear_clear = (
        np.percentile(distances[rear_mask], 10) > 0.3
        if np.any(rear_mask)
        else False
    )
 
    turn_dir = 1 if left > right else -1
    linear = -0.2 if rear_clear else 0.0
 
    return {
        "linear": linear,
        "angular": 0.5 * turn_dir
    }
 
 
def scan_behavior(scan_eff, desired_distance=1.0):
    distances = scan_eff[:, 0]
    angles = scan_eff[:, 1]
 
    idx = np.argmin(distances)
    min_dist = distances[idx]
    theta_obj = angles[idx]
 
    K_linear = 0.3
    K_angular = 1.0
 
    # FIX #10a — contrôle proportionnel au lieu de bang-bang.
    # np.sign(theta_obj) produit ±1 uniquement : le robot tourne toujours
    # à pleine vitesse angulaire, ce qui cause des oscillations.
    # Le contrôle proportionnel est stable et permet un suivi doux.
 
    # FIX #10b — logique d'orbite réelle.
    # L'original faisait un yo-yo face à l'objet (pas d'orbite).
    # On décompose le contrôle en deux axes indépendants :
    #   - radial  : maintien de la distance à l'objet (linear)
    #   - tangentiel : l'objet est maintenu à 90° sur le côté gauche (angular)
    # Ce schéma produit une orbite circulaire stable.
    orbit_angle = np.pi / 2
    linear = K_linear * (min_dist - desired_distance)
    angular = K_angular * (theta_obj - orbit_angle)
 
    linear = np.clip(linear, -0.3, 0.3)
    angular = np.clip(angular, -1.0, 1.0)
 
    return {"linear": linear, "angular": angular}
 
 
def retour_base(theta_goal):
    # FIX #7 — comportement distinct de navigate().
    # L'original était une copie exacte, ce qui rendait l'état RETOUR_BASE inutile.
    # On utilise ici un gain réduit et une vitesse plus basse pour une approche
    # de base plus douce et contrôlée.
    K = 0.8
    angular = np.clip(K * theta_goal, -0.8, 0.8)
    linear = 0.3 * (1.0 - 0.5 * abs(angular))
 
    return {
        "linear": linear,
        "angular": angular
    }