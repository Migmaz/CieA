
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

from FTG import FollowGap #Ramène la classe FollowGap

import numpy as np

def navigate(theta_goal):
    """
    Comportement principal : navigation vers la cible avec évitement.

    Args:
        scan (list of float):
            Données LiDAR.
        yaw (float):
            Orientation actuelle (rad).
        k (float):
            Poids de l'influence de la cible.

    Returns:
        cmd_vel (dict):
            {
                "linear": float (m/s),
                "angular": float (rad/s)
            }
    """

    return {
        "linear": 0.5,
        "angular": theta_goal
    }


def stop():
    """
    Détecte une situation dangereuse à partir du scan LiDAR.

    Args:
        scan (np.ndarray): Scan LiDAR Nx2 [distance, angle]
        distance_critique (float): distance minimale sécuritaire (m)

    Returns:
        dict ou None:
            {"linear": 0.0, "angular": 0.0} si danger
            None sinon
    """
    return {"linear": 0.0, "angular": 0.0}


def escape(prev_state):
    """
    Comportement d'urgence si le robot est bloqué.

    Returns:
        cmd_vel (dict):
            Commande pour tourner ou reculer.
    """

    # 1. Trop proche → reculer
    if prev_state == "STOP":

        return {"linear": -0.2, "angular": 0.3}

    # 2. CUL-DE-SAC → tourner doucement
    if prev_state == "CUL-DE-SAC":
        return {"linear": 0.0, "angular": 0.5}
    

def scan(scan_eff, desired_distance=1.0):
    """
    Fait tourner le robot autour d'un objet à distance constante.
    
    Args:
        scan (np.ndarray Nx2): Scan LiDAR [distance, angle]
        desired_distance (float): Distance cible à maintenir (m)
    
    Returns:
        cmd_vel (dict): Commande moteur {"linear": ..., "angular": ...}
    """
    
    distances = scan_eff[:, 0]
    angles = scan_eff[:, 1]

    # Distance minimale et angle correspondant
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    theta_obj = angles[min_dist_idx]

    # Gains simples pour contrôler vitesse
    K_linear = 0.3
    K_angular = 0.5

    # Correction linéaire pour garder distance désirée
    linear = K_linear * (min_dist - desired_distance)

    # Correction angulaire pour tourner autour
    # Si l'objet est à gauche → tourner dans le sens horaire, etc.
    # On peut juste tourner autour avec une valeur fixe
    angular = K_angular * np.sign(theta_obj)  # positive = tourner gauche, negative = droite

    # Limiter vitesses
    linear = np.clip(linear, -0.3, 0.3) #la vitesse linéaire (avance/recul) est forcée à rester entre -0.3 m/s (recule) et +0.3 m/s (avance).
    angular = np.clip(angular, -0.5, 0.5) #la vitesse angulaire (rotation) est limitée entre -0.5 rad/s et +0.5 rad/s (rotation horaire ou antihoraire).

    return {"linear": linear, "angular": angular}


def retour_base(theta_goal):

    return {"linear": 0.5, "angular": theta_goal}