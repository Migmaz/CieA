"""
NAVIGATION
Fonctions algorithmiques pures.

Rôle :
- Calculer la direction d’évitement (Follow-the-Gap)
- Calculer la direction vers la cible (angle cible)

Important :
- Ne contrôle PAS directement le robot
- Ne dépend PAS des états (FSM)

Entrées : données capteurs
Sorties : angles (directions)
"""

def follow_the_gap(scan, threshold=0.8):
    """
    Calcule la meilleure direction pour éviter les obstacles.

    Args:
        scan (lst of float):
            Distances LiDAR en mètres.
        threshold (float):
            Distance minimale pour considérer un espace libre.

    Returns:
        angle_gap (float):
            Angle en radians relatif au robot.
            0 = devant, + = gauche, - = droite
    """
    
    pass

def angle_correction(theta_algo:float, theta_goal:float, alpha = 0.3) -> float:
    """Correction de l'angle donnée par l'algo Follow the gap 
    afin d'avoir un biais vers l'objectif pour s'assurer de s'en approcher

    Args:
        theta_algo (float): Angle (rad) donnée par l'algo Follow the gap -> Fct Follow_the_gap
        theta_goal (float): Angle (rad) du goal par rapport au rover -> Fct theta_goal (tool)
        alpha (float, optional): Poid de correction. 0 -> Aucune correction, 1 -> Fait toujours face objectif. Defaults to 0.3.

    Returns:
        float: Angle (rad) corriger
    """
    return (1-alpha) * theta_algo + alpha * theta_goal