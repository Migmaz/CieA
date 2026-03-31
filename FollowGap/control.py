"""
CONTROL (FSM)
Gestion des états du robot.

Rôle :
- Décider quel comportement utiliser
- Gérer les transitions entre états

Exemples d’états :
- NAVIGATE
- SCAN
- ESCAPE
- RETURN

Entrées : données capteurs
Sortie : nom du behavior à exécuter
"""

# Ludo : À modifier les schémas de fonctions car peu clair et pas adapter


def update_state(scan, yaw, current_state):
    """
    Met à jour l'état du robot (FSM).

    Args:
        scan (list of float):
            Données LiDAR.
        yaw (float):
            Orientation actuelle.
        current_state (str):
            État actuel du robot.

    Returns:
        new_state (str):
            Nouvel état (ex: "NAVIGATE", "SCAN", "ESCAPE", "RETURN")
    """
    pass