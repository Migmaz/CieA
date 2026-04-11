"""
MAPPING
Mémoire et position du robot.
 
Rôle :
- Enregistrer la trajectoire (path memory)
- Estimer la position du robot
- Permettre le retour au point de départ
 
Fonctions possibles :
- save_position(x, y)
- compute_return_path()
"""
 
def update_position(x, y, yaw, linear_speed, dt):
    """
    Met à jour la position estimée du robot.
 
    Args:
        x, y (float):
            Position actuelle.
        yaw (float):
            Orientation (rad).
        linear_speed (float):
            Vitesse avant (m/s).
        dt (float):
            Temps écoulé (s).
 
    Returns:
        (new_x, new_y) (tuple of float):
            Nouvelle position estimée.
    """
    pass
 
 
def save_path(path, x, y):
    """
    Enregistre la position dans le trajet.
 
    Args:
        path (list of tuple):
            Liste des positions (x, y).
        x, y (float):
            Position actuelle.
 
    Returns:
        path (list):
            Trajectoire mise à jour.
    """
    pass