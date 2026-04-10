"""
ACTUATION
Contrôle des moteurs.

Rôle :
- Convertir cmd_vel (linear, angular) en signaux moteurs
- Envoyer les commandes via PWM (ex: PCA9685)

Entrée :
- cmd_vel = {"linear": x, "angular": y}

Sortie :
- signaux moteurs
"""

# Ludo : À modifier les schémas de fonctions car peu clair et pas adapter


def send_command(cmd_vel):

    # envoyer cmd_vel aux moteurs ici
               
        """
        Envoie une commande aux moteurs.

    Args:
        cmd_vel (dict):
            {
                "linear": float (m/s),
                "angular": float (rad/s)
            }

    Returns:
        None

    Effet:
        Convertit en PWM et contrôle les moteurs.
    """
 
pass
