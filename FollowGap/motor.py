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


import time
from pynput import keyboard




time_int = time.time()
key_temp = None
while True :
    with keyboard.Events() as events:
    # Block at most one second
        event = events.get(0.1)
        if event is None:
            print(key_temp)
        else:
            print('Received event {}'.format(event))
            print(event.key)
            key_temp = event.key
            if event.key == keyboard.Key.esc :
                break
    print(time.time()-time_int)
    print("test")
    
    