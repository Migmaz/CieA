"""
MAIN
Chef d’orchestre du projet.

Rôle :
- Boucle principale du robot
- Lit les données capteurs (sensor.py)
- Demande à control.py quel comportement utiliser
- Exécute le behavior correspondant (behaviors.py)
- Envoie la commande aux moteurs (actuation.py)
"""

# =================================================
# Importation de librairy
# =================================================
import time
import numpy as np

# =================================================
# Importation de fonctions
# =================================================


# =================================================
# Code
# =================================================