"""
SIMULATION
Tests sans robot réel.

Rôle :
- Générer de faux scans LiDAR
- Tester les behaviors et la navigation
- Permettre le debug rapide

Très utile pour :
- développer sans hardware
- valider les algorithmes
"""

def generate_fake_scan():
    """
    Génère un scan LiDAR simulé.

    Returns:
        scan (list of float):
            Données simulées avec obstacles.
    """
    pass


def test_navigation():
    """
    Test complet du pipeline sans robot réel.

    Étapes :
    - Génère un scan
    - Simule un yaw
    - Appelle navigate()
    - Affiche la commande

    Returns:
        None
    """
    pass