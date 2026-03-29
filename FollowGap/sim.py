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
import numpy as np

def generate_multi_gap_lidar_xyz(
    num_points=360,
    max_range=10.0,
    noise_std=0.05
):
    angles = np.linspace(-np.pi, np.pi, num_points)
    ranges = np.ones(num_points) * max_range

    # 🔹 Étape 1 : créer des obstacles (zones bloquées)
    obstacles = [
        (-1.8, -1.2, 1.5),   # obstacle gauche
        (-0.8, -0.3, 2.0),   # obstacle centre gauche
        (0.2, 0.6, 1.2),     # obstacle centre droit
        (1.0, 1.6, 1.8)      # obstacle droite
    ]

    for (a_min, a_max, dist) in obstacles:
        mask = (angles > a_min) & (angles < a_max)
        ranges[mask] = dist + np.random.normal(0, noise_std, np.sum(mask))

    # 🔹 Étape 2 : créer des gaps (zones ouvertes)
    gaps = [
        (-0.2, 0.1),   # gap 1 (petit)
        (0.7, 0.9)     # gap 2 (plus large)
    ]

    for (a_min, a_max) in gaps:
        mask = (angles > a_min) & (angles < a_max)
        ranges[mask] = max_range - np.random.uniform(0, 1, np.sum(mask))

    # 🔹 Étape 3 : ajouter bruit global
    ranges += np.random.normal(0, noise_std, num_points)

    # 🔹 Étape 4 : clamp (éviter valeurs négatives)
    ranges = np.clip(ranges, 0.05, max_range)

    # 🔹 Étape 5 : conversion XYZ
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)

    points = np.stack((x, y, z), axis=1)

    return points, angles, ranges


if __name__ == "__main__":
    from tool import trans_lidar
    from follow_gap import preprocess_lidar, safety_buble, find_best_gap, apply_obstacle_threshold
    
    pts, theta, range = generate_multi_gap_lidar_xyz()
    scan = trans_lidar(pts)
    scan_filter = preprocess_lidar(scan)
    scan_filter = apply_obstacle_threshold(scan_filter)
    scan_proc = safety_buble(scan_filter)
    best_gap = find_best_gap(scan_proc,0)
    print(best_gap)
    