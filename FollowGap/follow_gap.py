

import numpy as np

def find_best_gap(scan_filter:np.ndarray, theta_goal:float, weight_goal=0.7) -> tuple[int,int]:
    """
    Identifie le meilleur gap dans un scan LiDAR Nx2 [distance, angle] en combinant la distance au goal et la longueur du gap.

    Args:
        scan_filter (np.ndarray): Scan LiDAR filtré Nx2 [distance, angle] (0 = obstacle ou invalide)
        theta_goal (float): Angle cible (rad)
        weight_goal (float, optional): Poids pour la priorité vers l'angle goal (0..1). Defaults to 0.7.

    Returns:
        tuple[int,int]: (start_index, stop_index) du meilleur gap
    """
    distances = scan_filter[:, 0]
    angles = scan_filter[:, 1]

    # --- 1. Détection des gaps (zones > 0)
    valid = distances > 0

    # transitions
    diff = np.diff(valid.astype(int))

    gap_starts = np.where(diff == 1)[0] + 1
    gap_stops  = np.where(diff == -1)[0] + 1

    # gérer bords
    if valid[0]:
        gap_starts = np.insert(gap_starts, 0, 0)
    if valid[-1]:
        gap_stops = np.append(gap_stops, len(valid))

    if len(gap_starts) == 0:
        return None, None

    # --- 2. Largeur du gap
    gap_lens = gap_stops - gap_starts
    gap_lens_norm = gap_lens / np.max(gap_lens)

    middles = ((gap_starts + gap_stops) // 2).astype(int)
    middles = np.clip(middles, 0, len(distances) - 1)

    # --- 3. Différence avec l'angle du goal
    delta_theta = angles[middles] - theta_goal
    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
    delta_theta = np.abs(delta_theta)

    # --- 4. Score
    scores = (
        weight_goal * np.exp(-delta_theta * 2) +
        (1 - weight_goal) * gap_lens_norm
    )

    # --- 5. Tie-break deterministic (IMPORTANT)
    # ajoute un biais très faible pour stabiliser
    scores += 1e-6 * gap_lens

    best_idx = np.argmax(scores)
    return gap_starts[best_idx], gap_stops[best_idx]
    

def find_best_point(start_i:int, end_i:int, scan_filter:np.ndarray, conv_size = 80) -> int:
        averaged_max_gap = np.convolve(scan_filter[start_i:end_i], np.ones(conv_size),'same') /conv_size
        return averaged_max_gap.argmax() + start_i
    
def preprocess_lidar(
    scan:np.ndarray,
    max_range = 10.0,
    min_range = 0.05,
    fov = (-np.pi, np.pi),
    smooth_window = 5,
                     ) -> np.ndarray:
    """
    Prétraite un scan LiDAR Nx2 (distance, angle).

    Args:
        scan (np.ndarray): Scan du LiDAR [distance, angle] Nx2
        max_range (float, optional): Distance maximal valide. Defaults to 10.0.
        min_range (float, optional): Distance minimale valide. Defaults to 0.05.
        fov (tuple, optional): Champs de vision à converser (rad). Defaults to (-np.pi, np.pi).
        smooth_window (int, optional): Taille de la fenêtre pour le lissage (doit être impair). Defaults to 5.

    Returns:
        np.ndarray: Scan filtré Nx2 (distance, angle)
    """
    scan = scan.copy()

    distances = scan[:, 0]
    angles = scan[:, 1]

    # Filtrage des NaN et les inf du scan LiDAR
    invalid_mask = np.isnan(distances) | np.isinf(distances)
    distances[invalid_mask] = max_range

    # Filtrage des disntance max et min
    distances = np.clip(distances, min_range, max_range)

    # Filtrage du champ de vision (uniquement devant nous)
    fov_mask = (angles >= fov[0]) & (angles <= fov[1])
    distances = distances[fov_mask]
    angles = angles[fov_mask]

    # Lissage
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        distances = np.convolve(distances, kernel, mode='same')

    processed_scan = np.stack((distances, angles), axis=1)

    return processed_scan

def safety_buble(processed_scan:np.ndarray, bubble_radius=16) -> np.ndarray:
    
    processed_scan = processed_scan.copy()
    closest = np.argmin(processed_scan[:,0])
    
    min_index = closest - bubble_radius
    max_index = closest + bubble_radius
    
    if min_index < 0 : 
        min_index = 0
    if max_index >= len(processed_scan): 
        max_index = len(processed_scan) - 1
    
    processed_scan[min_index:max_index + 1, 0] = 0
    
    return processed_scan

def apply_obstacle_threshold(scan: np.ndarray, threshold=2.0) -> np.ndarray:
    scan = scan.copy()
    scan[scan[:, 0] < threshold, 0] = 0
    return scan