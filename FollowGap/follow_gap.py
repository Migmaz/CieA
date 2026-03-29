

import numpy as np

def find_best_gap(scan_filter:np.ndarray, theta_goal:float, weight_goal=0.7) -> tuple[int,int]:
    """
    Identifie le meilleur gap dans un scan LiDAR en combinant la distance au goal et la longueur du gap.

    Args:
        scan_filter (np.ndarray): Scan LiDAR filtré (0 = obstacle ou invalide)
        theta_goal (float): Angle cible (rad)
        weight_goal (float, optional): Poids pour la priorité vers l'angle goal (0..1). Defaults to 0.7.

    Returns:
        tuple[int,int]: (start_index, stop_index) du meilleur gap
    """
    # Masquer les distances invalides
    mask = np.ma.masked_where(scan_filter == 0, scan_filter)
    all_gaps = np.ma.notmasked_contiguous(mask)

    if len(all_gaps) == 0:
        return None, None  # Pas de gap disponible

    # Calcul vectorisé des longueurs, milieux et delta_theta
    gap_starts = np.array([g.start for g in all_gaps])
    gap_stops = np.array([g.stop for g in all_gaps])
    gap_lens = gap_stops - gap_starts
    middles = ((gap_starts + gap_stops) / 2).astype(int)

    # Attention aux index hors limites
    middles = np.clip(middles, 0, len(scan_filter)-1)

    delta_theta = np.abs(scan_filter[middles] - theta_goal)

    # Calcul du score vectorisé
    scores = weight_goal * (1 / (1 + delta_theta)) + (1 - weight_goal) * gap_lens

    # Sélection du meilleur gap
    best_idx = np.argmax(scores)
    best_gap = all_gaps[best_idx]

    return best_gap.start, best_gap.stop
    
def find_best_point(start_i, end_i, scan_filter):
        averaged_max_gap = np.convolve(scan_filter[start_i:end_i], np.ones(BEST_POINT_CONV_SIZE),'same') /BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i
    
