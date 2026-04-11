"""
CONTROL (FSM)
Gestion des états du robot.

Rôle :
- Décider quel comportement utiliser
- Gérer les transitions entre états

États :
- NAVIGATE     : déplacement principal vers l'objectif
- SCAN         : inspection de l'objectif à l'arrivée
- ESCAPE       : sortir d'un blocage simple
- CUL-DE-SAC   : blocage persistant détecté
- RETOUR_BASE  : retour au point de départ
- STOP         : arrêt d'urgence (obstacle très proche)
- BASE_REACHED : mission complète (scan + retour à la base)

Entrées : données capteurs
Sortie : nom du behavior à exécuter
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque


# =========================
# ===== ROBOT STATE =======
# =========================

@dataclass
class RobotState:
    prev_state: str = "NAVIGATE"

    # Historique
    theta_history: deque = field(default_factory=lambda: deque(maxlen=10))
    progress_history: deque = field(default_factory=lambda: deque(maxlen=10))
    prev_dist_to_goal: float = None

    # Stuck detection
    stuck_counter: int = 0

    # Timers
    escape_timer: int = 0
    scan_timer: int = 0

    # RETOUR_BASE : persistance jusqu'à arrivée à la base
    returning: bool = False

    # Seuils d'arrivée (m)
    goal_radius: float = 0.3
    base_radius: float = 0.3


# =========================
# ===== MAIN FSM ==========
# =========================

def update_state(
    scan: np.ndarray,
    theta_goal: float,
    robot_pos: tuple,
    goal_pos: tuple,
    state: RobotState,
    base_pos: tuple = (0.0, 0.0),
    has_valid_gap: bool = True
    
) -> str:
    """
    Met à jour la FSM et retourne l'état courant.

    Args:
        scan         : np.ndarray Nx2 [distance, angle]
        theta_goal   : float, angle vers l'objectif (rad)
        robot_pos    : tuple (x, y), position courante
        goal_pos     : tuple (x, y), position de l'objectif
        state        : RobotState, état interne du robot
        base_pos     : tuple (x, y), position de départ (défaut (0, 0))
        has_valid_gap: bool, Follow-the-Gap a trouvé un gap valide

    Returns:
        str : nom de l'état courant
    """

    distances = scan[:, 0]
    angles = scan[:, 1]

    # =========================
    # === DONNÉES VIDES =======
    # =========================

    if len(distances) == 0:
        return "STOP"

    # =========================
    # === MASQUES =============
    # =========================

    front_mask = np.abs(angles) < np.pi / 2
    local_mask = front_mask & (distances < 1.2)

    front_dist = distances[front_mask]
    local_dist = distances[local_mask]

    if len(front_dist) == 0:
        return "STOP"

    # =========================
    # === DISTANCES ===========
    # =========================

    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(robot_pos))
    dist_to_base = np.linalg.norm(np.array(base_pos) - np.array(robot_pos))

    # =========================
    # === BASE_REACHED ========
    # Mission complète : scan terminé + retour à la base
    # =========================

    if state.returning and dist_to_base < state.base_radius:
        state.returning = False
        state.prev_state = "BASE_REACHED"
        return "BASE_REACHED"

    # =========================
    # === STOP (sécurité) =====
    # =========================

    if np.percentile(front_dist, 10) < 0.2:
        state.prev_state = "STOP"
        state.escape_timer = 8
        return "STOP"

    # =========================
    # === ESCAPE ==============
    # =========================
    if state.escape_timer > 0 or not has_valid_gap:
        state.escape_timer = max(state.escape_timer, 6)
        state.escape_timer -= 1
        state.prev_state = "ESCAPE"
        return "ESCAPE"

    # =========================
    # === PROGRÈS VERS GOAL ===
    # =========================

    if state.prev_dist_to_goal is None:
        state.prev_dist_to_goal = dist_to_goal

    progress = state.prev_dist_to_goal - dist_to_goal
    state.prev_dist_to_goal = dist_to_goal
    state.progress_history.append(progress)
    cumulative_progress = sum(state.progress_history)

    # =========================
    # === HISTORIQUE ANGLE ====
    # =========================

    if theta_goal is not None:
        state.theta_history.append(theta_goal)

    if len(state.theta_history) > 1:
        thetas = np.array(state.theta_history)
        theta_var = 1.0 - np.abs(np.mean(np.exp(1j * thetas)))
    else:
        theta_var = 0.0

    # =========================
    # === ENVIRONNEMENT LOCAL =
    # =========================

    if len(local_dist) > 0:
        free_ratio = np.sum(local_dist > 0.8) / len(local_dist)
    else:
        free_ratio = 0.0

    is_blocked = np.percentile(front_dist, 30) < 0.5

    # =========================
    # === CUL-DE-SAC ==========
    # =========================

    if cumulative_progress < 0.05 and is_blocked:
        state.stuck_counter += 1
    else:
        state.stuck_counter = max(0, state.stuck_counter - 1)

    if state.stuck_counter > 5:
        state.escape_timer = 10
        state.stuck_counter = 0
        state.prev_state = "CUL-DE-SAC"
        return "CUL-DE-SAC"

    # =========================
    # === RETOUR_BASE =========
    # Persiste jusqu'à arrivée à la base
    # =========================

    if state.returning:
        state.prev_state = "RETOUR_BASE"
        return "RETOUR_BASE"

    # =========================
    # === SCAN ================
    # Déclenché à l'arrivée à l'objectif
    # =========================

    if dist_to_goal < state.goal_radius and state.scan_timer == 0:
        state.scan_timer = 20

    if state.scan_timer > 0:
        state.scan_timer -= 1
        state.prev_state = "SCAN"
        return "SCAN"

    # Fin du SCAN → déclenche le retour à la base
    if state.prev_state == "SCAN" and state.scan_timer == 0:
        state.returning = True
        state.prev_state = "RETOUR_BASE"
        return "RETOUR_BASE"

    # =========================
    # === NAVIGATE ============
    # =========================

    state.prev_state = "NAVIGATE"
    return "NAVIGATE"