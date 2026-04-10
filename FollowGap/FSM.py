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

import numpy as np
from dataclasses import dataclass, field

@dataclass
class RobotState:
    prev_state: str = "NAVIGATE"
    theta_history: list = field(default_factory=list)
    stuck_counter: int = 0
    prev_dist_to_goal: float = None

def update_state(scan_true: np.ndarray, theta_goal: float, robot_pos: tuple, goal_pos: tuple, state: RobotState) -> str:
    """
    Met à jour l'état du robot (FSM) sans globals.
    
    Args:
        scan_true (np.ndarray): Nx2 [distance, angle]
        theta_goal (float): angle vers le goal
        robot_pos (tuple): (x, y)
        goal_pos (tuple): (x, y)
        state (RobotState): objet contenant l'état et historiques

    Returns:
        str: nouvel état
    """
    distances = scan_true[:,0]
    
    # --- État STOP ---
    if np.min(distances) <= 0.15:
        state.prev_state = "STOP"
        return "STOP"
    
    # --- calcul distance au goal et progrès ---
    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(robot_pos))
    if state.prev_dist_to_goal is None:
        state.prev_dist_to_goal = dist_to_goal
    progress = state.prev_dist_to_goal - dist_to_goal
    state.prev_dist_to_goal = dist_to_goal

    # --- État cul-de-sac ---
    if theta_goal is not None:
        state.theta_history.append(theta_goal)
    if len(state.theta_history) > 10:
        state.theta_history.pop(0)
    theta_var = np.var(state.theta_history) if len(state.theta_history) > 1 else 0
    free_ratio = np.sum(distances > 1.0) / len(distances)

    if progress < 0.01 and theta_var > 0.3 and free_ratio < 0.3:
        state.stuck_counter += 1
    else:
        state.stuck_counter = 0

    if state.stuck_counter > 5:
        state.prev_state = "CUL-DE-SAC"
        return "CUL-DE-SAC"

    # --- État ESCAPE ---
    if state.prev_state in ["STOP", "CUL-DE-SAC"]:
        state.prev_state = "ESCAPE"
        return "ESCAPE"

    # --- État SCAN ---
    if dist_to_goal <= 1.0:
        state.prev_state = "SCAN"
        return "SCAN"

    # --- État RETOUR_BASE ---
    if state.prev_state == "SCAN":
        state.prev_state = "RETOUR_BASE"
        return "RETOUR_BASE"

    # --- NAVIGATE par défaut ---
    state.prev_state = "NAVIGATE"
    return "NAVIGATE"