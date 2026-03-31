import numpy as np
import math

def normalize_angle(theta: float) -> float:
    """
    Normalise un angle en radians dans l'intervalle [-pi, pi]

    Args:
        theta (float): Angle (rad)

    Returns:
        float: angle normalisé dans [-pi, pi]
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi

def theta_goal(Pr:list[float], Pg:list[float],yaw:float) -> float:
    """Calcul le theta par rapport au rover vers le goal

    Args:
        Pr (list): Coordonné (x,y) du rover (m) 
        Pg (list): Coordonnée (x,y) du goal (m)
        yaw (float): Orientation du rover (rad) -> fct get_yaw (sensor)

    Returns:
        float: Angle (rad) par rapport au rover vers le goal
    """
    x, y = Pr[0], Pr[1] 
    x_goal, y_goal = Pg[0], Pg[1]
    
    dx = x_goal - x
    dy = y_goal -y
    
    theta_global = np.arctan2(dy,dx)
    theta_goal = normalize_angle(theta_global - yaw)
    
    return theta_goal

def scan_process(scan: np.ndarray, theta_lidar=0.0, translation=(0,0,0)) -> np.ndarray:
    """
    Transforme un scan LiDAR brut vers le plan du rover.

    Args:
        scan (np.ndarray): Scan du LiDAR [x, y, z], shape Nx3
        theta_lidar (float, optional): Angle d'inclinaison du LiDAR (rad). Defaults to 0.0.
        translation (tuple, optional): Position du LiDAR dans le rover (tx, ty, tz). Defaults to (0,0,0).

    Returns:
        np.ndarray: Scan formaté [distance, theta], shape Nx2
    """
    tx, ty, tz = translation

    # Rotation autour de l'axe X (inclinée)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta_lidar), -np.sin(theta_lidar)],
        [0, np.sin(theta_lidar),  np.cos(theta_lidar)]
    ])
    t = np.array([tx, ty, tz])

    pts_rover = (R @ scan.T).T + t

    x_r = pts_rover[:,0]
    y_r = pts_rover[:,1]

    distances = np.sqrt(x_r**2 + y_r**2)
    theta = np.arctan2(y_r, x_r)

    scan_proc = np.stack([distances, theta], axis=1)
    return scan_proc