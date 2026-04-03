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
        yaw (float): Orientation du rover (rad) -> fct IMU (sensor)

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

def trans_to_rover(scan: np.ndarray, pitch: float, translation: tuple[float,float,float] =(0,0,0)) -> np.ndarray :
    """
    Transforme un scan LiDAR [r, theta] en points 3D dans le repère du rover
    
    Étapes :
    1. Conversion polaire -> cartésien (repère LiDAR)
    2. Rotation autour de l'axe X (pitch du LiDAR)
    3. Translation dans le repère du rover

    Args:
        scan (np.ndarray): Scan du LiDAR [distance, angle] Nx2
        pitch (float): Inclinaison du LiDAR (rad)
        translation (tuple[float,float,float], optional): Position du LiDAR par rapport au rover. Defaults to (0,0,0).

    Returns:
        np.ndarray: Points [x, y, z] Nx3 dans le repère du rover
    """
    # Transforme en 3D
    r = scan[:,0]
    theta = scan[:,1]
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    
    # Fait la rotation et translation pour remettre sur le plan rover (xy)
    cx, sx = np.cos(pitch), np.sin(pitch)
    tx, ty, tz = translation
    
    x_rot = x + tx
    y_rot = cx * y - sx * z + ty
    z_rot = sx * y + cx * z + tz
    
    return np.stack((x_rot, y_rot, z_rot), axis=1)


def filter_ground(pts_rot: np.ndarray, z_min=0.05) -> np.ndarray:
    """
    Supprime les points correspondant au sol en se basant sur la hauteur z

    Args:
        pts_rot (np.ndarray): Points [x, y, z] Nx3 dans le repère du rover
        z_min (float, optional): Seuil minimum pour considérer un point comme sol. Defaults to 0.05.

    Returns:
        np.ndarray: Points filtrer [x, y, z] où z > z_min
    """
    return pts_rot[pts_rot[:,2] > z_min]

def compute_scan(pts_rot: np.ndarray, z_min = 0.05, z_max = 0.5) ->  np.ndarray:
    """
    Génère un scan 2D pondéré à partir des points 3D
    
    Étapes :
    1. Projection sur le plan XY
    2. Calcul des distances et angles
    3. Pondération des distances selon la hauteur (z)

    Args:
        pts_rot (np.ndarray): Points [x, y, z] Nx3 dans le repère du rover
        z_min (float, optional): Hauteur minimal pour commencer à considérer un obstacle. Defaults to 0.05.
        z_max (float, optional): Hauteur maximal (normalisation du poids). Defaults to 0.5.

    Returns:
        np.ndarray: Scan 2D pondéré [distance, angle] Nx2
    """
    x = pts_rot[:,0]
    y = pts_rot[:,1]
    z = pts_rot[:,2]
    
    dist = np.hypot(x,y)
    theta = np.arctan2(y,x)
    
    weight = np.clip((z-z_min) / (z_max - z_min), 0, 1)
    
    dist_weight = dist * weight
    
    return np.column_stack(dist_weight, theta)