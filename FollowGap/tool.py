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

def trans_theta(r : float, theta : float) -> tuple[float,float]:
    """Transformation géométrique du plan de référence du LiDAR incliné vers celui du rover

    Args:
        r (float): distance r (m) obtenu du scan du LiDAR
        theta (float): Angle (rad) associé à une distance r obtenu du scan du LiDAR

    Returns:
        tuple: Retourne la distance (m) et l'angle (rad) transformer au repère du rover
    """
    theta_lidar = np.deg2rad(10)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta) * np.cos(theta_lidar)
    
    d = np.sqrt(x**2 + y**2)  
    theta_trans = np.atan2(y,x)
    
    return d,theta_trans

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