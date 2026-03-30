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
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def simulate_lidar(position, obstacles, n_rays=360, max_range=10.0):
    angles = np.linspace(-np.pi, np.pi, n_rays)
    distances = np.full(n_rays, max_range)

    for i, theta in enumerate(angles):
        direction = np.array([np.cos(theta), np.sin(theta)])

        for obs in obstacles:
            oc = position - obs["center"]

            b = 2 * np.dot(direction, oc)
            c = np.dot(oc, oc) - obs["radius"]**2
            delta = b**2 - 4*c

            if delta >= 0:
                t = (-b - np.sqrt(delta)) / 2
                if 0 < t < distances[i]:
                    distances[i] = t

    return np.column_stack((distances, angles))

# =================================
# Mini Benchmark / Simulation + complète
# =================================

obstacles = [
    {"center": np.array([4, 0]), "radius": 1},
    {"center": np.array([0, 5]), "radius": 1},
    {"center": np.array([0, -3]), "radius": 1},
     {"center": np.array([4, 4]), "radius": 0.5},
]

goal = np.array([6, 6])

if __name__ == "__main__":
    from tool import trans_lidar
    from follow_gap import FollowGap
    
    pts, theta, blblbl = generate_multi_gap_lidar_xyz()
    scan = trans_lidar(pts)
    
    fg = FollowGap()
    
    idx,angle,debug_scan = fg.compute(scan,np.pi/2)
    print(idx)
    print(debug_scan[idx], angle)
    
    def run_simulation():
        pos = np.array([0.0, 0.0])
        heading = 0.0

        trajectory = [pos.copy()]
        scans = []

        for _ in range(100):
            scan = simulate_lidar(pos, obstacles)

            # repère robot
            scan[:, 1] -= heading
            scan[:, 1] = (scan[:, 1] + np.pi) % (2*np.pi) - np.pi

            theta_goal = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
            theta_goal -= heading
            theta_goal = (theta_goal + np.pi) % (2*np.pi) - np.pi

            idx, theta_target, debug_scan = fg.compute(scan, theta_goal)

            if theta_target is None:
                break

            # dynamique
            max_turn_rate = 0.3
            angle_diff = (theta_target + np.pi) % (2*np.pi) - np.pi
            angle_diff = np.clip(angle_diff, -max_turn_rate, max_turn_rate)

            heading += angle_diff

            step_size = 0.2
            pos += step_size * np.array([np.cos(heading), np.sin(heading)])

            trajectory.append(pos.copy())
            scans.append(scan.copy())

            if np.linalg.norm(pos - goal) < 0.1:
                print("Goal reached!")
                break

        return np.array(trajectory), scans

    def animate_simulation(obstacles, goal):
        fig, ax = plt.subplots(figsize=(6,6))

        pos = np.array([0.0, 0.0])
        heading = 0.0

        traj = [pos.copy()]

        # éléments graphiques
        traj_line, = ax.plot([], [], '-b', label="Trajectory")
        robot_point = ax.scatter([], [], c='blue', s=50)
        lidar_points = ax.scatter([], [], s=2, alpha=0.2)

        # obstacles
        for obs in obstacles:
            circle = plt.Circle(obs["center"], obs["radius"], color='r', alpha=0.5)
            ax.add_patch(circle)

        # goal
        ax.scatter(goal[0], goal[1], c='g', s=100, label="Goal")

        ax.set_xlim(-2, 8)
        ax.set_ylim(-5, 8)
        ax.set_aspect('equal')
        ax.grid()
        ax.legend()

        def update(frame):
            nonlocal pos, heading, traj

            # 🔹 LiDAR
            scan = simulate_lidar(pos, obstacles)

            # repère robot
            scan[:,1] -= heading
            scan[:,1] = (scan[:,1] + np.pi) % (2*np.pi) - np.pi

            # angle goal
            theta_goal = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
            theta_goal -= heading
            theta_goal = (theta_goal + np.pi) % (2*np.pi) - np.pi

            # Follow Gap
            idx, theta_target, debug_scan = fg.compute(scan, theta_goal)

            if theta_target is None:
                return traj_line, robot_point, lidar_points

            # dynamique
            max_turn_rate = 0.3
            angle_diff = (theta_target + np.pi) % (2*np.pi) - np.pi
            angle_diff = np.clip(angle_diff, -max_turn_rate, max_turn_rate)

            heading += angle_diff

            # mouvement
            step_size = 0.2
            pos += step_size * np.array([np.cos(heading), np.sin(heading)])

            traj.append(pos.copy())
            
            if np.linalg.norm(pos - goal) < 0.1:
                print("Goal reached!")
                print(frame)
                anim.save("simulation(1).gif", writer="pillow", fps=10)
                anim.event_source.stop()

            # 🔹 update traj
            traj_np = np.array(traj)
            traj_line.set_data(traj_np[:,0], traj_np[:,1])

            # 🔹 robot
            robot_point.set_offsets(pos)

            # 🔹 LiDAR points (dans monde)
            x = pos[0] + scan[:,0] * np.cos(scan[:,1] + heading)
            y = pos[1] + scan[:,0] * np.sin(scan[:,1] + heading)
            lidar_points.set_offsets(np.c_[x,y])

            return traj_line, robot_point, lidar_points

        anim = FuncAnimation(fig, update, frames=150, interval=50, blit=True)

        plt.show()

        return anim
    
    animate_simulation(obstacles, goal)
    
    anim = animate_simulation(obstacles, goal)
    anim.save("simulation.gif", writer="pillow", fps=10)