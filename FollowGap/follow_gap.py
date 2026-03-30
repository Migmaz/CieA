

import numpy as np

class FollowGap:
    """
    Implémentation complète de l'algorithme Follow-The-Gap pour navigation LiDAR.
    Pipeline:
        preprocess → threshold → safety bubble → find gap → best point
    """
    def __init__(
        self,
        max_range=10.0,
        min_range=0.05,
        fov=(-np.pi, np.pi),
        smooth_window=5,
        bubble_radius=16,
        threshold=2.0,
        conv_size=80,
        weight_goal=0.7
    ):
        self.max_range = max_range
        self.min_range = min_range
        self.fov = fov
        self.smooth_window = smooth_window
        self.bubble_radius = bubble_radius
        self.threshold = threshold
        self.conv_size = conv_size
        self.weight_goal = weight_goal

    def preprocess_lidar(self, scan:np.ndarray) -> np.ndarray:
        """
        Prétraite un scan LiDAR Nx2 (distance, angle).

        Args:
            scan (np.ndarray): Scan du LiDAR [distance, angle] Nx2

        Returns:
            np.ndarray: Scan filtré Nx2 (distance, angle)
        """
        scan = scan.copy()

        distances = scan[:, 0]
        angles = scan[:, 1]

        # Filtrage des NaN et les inf du scan LiDAR
        invalid_mask = np.isnan(distances) | np.isinf(distances)
        distances[invalid_mask] = self.max_range

        # Filtrage des distances max et min
        distances = np.clip(distances, self.min_range, self.max_range)

        # Filtrage du champ de vision
        fov_mask = (angles >= self.fov[0]) & (angles <= self.fov[1])
        distances[~fov_mask] = 0

        # Lissage
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            distances = np.convolve(distances, kernel, mode='same')

        processed_scan = np.stack((distances, angles), axis=1)

        return processed_scan

    def obstacle_threshold(self, processed_scan: np.ndarray) -> np.ndarray:
        """
        Filtre les points du scan LiDAR en appliquant un seuil minimal de distance.

        Tous les points avec une distance inférieure à `threshold` sont considérés comme des obstacles ou invalides.

        Args:
            processed_scan (np.ndarray): Scan LiDAR prétraité Nx2 [distance, angle]

        Returns:
            np.ndarray: Scan prétraité avec les obstacles Nx2 [distance, angle]
        """
        processed_scan = processed_scan.copy()
        processed_scan[processed_scan[:, 0] < self.threshold, 0] = 0
        return processed_scan

    def safety_buble(self, processed_scan:np.ndarray) -> np.ndarray:
        """
        Applique une bulle de sécurité autour de l'obstacle le plus proche dans un scan LiDAR.

        Tous les points dans un rayon d'indices défini autour de l'obstacle le plus proche 
        sont mis à zéro pour indiquer une zone non navigable.

        Args:
            processed_scan (np.ndarray): Scan LiDAR avec le traitement d'obstacle Nx2 [distance, angle]

        Returns:
            np.ndarray: Scan du LiDAR filtrée avec la bulle de sécurité appliqué
        """
        processed_scan = processed_scan.copy()

        distances = processed_scan[:, 0]
        valid = distances > 0

        if not np.any(valid):
            return processed_scan

        closest = np.argmin(np.where(valid, distances, np.inf))

        min_index = max(0, closest - self.bubble_radius)
        max_index = min(len(processed_scan) - 1, closest + self.bubble_radius)

        processed_scan[min_index:max_index + 1, 0] = 0

        return processed_scan

    def find_best_gap(self, scan_filter:np.ndarray, theta_goal:float) -> tuple[int,int]:
        """
        Identifie le meilleur gap dans un scan LiDAR Nx2 [distance, angle] en combinant la distance au goal et la longueur du gap.

        Args:
            scan_filter (np.ndarray): Scan LiDAR filtré Nx2 [distance, angle] (0 = obstacle ou invalide)
            theta_goal (float): Angle cible (rad)

        Returns:
            tuple[int,int]: (start_index, stop_index) du meilleur gap
        """
        distances = scan_filter[:, 0]
        angles = scan_filter[:, 1]

        valid = distances > 0
        diff = np.diff(valid.astype(int))

        gap_starts = np.where(diff == 1)[0] + 1
        gap_stops  = np.where(diff == -1)[0] + 1

        if valid[0]:
            gap_starts = np.insert(gap_starts, 0, 0)
        if valid[-1]:
            gap_stops = np.append(gap_stops, len(valid))

        if len(gap_starts) == 0:
            return None, None

        gap_lens = gap_stops - gap_starts
        gap_lens_norm = gap_lens / np.max(gap_lens)

        middles = ((gap_starts + gap_stops) // 2).astype(int)
        middles = np.clip(middles, 0, len(distances) - 1)

        delta_theta = angles[middles] - theta_goal
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
        delta_theta = np.abs(delta_theta)

        goal_score = np.exp(-delta_theta * 2)

        # distance moyenne du gap (amélioration)
        gap_dist = np.array([
            np.mean(distances[s:e]) if e > s else 0
            for s, e in zip(gap_starts, gap_stops)
        ])
        gap_dist_norm = gap_dist / (np.max(gap_dist) + 1e-6)

        scores = (
            self.weight_goal * goal_score +
            0.3 * gap_lens_norm +
            0.3 * gap_dist_norm
        )

        scores += 1e-6 * gap_lens

        best_idx = np.argmax(scores)
        return gap_starts[best_idx], gap_stops[best_idx]

    def find_best_point(self, start_i:int, end_i:int, scan_filter:np.ndarray) -> int:
        """
        Identifie le point optimal à l'intérieur d'un gap LiDAR donné en utilisant une moyenne mobile.

        Cette fonction cherche le point correspondant à la plus grande distance 
        à l'intérieur du gap, en lissant les distances avec une fenêtre de convolution
        pour éviter les variations locales trop brusques.

        Args:
            start_i (int): Index du début du gap.
            end_i (int): Index de fin du gap.
            scan_filter (np.ndarray): Scan LiDAR prétraité avec les obstacles Nx2 [distance, angle]

        Returns:
            int: Index du point optimal à l'intérieur du gap.
        """
        gap_size = end_i - start_i

        if gap_size <= 1:
            return start_i

        conv_size = min(self.conv_size, gap_size)

        if conv_size < 3:
            return (start_i + end_i) // 2

        kernel = np.ones(conv_size) / conv_size
        averaged_max_gap = np.convolve(scan_filter[start_i:end_i, 0], kernel, 'same')

        return averaged_max_gap.argmax() + start_i

    def compute(self, scan:np.ndarray, theta_goal:float):
        """
        Exécute l'algorithme Follow Gap complet.

        Args:
            scan (np.ndarray): Scan LiDAR brut Nx2 [distance, angle]
            theta_goal (float): Angle cible (rad)

        Returns:
            tuple: (best_index, best_angle, processed_scan)
        """
        scan = self.preprocess_lidar(scan)
        scan = self.obstacle_threshold(scan)
        scan = self.safety_buble(scan)

        start_i, end_i = self.find_best_gap(scan, theta_goal)

        if start_i is None:
            return None, None, scan

        best_i = self.find_best_point(start_i, end_i, scan)
        best_angle = scan[best_i, 1]

        return best_i, best_angle, scan