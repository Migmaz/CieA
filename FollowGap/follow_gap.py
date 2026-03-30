
import numpy as np


class FollowGap:
    """
    Implémentation améliorée (V2) de Follow-The-Gap.

    Améliorations:
        - Masque valide au lieu de zéros destructifs
        - Safety bubble vectorisée sur tous les obstacles
        - Meilleur scoring des gaps
        - Best point orienté vers le goal
        - Fusion goal + gap pour convergence globale
    """

    def __init__(
        self,
        max_range=10.0,
        min_range=0.05,
        smooth_window=5,
        bubble_radius=16,
        threshold=2.0,
        conv_size=80,
        weight_goal=0.6,
        weight_dist=0.3,
        weight_len=0.3,
        alpha_point=1.0,
        alpha_final=0.7
    ):
        self.max_range = max_range
        self.min_range = min_range
        self.smooth_window = smooth_window
        self.bubble_radius = bubble_radius
        self.threshold = threshold
        self.conv_size = conv_size

        self.weight_goal = weight_goal
        self.weight_dist = weight_dist
        self.weight_len = weight_len

        self.alpha_point = alpha_point
        self.alpha_final = alpha_final

    def preprocess_lidar(self, scan: np.ndarray) -> np.ndarray:
        """
        Nettoie et lisse le scan LiDAR.
        """
        scan = scan.copy()

        distances = scan[:, 0]
        angles = scan[:, 1]

        invalid = np.isnan(distances) | np.isinf(distances)
        distances[invalid] = self.max_range

        distances = np.clip(distances, self.min_range, self.max_range)

        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            distances = np.convolve(distances, kernel, mode='same')

        return np.stack((distances, angles), axis=1)

    def build_valid_mask(self, scan: np.ndarray) -> np.ndarray:
        """
        Construit un masque des points navigables.
        """
        return scan[:, 0] > self.threshold

    def safety_bubble(self, scan: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Applique une dilatation autour des obstacles (vectorisée).
        """
        obstacle_mask = ~valid

        kernel = np.ones(2 * self.bubble_radius + 1)
        inflated = np.convolve(obstacle_mask.astype(float), kernel, mode='same') > 0

        return valid & (~inflated)

    def find_best_gap(self, scan: np.ndarray, valid: np.ndarray, theta_goal: float):
        """
        Sélectionne le meilleur gap avec scoring multi-critères.
        """
        distances = scan[:, 0]
        angles = scan[:, 1]

        diff = np.diff(valid.astype(int))

        starts = np.where(diff == 1)[0] + 1
        stops  = np.where(diff == -1)[0] + 1

        if valid[0]:
            starts = np.insert(starts, 0, 0)
        if valid[-1]:
            stops = np.append(stops, len(valid))

        if len(starts) == 0:
            return None, None

        lengths = stops - starts
        lengths_norm = lengths / (np.max(lengths) + 1e-6)

        # moyenne distance vectorisée
        cumsum = np.cumsum(distances)
        sums = cumsum[stops - 1] - np.concatenate(([0], cumsum[starts[:-1] - 1]))
        means = sums / (lengths + 1e-6)
        means_norm = means / (np.max(means) + 1e-6)

        centers = ((starts + stops) // 2).astype(int)
        centers = np.clip(centers, 0, len(angles) - 1)

        delta = angles[centers] - theta_goal
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        goal_score = np.exp(-np.abs(delta))

        scores = (
            self.weight_goal * goal_score +
            self.weight_len  * lengths_norm +
            self.weight_dist * means_norm
        )

        best = np.argmax(scores)

        return starts[best], stops[best]

    def find_best_point(self, scan: np.ndarray, start: int, stop: int, theta_goal: float):
        """
        Sélectionne le meilleur point dans le gap avec biais directionnel.
        """
        distances = scan[start:stop, 0]
        angles = scan[start:stop, 1]

        if len(distances) <= 1:
            return start

        k = min(self.conv_size, len(distances))
        if k < 3:
            return (start + stop) // 2

        kernel = np.ones(k) / k
        smooth = np.convolve(distances, kernel, mode='same')

        delta = angles - theta_goal
        delta = np.arctan2(np.sin(delta), np.cos(delta))

        score = smooth - self.alpha_point * np.abs(delta)

        return np.argmax(score) + start

    def compute(self, scan: np.ndarray, theta_goal: float):
        """
        Pipeline complet avec fusion goal + gap.
        """
        scan = self.preprocess_lidar(scan)

        valid = self.build_valid_mask(scan)
        valid = self.safety_bubble(scan, valid)

        start, stop = self.find_best_gap(scan, valid, theta_goal)

        if start is None:
            return None, None, scan

        best_i = self.find_best_point(scan, start, stop, theta_goal)

        theta_gap = scan[best_i, 1]

        # fusion goal + gap
        theta_final = (
            self.alpha_final * theta_gap +
            (1 - self.alpha_final) * theta_goal
        )

        # limitation angle
        theta_final = np.clip(theta_final, -np.pi/2, np.pi/2)

        return best_i, theta_final, scan