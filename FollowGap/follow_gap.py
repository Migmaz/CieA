
import numpy as np


class FollowGap:
    """
    Identifie le meilleur gap dans un scan LiDAR Nx2 [distance, angle] en combinant la distance au goal et la longueur du gap.
    Implémentation complète de l'algorithme Follow-The-Gap pour navigation LiDAR.
    Pipeline:
        preprocess → threshold → safety bubble → find gap → best point
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
        Prétraite un scan LiDAR Nx2 [distance, angle].
        
        Nettoie les données en remplaçant les valeurs invalides (NaN, inf),
        applique un clipping des distances et un lissage optionnel pour réduire le bruit.

        Args:
            scan (np.ndarray): Scan LiDAR brut Nx2 [distance, angle]

        Returns:
            np.ndarray: Scan LiDAR prétraité Nx2 [distance, angle]
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
        Construit un masque de points navigables basé sur un seuil de distance.
        
        Tous les points avec une distance supérieure au seuil sont considérés comme valides
        et potentiellement navigables.

        Args:
            scan (np.ndarray): Scan LiDAR prétraité Nx2 [distance, angle]

        Returns:
            np.ndarray: Masque booléen Nx1 (True = libre, False = obstacle)
        """
        return scan[:, 0] > self.threshold

    def safety_bubble(self, scan: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Applique une bulle de sécurité autour des obstacles détectés dans le scan LiDAR.
        
        Cette fonction dilate les zones occupées (obstacles) afin de créer une marge de sécurité
        autour de ceux-ci. Tous les points situés dans un rayon d'indices défini autour des obstacles
        sont marqués comme non navigables.

        Args:
            scan (np.ndarray): Scan LiDAR Nx2 [distance, angle]
            valid (np.ndarray): Masque booléen Nx1 indiquant les points navigables

        Returns:
            np.ndarray: Nouveau masque booléen Nx1 avec la bulle de sécurité appliquée
        """
        obstacle_mask = ~valid

        kernel = np.ones(2 * self.bubble_radius + 1)
        inflated = np.convolve(obstacle_mask.astype(float), kernel, mode='same') > 0

        return valid & (~inflated)

    def find_best_gap(self, scan: np.ndarray, valid: np.ndarray, theta_goal: float) -> tuple[int,int]:
        """
        Identifie le meilleur gap dans un scan LiDAR Nx2 [distance, angle]
        en combinant plusieurs critères : orientation vers le goal, taille du gap et distance moyenne.

        Chaque gap est évalué selon un score pondéré permettant de favoriser
        les zones larges, dégagées et alignées avec la direction cible.

        Args:
            scan (np.ndarray): Scan LiDAR prétraité Nx2 [distance, angle]
            valid (np.ndarray): Masque booléen Nx1 (True = libre, False = obstacle)
            theta_goal (float): Angle cible (rad)

        Returns:
            tuple[int,int]: (start_index, stop_index) du meilleur gap
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

    def find_best_point(self, scan: np.ndarray, start: int, stop: int, theta_goal: float)->int:
        """
        Identifie le point optimal à l'intérieur d'un gap LiDAR donné en utilisant une moyenne mobile
        et un biais directionnel vers le goal.

        Cette fonction cherche le point correspondant à une grande distance libre
        tout en favorisant les directions proches de l'angle cible. Les distances sont lissées
        par convolution afin d'éviter les variations locales trop brusques.

        Args:
            scan (np.ndarray): Scan LiDAR prétraité Nx2 [distance, angle]
            start (int): Index du début du gap
            stop (int): Index de fin du gap
            theta_goal (float): Angle cible (rad)

        Returns:
            int: Index du point optimal à l'intérieur du gap
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

    def compute(self, scan: np.ndarray, theta_goal: float) -> tuple[int,float,np.ndarray]:
        """
        Exécute l'algorithme Follow-The-Gap complet avec fusion entre navigation locale et direction globale.

        Pipeline:
            preprocess → masque valide → safety bubble → détection du gap → sélection du point optimal → fusion avec goal

        Args:
            scan (np.ndarray): Scan LiDAR brut Nx2 [distance, angle]
            theta_goal (float): Angle cible global (rad)

        Returns:
            tuple:
                best_index (int): Index du point sélectionné dans le scan
                best_angle (float): Angle final de navigation (rad)
                scan (np.ndarray): Scan prétraité Nx2 [distance, angle]
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