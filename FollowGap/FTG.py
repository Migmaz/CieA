import numpy as np

class FollowGap:
    """
    Algorithme Follow-The-Gap pour navigation réactive basée sur LiDAR.

    Prend en entrée deux représentations d'un même scan LiDAR (Nx2, [distance, angle]) :
      - scan_true : distances réelles, utilisées pour la détection d'obstacles et le masque de validité.
      - scan_eff  : distances pondérées/modifiées, utilisées pour le scoring des gaps et la sélection du point cible.

    Pipeline :
        build_valid_mask → safety bubble → meilleur gap → meilleur point → fusion goal

    Paramètres
    ----------
    max_range : float
        Distance maximale retenue (m). Au-delà, les points sont ramenés à cette valeur.
    min_range : float
        Distance minimale valide (m). En dessous, les points sont considérés comme trop proches.
    bubble_radius_m : float
        Rayon de la bulle de sécurité autour des obstacles (m).
    threshold : float
        Seuil de distance (m) pour qu'un point soit considéré comme libre.
    conv_size : int
        Taille du kernel de lissage dans `find_best_point`.
    weight_goal : float
        Poids de l'alignement avec le goal dans le score de gap.
    weight_dist : float
        Poids de la distance moyenne dans le score de gap.
    weight_len : float
        Poids de la longueur du gap dans le score de gap.
    alpha_point : float
        Influence de l'angle dans la sélection du point optimal.
    alpha_final : float
        Taux de fusion entre l'angle du gap et l'angle du goal (0 = goal pur, 1 = gap pur).
    """

    def __init__(
        self,
        max_range: float = 10.0,
        min_range: float = 0.05,
        bubble_radius_m: float = 0.35,
        threshold: float = 1.2,
        conv_size: int = 20,
        weight_goal: float = 0.6,
        weight_dist: float = 0.1,  
        weight_len: float = 0.3,
        alpha_point: float = 1.0,
        alpha_final: float = 0.7,
    ):
        if weight_goal < 0 or weight_dist < 0 or weight_len < 0:
            raise ValueError("Les poids doivent être positifs ou nuls.")
        if not (0.0 <= alpha_final <= 1.0):
            raise ValueError("alpha_final doit être dans [0, 1].")
        if not (0.0 <= alpha_point):
            raise ValueError("alpha_point doit être positif ou nul.")

        self.max_range = max_range
        self.min_range = min_range
        self.bubble_radius_m = bubble_radius_m
        self.threshold = threshold
        self.conv_size = conv_size

        self.weight_goal = weight_goal
        self.weight_dist = weight_dist
        self.weight_len = weight_len

        self.alpha_point = alpha_point
        self.alpha_final = alpha_final

    def _validate_scan(self, scan: np.ndarray, name: str = "scan") -> None:
        if scan.ndim != 2 or scan.shape[1] != 2:
            raise ValueError(f"{name} doit être de forme (N, 2), reçu {scan.shape}.")
        if scan.shape[0] < 2:
            raise ValueError(f"{name} doit contenir au moins 2 points.")

    def _angle_resolution(self, angles: np.ndarray) -> float:
        diffs = np.abs(np.diff(angles))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            raise ValueError("Impossible d'estimer la résolution angulaire : angles constants.")
        return float(np.median(diffs))

    def build_valid_mask(self, scan_true: np.ndarray) -> np.ndarray:
        """
        Construit le masque des points libres à partir des distances réelles.
        """
        return scan_true[:, 0] > self.threshold  #modif

    def safety_bubble(self, scan_true: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Dilate les zones d'obstacle d'une marge de sécurité physique.
        """
        angles = scan_true[:, 1]
        angle_res = self._angle_resolution(angles)

        obstacle_dists = scan_true[~valid, 0]
        ref_dist = float(np.min(obstacle_dists)) if len(obstacle_dists) > 0 else self.max_range
        ref_dist = max(ref_dist, 1e-6)

        bubble_radius_pts = max(int(np.ceil(self.bubble_radius_m / (ref_dist * angle_res))), 1)

        kernel = np.ones(2 * bubble_radius_pts + 1)
        inflated = np.convolve((~valid).astype(float), kernel, mode="same") > 0

        return valid & (~inflated)

    def find_best_gap(self, scan_eff: np.ndarray, valid: np.ndarray, theta_goal: float) -> tuple[int, int]:
        """
        Identifie le meilleur gap parmi les zones libres du scan.
        """
        angles = scan_eff[:, 1]
        distances = scan_eff[:, 0]

        # Limitons le scan aux angles avant [-pi/2, pi/2]
        forward_mask = (angles >= -np.pi / 2) & (angles <= np.pi / 2)
        valid = valid & forward_mask

        diff = np.diff(valid.astype(int))
        starts = np.where(diff == 1)[0] + 1
        stops = np.where(diff == -1)[0] + 1

        if valid[0]:
            starts = np.insert(starts, 0, 0)
        if valid[-1]:
            stops = np.append(stops, len(valid))

        if len(starts) == 0 or len(stops) == 0 or len(starts) != len(stops):
            return None, None

        lengths = stops - starts
        lengths_norm = lengths / (np.max(lengths) + 1e-9)

        cumsum = np.cumsum(distances)
        sums = np.array([
            cumsum[stops[i]-1] - (cumsum[starts[i]-1] if starts[i]>0 else 0.0)
            for i in range(len(starts))
        ])
        means = sums / (lengths + 1e-9)
        means_norm = means / (np.max(means) + 1e-9)

        centers = np.clip((starts + stops) // 2, 0, len(angles)-1)
        delta = np.arctan2(np.sin(angles[centers]-theta_goal), np.cos(angles[centers]-theta_goal))
        goal_score = np.exp(-np.abs(delta))

        total_weight = self.weight_goal + self.weight_len + self.weight_dist + 1e-9
        scores = (self.weight_goal*goal_score + self.weight_len*lengths_norm + self.weight_dist*means_norm)/total_weight

        best = int(np.argmax(scores))
        return int(starts[best]), int(stops[best])

    def find_best_point(self, scan_eff: np.ndarray, start: int, stop: int, theta_goal: float) -> int:
        """
        Sélectionne le point optimal à l'intérieur d'un gap.
        """
        if stop <= start:
            return start

        distances = scan_eff[start:stop, 0]
        angles = scan_eff[start:stop, 1]
        n = len(distances)

        if n == 1:
            return start

        k = min(self.conv_size, n)
        if k < 3:
            return (start + stop) // 2

        kernel = np.ones(k) / k
        smooth = np.convolve(distances, kernel, mode="same")

        delta = np.arctan2(np.sin(angles-theta_goal), np.cos(angles-theta_goal))
        score = smooth - self.alpha_point*np.abs(delta)

        return int(np.argmax(score)) + start

    def compute(self, scan_true: np.ndarray, scan_eff: np.ndarray, theta_goal: float) -> tuple[int | None, float | None, np.ndarray]:
        """
        Exécute le pipeline Follow-The-Gap complet.
        """
        self._validate_scan(scan_true, "scan_true")
        self._validate_scan(scan_eff, "scan_eff")

        if scan_true.shape != scan_eff.shape:
            raise ValueError(f"scan_true et scan_eff doivent avoir la même shape, reçu {scan_true.shape} vs {scan_eff.shape}.")

        # Pas de preprocess ici (#modif)
        valid = self.build_valid_mask(scan_true)
        valid = self.safety_bubble(scan_true, valid)

        start, stop = self.find_best_gap(scan_eff, valid, theta_goal)
        if start is None:
            return None, None, scan_eff

        best_i = self.find_best_point(scan_eff, start, stop, theta_goal)
        theta_gap = float(scan_eff[best_i,1])
        theta_final = float(np.clip(self.alpha_final*theta_gap + (1.0-self.alpha_final)*theta_goal, -np.pi/2, np.pi/2))

        return best_i, theta_final, scan_eff