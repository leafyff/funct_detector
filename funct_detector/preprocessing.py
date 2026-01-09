import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Tuple, List


def compute_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.concatenate([[0], np.cumsum(ds)])
    return s


def resample_by_arc_length(x: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x, y

    s = compute_arc_length(x, y)

    if s[-1] == 0:
        return x, y

    s_uniform = np.linspace(0, s[-1], n_samples)

    f_x = interp1d(s, x, kind='linear', fill_value='extrapolate')
    f_y = interp1d(s, y, kind='linear', fill_value='extrapolate')

    x_resampled = f_x(s_uniform)
    y_resampled = f_y(s_uniform)

    return x_resampled, y_resampled


def smooth_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 15:
        return x, y

    window_length = min(15, len(x) if len(x) % 2 == 1 else len(x) - 1)
    polyorder = min(3, window_length - 1)

    try:
        y_smooth = savgol_filter(y, window_length, polyorder)
        return x, y_smooth
    except (ValueError, TypeError):
        return x, y


def remove_outliers(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 5:
        return x, y

    curvatures = []
    radius = max(2, int(len(x) * 0.02))

    for i in range(len(x)):
        i_prev = max(0, i - radius)
        i_next = min(len(x) - 1, i + radius)

        if i_next - i_prev < 2:
            curvatures.append(0)
            continue

        x1, y1 = x[i_prev], y[i_prev]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i_next], y[i_next]

        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        if a * b * c < 1e-10:
            curvatures.append(0)
            continue

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        curvature = 4 * area / (a * b * c)
        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    median_curv = np.median(curvatures)
    std_curv = np.std(curvatures)

    threshold = median_curv + 3 * std_curv
    mask = curvatures < threshold

    return x[mask], y[mask]


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points

    def perpendicular_distance(point, line_start, line_end):
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)

        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

    dmax = 0
    index = 0

    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        rec_results1 = douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)

        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.array([points[0], points[-1]])


def preprocess_stroke(x: np.ndarray, y: np.ndarray, domain_width: float,
                      domain_height: float) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = max(100, int(domain_width * 50))
    x_resampled, y_resampled = resample_by_arc_length(x, y, n_samples)

    x_smooth, y_smooth = smooth_curve(x_resampled, y_resampled)

    x_clean, y_clean = remove_outliers(x_smooth, y_smooth)

    if len(x_clean) > 50:
        points = np.column_stack((x_clean, y_clean))
        epsilon = 0.01 * domain_height
        simplified = douglas_peucker(points, epsilon)
        x_clean, y_clean = simplified[:, 0], simplified[:, 1]

    return x_clean, y_clean


def is_function(x: np.ndarray, y: np.ndarray, domain_width: float) -> Tuple[bool, str]:
    epsilon = domain_width / 1000

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    for i in range(len(x_sorted) - 1):
        for j in range(i + 1, len(x_sorted)):
            if x_sorted[j] - x_sorted[i] > epsilon:
                break

            if abs(y_sorted[j] - y_sorted[i]) > epsilon:
                return False, f"Multiple y-values detected at x ≈ {x_sorted[i]:.2f}. Not a function y=f(x)."

    dx = np.diff(x_sorted)
    vertical_segments = np.sum(dx < epsilon)

    if vertical_segments > len(x_sorted) * 0.1:
        return False, "Too many vertical segments detected. Not a function y=f(x)."

    return True, ""


def detect_discontinuities(x: np.ndarray, y: np.ndarray) -> List[Tuple[int, int]]:
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    dy = np.abs(np.diff(y_sorted))
    median_dy = np.median(dy)

    threshold = 5 * median_dy

    discontinuity_indices = np.where(dy > threshold)[0]

    if len(discontinuity_indices) == 0:
        return [(0, len(x_sorted) - 1)]

    segments = []
    start = 0

    for idx in discontinuity_indices:
        if idx > start:
            segments.append((start, idx))
        start = idx + 1

    if start < len(x_sorted):
        segments.append((start, len(x_sorted) - 1))

    segment_points = []
    for seg_start, seg_end in segments:
        segment_points.append((sorted_indices[seg_start], sorted_indices[seg_end]))

    return segment_points if len(segment_points) > 1 else [(0, len(x) - 1)]
