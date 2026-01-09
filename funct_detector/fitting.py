import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, CubicSpline
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import warnings


@dataclass
class FittedModel:
    name: str
    evaluate: Callable[[np.ndarray], np.ndarray]
    latex_repr: str
    rmse: float
    aic: float
    complexity: float
    params: dict


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_aic(n: int, mse: float, k: int) -> float:
    if mse <= 0:
        return np.inf
    return n * np.log(mse) + 2 * k


def fit_polynomial(x: np.ndarray, y: np.ndarray, accuracy: float,
                   max_degree: int = 15) -> Optional[FittedModel]:
    best_degree = None
    best_aic = np.inf
    best_coeffs = None

    target_rmse = max(0.0001, accuracy)

    for degree in range(1, min(max_degree + 1, len(x) - 1)):
        try:
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            rmse = compute_rmse(y, y_pred)
            mse = rmse ** 2
            aic = compute_aic(len(x), mse, degree + 1)

            penalty = 0.5 if degree > 8 else 0
            aic_penalized = aic + penalty * degree

            if aic_penalized < best_aic:
                best_aic = aic_penalized
                best_degree = degree
                best_coeffs = coeffs

            if rmse < target_rmse and degree > 1:
                break

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_coeffs is None:
        return None

    y_pred = np.polyval(best_coeffs, x)
    rmse = compute_rmse(y, y_pred)

    def evaluate(x_eval):
        return np.polyval(best_coeffs, x_eval)

    return FittedModel(
        name=f"Polynomial (degree {best_degree})",
        evaluate=evaluate,
        latex_repr="polynomial",
        rmse=rmse,
        aic=best_aic,
        complexity=best_degree * 0.5,
        params={"coeffs": best_coeffs, "degree": best_degree}
    )


def fit_sinusoidal(x: np.ndarray, y: np.ndarray, accuracy: float) -> Optional[FittedModel]:
    try:
        y_fft = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(len(x), np.mean(np.diff(x)))

        positive_freqs = freqs[:len(freqs) // 2]
        positive_fft = np.abs(y_fft[:len(y_fft) // 2])

        if len(positive_fft) < 2:
            return None

        dominant_idx = np.argmax(positive_fft[1:]) + 1
        f0 = np.abs(positive_freqs[dominant_idx])

        power_ratio = positive_fft[dominant_idx] / np.sum(positive_fft)

        if power_ratio < 0.4 or f0 == 0:
            return None

        a_init = 2 * positive_fft[dominant_idx] / len(x)
        b_init = np.mean(y)
        phase_init = 0

        def sin_func(x_vals, amplitude, freq, phase, offset):
            return amplitude * np.sin(2 * np.pi * freq * x_vals + phase) + offset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = curve_fit(sin_func, x, y,
                               p0=[a_init, f0, phase_init, b_init],
                               maxfev=5000)
            popt: np.ndarray = np.asarray(result[0])

        y_pred = sin_func(x, popt[0], popt[1], popt[2], popt[3])
        rmse = compute_rmse(y, y_pred)

        target_rmse = max(0.0001, accuracy)
        if rmse / np.std(y) > 0.5 or rmse > target_rmse * 10:
            return None

        aic = compute_aic(len(x), rmse ** 2, 4)

        def evaluate(x_eval):
            return sin_func(x_eval, popt[0], popt[1], popt[2], popt[3])

        return FittedModel(
            name="Sinusoidal",
            evaluate=evaluate,
            latex_repr="sinusoidal",
            rmse=rmse,
            aic=aic,
            complexity=3.0,
            params={"A": float(popt[0]), "f": float(popt[1]),
                    "phase": float(popt[2]), "B": float(popt[3])}
        )
    except (RuntimeError, ValueError, TypeError):
        return None


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Optional[FittedModel]:
    try:
        if np.all(y > 0):
            y_log = np.log(y)
        elif np.all(y < 0):
            y_log = np.log(-y)
        else:
            return None

        coeffs = np.polyfit(x, y_log, 1)
        r_squared = 1 - (np.var(y_log - np.polyval(coeffs, x)) / np.var(y_log))

        if r_squared < 0.85:
            return None

        def exp_func(x_vals, amplitude, rate, offset):
            return amplitude * np.exp(rate * x_vals) + offset

        a_init = np.exp(coeffs[1]) if np.all(y > 0) else -np.exp(coeffs[1])
        b_init = coeffs[0]
        c_init = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = curve_fit(exp_func, x, y,
                               p0=[a_init, b_init, c_init],
                               maxfev=5000)
            popt: np.ndarray = np.asarray(result[0])

        y_pred = exp_func(x, popt[0], popt[1], popt[2])
        rmse = compute_rmse(y, y_pred)
        aic = compute_aic(len(x), rmse ** 2, 3)

        def evaluate(x_eval):
            return exp_func(x_eval, popt[0], popt[1], popt[2])

        return FittedModel(
            name="Exponential",
            evaluate=evaluate,
            latex_repr="exponential",
            rmse=rmse,
            aic=aic,
            complexity=3.0,
            params={"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2])}
        )
    except (RuntimeError, ValueError, TypeError):
        return None


def fit_logarithmic(x: np.ndarray, y: np.ndarray) -> Optional[FittedModel]:
    try:
        if not np.all(x > 0):
            return None

        x_log = np.log(x)
        coeffs = np.polyfit(x_log, y, 1)
        r_squared = 1 - (np.var(y - np.polyval(coeffs, x_log)) / np.var(y))

        if r_squared < 0.85:
            return None

        def log_func(x_vals, amplitude, offset):
            return amplitude * np.log(x_vals) + offset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = curve_fit(log_func, x, y, p0=[coeffs[0], coeffs[1]])
            popt: np.ndarray = np.asarray(result[0])

        y_pred = log_func(x, popt[0], popt[1])
        rmse = compute_rmse(y, y_pred)
        aic = compute_aic(len(x), rmse ** 2, 2)

        def evaluate(x_eval):
            return log_func(x_eval, popt[0], popt[1])

        return FittedModel(
            name="Logarithmic",
            evaluate=evaluate,
            latex_repr="logarithmic",
            rmse=rmse,
            aic=aic,
            complexity=3.0,
            params={"A": float(popt[0]), "B": float(popt[1])}
        )
    except (RuntimeError, ValueError, TypeError):
        return None


def fit_spline(x: np.ndarray, y: np.ndarray, accuracy: float) -> FittedModel:
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    noise_estimate = np.std(np.diff(y_sorted)) / np.sqrt(2)
    s_base = len(x) * noise_estimate ** 2

    s = s_base * (accuracy / 0.01)

    try:
        spline = UnivariateSpline(x_sorted, y_sorted, s=s, k=3)
        y_pred = spline(x_sorted)
        rmse = compute_rmse(y_sorted, y_pred)

        knots = spline.get_knots()
        n_knots = len(knots)
        aic = compute_aic(len(x), rmse ** 2, n_knots + 4)

        def evaluate(x_eval):
            return spline(x_eval)

        return FittedModel(
            name=f"Cubic Spline ({n_knots} knots)",
            evaluate=evaluate,
            latex_repr="spline",
            rmse=rmse,
            aic=aic,
            complexity=n_knots * 0.3,
            params={"spline": spline, "knots": knots}
        )
    except (ValueError, TypeError):
        spline = CubicSpline(x_sorted, y_sorted)
        y_pred = spline(x_sorted)
        rmse = compute_rmse(y_sorted, y_pred)

        n_segments = len(x_sorted) - 1
        aic = compute_aic(len(x), rmse ** 2, n_segments * 4)

        def evaluate(x_eval):
            return spline(x_eval)

        return FittedModel(
            name=f"Cubic Spline ({n_segments} segments)",
            evaluate=evaluate,
            latex_repr="spline",
            rmse=rmse,
            aic=aic,
            complexity=n_segments * 0.3,
            params={"spline": spline, "n_segments": n_segments}
        )


def fit_piecewise(x: np.ndarray, y: np.ndarray,
                  segments: List[Tuple[int, int]], accuracy: float) -> Optional[FittedModel]:
    if len(segments) <= 1:
        return None

    segment_models = []
    total_rmse = 0
    total_complexity = 0

    for start_idx, end_idx in segments:
        x_seg = x[start_idx:end_idx + 1]
        y_seg = y[start_idx:end_idx + 1]

        if len(x_seg) < 4:
            return None

        seg_models = fit_models(x_seg, y_seg, [(0, len(x_seg) - 1)], accuracy)

        if not seg_models:
            return None

        best_seg = select_best_model(seg_models, x_seg, y_seg)
        segment_models.append((x_seg[0], x_seg[-1], best_seg))

        total_rmse += best_seg.rmse * len(x_seg)
        total_complexity += best_seg.complexity

    total_rmse /= len(x)

    def evaluate(x_eval):
        y_new = np.zeros_like(x_eval)
        for x_min, x_max, model in segment_models:
            mask = (x_eval >= x_min) & (x_eval <= x_max)
            y_new[mask] = model.evaluate(x_eval[mask])
        return y_new

    aic = compute_aic(len(x), total_rmse ** 2, int(total_complexity))

    return FittedModel(
        name=f"Piecewise ({len(segments)} segments)",
        evaluate=evaluate,
        latex_repr="piecewise",
        rmse=total_rmse,
        aic=aic,
        complexity=total_complexity + 2,
        params={"segments": segment_models}
    )


def fit_models(x: np.ndarray, y: np.ndarray,
               segments: List[Tuple[int, int]], accuracy: float = 0.01) -> List[FittedModel]:
    models = []

    accuracy = max(0.0001, accuracy)

    if len(segments) > 1:
        piecewise_model = fit_piecewise(x, y, segments, accuracy)
        if piecewise_model:
            models.append(piecewise_model)

    poly_model = fit_polynomial(x, y, accuracy)
    if poly_model:
        models.append(poly_model)

    sin_model = fit_sinusoidal(x, y, accuracy)
    if sin_model:
        models.append(sin_model)

    exp_model = fit_exponential(x, y)
    if exp_model:
        models.append(exp_model)

    log_model = fit_logarithmic(x, y)
    if log_model:
        models.append(log_model)

    spline_model = fit_spline(x, y, accuracy)
    models.append(spline_model)

    return models


def select_best_model(models: List[FittedModel], x: np.ndarray, y: np.ndarray) -> FittedModel:
    y_std = np.std(y)

    scores = []
    for model in models:
        normalized_rmse = model.rmse / y_std if y_std > 0 else model.rmse
        score = normalized_rmse + 0.1 * model.aic + model.complexity
        scores.append(score)

    best_idx = np.argmin(scores)

    if len(models) > 1:
        sorted_scores = np.argsort(scores)
        second_best_idx = sorted_scores[1]
        if abs(scores[best_idx] - scores[second_best_idx]) < 0.05 * scores[best_idx]:
            if models[best_idx].complexity > models[second_best_idx].complexity:
                return models[second_best_idx]

    return models[best_idx]
