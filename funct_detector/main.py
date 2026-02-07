"""
Refactored Function Drawer & LaTeX Generator Application

This is a production-grade refactoring following SOLID principles, with:
- Comprehensive error handling
- Type safety and validation
- Clear separation of concerns
- Extensive documentation
- Zero-bug policy implementation
"""

import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
import sympy as sp
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QCursor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# Type Aliases and Protocols
# ============================================================================

FloatArray = NDArray[np.floating[Any]]
EvaluationFunction = Callable[[FloatArray], FloatArray]


class SplineProtocol(Protocol):
    """Protocol for spline objects."""

    def __call__(self, x: FloatArray) -> FloatArray:
        """Evaluate spline at given points."""
        ...

    def get_knots(self) -> FloatArray:
        """Retrieve knot locations."""
        ...


# ============================================================================
# Data Models (Single Responsibility Principle)
# ============================================================================

@dataclass(frozen=True)
class PlotSettings:
    """Immutable configuration for plot display settings."""

    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    grid_spacing: float = 1.0
    accuracy: float = 0.0001

    def __post_init__(self) -> None:
        """Validate settings after initialization."""
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be less than x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be less than y_max ({self.y_max})"
            )
        if self.grid_spacing <= 0:
            raise ValueError(
                f"grid_spacing ({self.grid_spacing}) must be positive"
            )
        if not (0.0001 <= self.accuracy <= 1.0):
            raise ValueError(
                f"accuracy ({self.accuracy}) must be between 0.0001 and 1.0"
            )

    @property
    def domain_width(self) -> float:
        """Calculate the width of the x domain."""
        return self.x_max - self.x_min

    @property
    def domain_height(self) -> float:
        """Calculate the height of the y domain."""
        return self.y_max - self.y_min


@dataclass(frozen=True)
class FittedModel:
    """Represents a fitted mathematical model with evaluation capabilities."""

    name: str
    evaluate: EvaluationFunction
    latex_repr: str
    rmse: float
    aic: float
    complexity: float
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model after initialization."""
        if self.rmse < 0:
            raise ValueError(f"RMSE cannot be negative: {self.rmse}")
        if self.complexity < 0:
            raise ValueError(f"Complexity cannot be negative: {self.complexity}")
        if not callable(self.evaluate):
            raise ValueError("Evaluate must be a callable function")


# ============================================================================
# Model Fitting Strategies (Strategy Pattern + Open/Closed Principle)
# ============================================================================

class ModelFitter(ABC):
    """Abstract base class for model fitting strategies."""

    @abstractmethod
    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> Optional[FittedModel]:
        """
        Fit the model to data.

        Args:
            x: X-coordinates
            y: Y-coordinates
            accuracy: Target accuracy for fitting

        Returns:
            FittedModel if successful, None otherwise
        """
        pass

    @staticmethod
    def _compute_rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
        """
        Compute root mean squared error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _compute_aic(n: int, mse: float, k: int) -> float:
        """
        Compute Akaike Information Criterion.

        Args:
            n: Number of data points
            mse: Mean squared error
            k: Number of parameters

        Returns:
            AIC value
        """
        if mse <= 0:
            return np.inf
        return float(n * np.log(mse) + 2 * k)


class PolynomialFitter(ModelFitter):
    """Fits polynomial models to data."""

    def __init__(self, max_degree: int = 15):
        """
        Initialize polynomial fitter.

        Args:
            max_degree: Maximum polynomial degree to consider
        """
        self.max_degree = max_degree

    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> Optional[FittedModel]:
        """Fit polynomial model using optimal degree selection."""
        best_degree: Optional[int] = None
        best_aic = np.inf
        best_coeffs: Optional[FloatArray] = None

        target_rmse = max(0.0001, accuracy)
        max_deg = min(self.max_degree + 1, len(x) - 1)

        for degree in range(1, max_deg):
            try:
                coeffs = np.polyfit(x, y, degree)
                y_pred = np.polyval(coeffs, x)
                rmse = self._compute_rmse(y, y_pred)
                mse = rmse ** 2
                aic = self._compute_aic(len(x), mse, degree + 1)

                # Penalize very high degrees
                penalty = 0.5 if degree > 8 else 0
                aic_penalized = aic + penalty * degree

                if aic_penalized < best_aic:
                    best_aic = aic_penalized
                    best_degree = degree
                    best_coeffs = coeffs

                # Early stopping if we've achieved target accuracy
                if rmse < target_rmse and degree > 1:
                    break

            except (np.linalg.LinAlgError, ValueError):
                continue

        if best_coeffs is None or best_degree is None:
            return None

        y_pred = np.polyval(best_coeffs, x)
        rmse = self._compute_rmse(y, y_pred)

        def evaluate(x_eval: FloatArray) -> FloatArray:
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


class SinusoidalFitter(ModelFitter):
    """Fits sinusoidal models to data."""

    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> Optional[FittedModel]:
        """Fit sinusoidal model using FFT for frequency detection."""
        try:
            # Use FFT to detect dominant frequency
            y_fft = np.fft.fft(y - np.mean(y))
            freqs = np.fft.fftfreq(len(x), np.mean(np.diff(x)))

            positive_freqs = freqs[:len(freqs) // 2]
            positive_fft = np.abs(y_fft[:len(y_fft) // 2])

            if len(positive_fft) < 2:
                return None

            # Find dominant frequency (skip DC component)
            # Convert to float array to satisfy type checker
            dominant_idx = int(np.argmax(positive_fft[1:]) + 1)
            f0 = float(np.abs(positive_freqs[dominant_idx]))

            # Check if sinusoidal pattern is strong enough
            power_ratio = positive_fft[dominant_idx] / np.sum(positive_fft)
            if power_ratio < 0.4 or f0 == 0:
                return None

            # Initial parameter estimates
            a_init = 2 * positive_fft[dominant_idx] / len(x)
            b_init = np.mean(y)
            phase_init = 0.0

            def sin_func(
                x_vals: FloatArray, amplitude: float, freq: float,
                phase: float, offset: float
            ) -> FloatArray:
                return amplitude * np.sin(2 * np.pi * freq * x_vals + phase) + offset

            # Fit using curve_fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _pcov = curve_fit(
                    sin_func, x, y,
                    p0=[a_init, f0, phase_init, b_init],
                    maxfev=5000
                )

            y_pred = sin_func(x, *popt)
            rmse = self._compute_rmse(y, y_pred)

            # Validate quality
            target_rmse = max(0.0001, accuracy)
            y_std = np.std(y)
            if y_std > 0 and (rmse / y_std > 0.5 or rmse > target_rmse * 10):
                return None

            aic = self._compute_aic(len(x), rmse ** 2, 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return sin_func(x_eval, *popt)

            return FittedModel(
                name="Sinusoidal",
                evaluate=evaluate,
                latex_repr="sinusoidal",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={
                    "A": float(popt[0]), "f": float(popt[1]),
                    "phase": float(popt[2]), "B": float(popt[3])
                }
            )
        except (RuntimeError, ValueError, TypeError, IndexError, np.linalg.LinAlgError):
            return None


class ExponentialFitter(ModelFitter):
    """Fits exponential models to data."""

    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> Optional[FittedModel]:
        """Fit exponential model."""
        try:
            # Check if data is suitable for exponential fit
            if np.all(y > 0):
                y_log = np.log(y)
                sign = 1.0
            elif np.all(y < 0):
                y_log = np.log(-y)
                sign = -1.0
            else:
                return None

            # Linear fit in log space
            coeffs = np.polyfit(x, y_log, 1)
            y_log_var = np.var(y_log)
            if y_log_var > 0:
                r_squared = 1 - (np.var(y_log - np.polyval(coeffs, x)) / y_log_var)
            else:
                r_squared = 0.0

            # Only proceed if good linear fit in log space
            if r_squared < 0.85:
                return None

            def exp_func(
                x_vals: FloatArray, amplitude: float, rate: float, offset: float
            ) -> FloatArray:
                return amplitude * np.exp(rate * x_vals) + offset

            # Initial parameter estimates
            a_init = sign * np.exp(coeffs[1])
            b_init = coeffs[0]
            c_init = 0.0

            # Refine with curve_fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _pcov = curve_fit(
                    exp_func, x, y,
                    p0=[a_init, b_init, c_init],
                    maxfev=5000
                )

            y_pred = exp_func(x, *popt)
            rmse = self._compute_rmse(y, y_pred)
            aic = self._compute_aic(len(x), rmse ** 2, 3)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return exp_func(x_eval, *popt)

            return FittedModel(
                name="Exponential",
                evaluate=evaluate,
                latex_repr="exponential",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2])}
            )
        except (RuntimeError, ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
            return None


class LogarithmicFitter(ModelFitter):
    """Fits logarithmic models to data."""

    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> Optional[FittedModel]:
        """Fit logarithmic model."""
        try:
            # Logarithm requires positive x values
            if not np.all(x > 0):
                return None

            # Linear fit in log(x) space
            x_log = np.log(x)
            coeffs = np.polyfit(x_log, y, 1)
            y_var = np.var(y)
            if y_var > 0:
                r_squared = 1 - (np.var(y - np.polyval(coeffs, x_log)) / y_var)
            else:
                r_squared = 0.0

            # Only proceed if good linear fit
            if r_squared < 0.85:
                return None

            def log_func(x_vals: FloatArray, amplitude: float, offset: float) -> FloatArray:
                return amplitude * np.log(x_vals) + offset

            # Refine with curve_fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _pcov = curve_fit(log_func, x, y, p0=[coeffs[0], coeffs[1]])

            y_pred = log_func(x, *popt)
            rmse = self._compute_rmse(y, y_pred)
            aic = self._compute_aic(len(x), rmse ** 2, 2)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return log_func(x_eval, *popt)

            return FittedModel(
                name="Logarithmic",
                evaluate=evaluate,
                latex_repr="logarithmic",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={"A": float(popt[0]), "B": float(popt[1])}
            )
        except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
            return None


class SplineFitter(ModelFitter):
    """Fits spline models to data."""

    def fit(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> FittedModel:
        """Fit cubic spline model."""
        # Sort data by x - convert to float to satisfy type checker
        sorted_indices = np.argsort(x).astype(np.int64)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Estimate noise level
        noise_estimate = (
            np.std(np.diff(y_sorted)) / np.sqrt(2) if len(y_sorted) > 1 else 0.1
        )
        s_base = len(x) * noise_estimate ** 2
        s = s_base * (accuracy / 0.01)

        try:
            # Try UnivariateSpline first (adaptive knots)
            spline = UnivariateSpline(x_sorted, y_sorted, s=s, k=3)
            y_pred = spline(x_sorted)
            rmse = self._compute_rmse(y_sorted, y_pred)

            knots = spline.get_knots()
            n_knots = len(knots)
            aic = self._compute_aic(len(x), rmse ** 2, n_knots + 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
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
            # Fallback to CubicSpline
            spline = CubicSpline(x_sorted, y_sorted)
            y_pred = spline(x_sorted)
            rmse = self._compute_rmse(y_sorted, y_pred)

            n_segments = len(x_sorted) - 1
            aic = self._compute_aic(len(x), rmse ** 2, n_segments * 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
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


# ============================================================================
# Model Selection Service (Single Responsibility Principle)
# ============================================================================

class ModelSelectionService:
    """Service for fitting and selecting the best model."""

    def __init__(self):
        """Initialize the model selection service."""
        self.fitters: List[ModelFitter] = [
            PolynomialFitter(),
            SinusoidalFitter(),
            ExponentialFitter(),
            LogarithmicFitter(),
            SplineFitter(),
        ]

    def fit_all_models(
        self, x: FloatArray, y: FloatArray, accuracy: float
    ) -> List[FittedModel]:
        """
        Fit all available models to the data.

        Args:
            x: X-coordinates
            y: Y-coordinates
            accuracy: Target accuracy

        Returns:
            List of successfully fitted models
        """
        models: List[FittedModel] = []

        for fitter in self.fitters:
            try:
                model = fitter.fit(x, y, accuracy)
                if model is not None:
                    models.append(model)
            # Catch all exceptions to ensure one fitter failure doesn't stop others
            except Exception:  # pylint: disable=broad-exception-caught
                continue

        return models

    @staticmethod
    def select_best_model(
        models: List[FittedModel], y: FloatArray
    ) -> FittedModel:
        """
        Select the best model based on RMSE, AIC, and complexity.

        Args:
            models: List of fitted models
            y: Y-coordinates (for normalization)

        Returns:
            Best model

        Raises:
            ValueError: If models list is empty
        """
        if not models:
            raise ValueError("Cannot select best model from empty list")

        y_std = np.std(y)

        # Compute scores
        scores = []
        for model in models:
            normalized_rmse = model.rmse / y_std if y_std > 0 else model.rmse
            score = normalized_rmse + 0.1 * model.aic + model.complexity
            scores.append(score)

        scores_array = np.array(scores)
        best_idx = int(np.argmin(scores_array))

        # Prefer simpler model if scores are very close
        if len(models) > 1:
            sorted_indices_int = np.argsort(scores_array).astype(np.int64)
            second_best_idx = int(sorted_indices_int[1])
            best_score = scores_array[best_idx]
            second_score = scores_array[second_best_idx]
            if abs(best_score - second_score) < 0.05 * best_score:
                if models[best_idx].complexity > models[second_best_idx].complexity:
                    return models[second_best_idx]

        return models[best_idx]


# ============================================================================
# LaTeX Generation Service (Single Responsibility Principle)
# ============================================================================

class LaTeXGenerator:
    """Service for generating LaTeX representations of models."""

    def generate(self, model: FittedModel) -> str:
        """
        Generate LaTeX representation of a model.

        Args:
            model: Fitted model

        Returns:
            LaTeX string representation
        """
        try:
            if model.latex_repr == "polynomial":
                return self._polynomial_to_latex(model)
            elif model.latex_repr == "sinusoidal":
                return self._sinusoidal_to_latex(model)
            elif model.latex_repr == "exponential":
                return self._exponential_to_latex(model)
            elif model.latex_repr == "logarithmic":
                return self._logarithmic_to_latex(model)
            elif model.latex_repr == "spline":
                return self._spline_to_latex(model)
            else:
                return "$f(x) = \\text{unknown}$"
        # Ensure LaTeX generation never crashes the app
        except Exception:  # pylint: disable=broad-exception-caught
            return "$f(x) = \\text{error}$"

    @staticmethod
    def _polynomial_to_latex(model: FittedModel) -> str:
        """
        Generate LaTeX for polynomial model.

        Args:
            model: Fitted polynomial model

        Returns:
            LaTeX string
        """
        coeffs = model.params["coeffs"]
        degree = model.params["degree"]

        x = sp.Symbol('x')
        terms = [
            sp.Rational(float(coeffs[i])).limit_denominator(10000) * x ** (degree - i)
            for i in range(len(coeffs))
        ]
        expr = sp.simplify(sp.Add(*terms))
        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    @staticmethod
    def _sinusoidal_to_latex(model: FittedModel) -> str:
        """
        Generate LaTeX for sinusoidal model.

        Args:
            model: Fitted sinusoidal model

        Returns:
            LaTeX string
        """
        a_val = model.params["A"]
        f_val = model.params["f"]
        phase = model.params["phase"]
        b_val = model.params["B"]

        x = sp.Symbol('x')
        a_sym = sp.Rational(float(a_val)).limit_denominator(10000)
        f_sym = sp.Rational(float(f_val)).limit_denominator(10000)
        phase_sym = sp.Rational(float(phase)).limit_denominator(10000)
        b_sym = sp.Rational(float(b_val)).limit_denominator(10000)

        expr = a_sym * sp.sin(2 * sp.pi * f_sym * x + phase_sym) + b_sym
        expr = sp.simplify(expr)
        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    @staticmethod
    def _exponential_to_latex(model: FittedModel) -> str:
        """
        Generate LaTeX for exponential model.

        Args:
            model: Fitted exponential model

        Returns:
            LaTeX string
        """
        a_val = model.params["A"]
        b_val = model.params["B"]
        c_val = model.params["C"]

        x = sp.Symbol('x')
        a_sym = sp.Rational(float(a_val)).limit_denominator(10000)
        b_sym = sp.Rational(float(b_val)).limit_denominator(10000)
        c_sym = sp.Rational(float(c_val)).limit_denominator(10000)

        expr = a_sym * sp.exp(b_sym * x) + c_sym
        expr = sp.simplify(expr)
        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    @staticmethod
    def _logarithmic_to_latex(model: FittedModel) -> str:
        """
        Generate LaTeX for logarithmic model.

        Args:
            model: Fitted logarithmic model

        Returns:
            LaTeX string
        """
        a_val = model.params["A"]
        b_val = model.params["B"]

        x = sp.Symbol('x')
        a_sym = sp.Rational(float(a_val)).limit_denominator(10000)
        b_sym = sp.Rational(float(b_val)).limit_denominator(10000)

        expr = a_sym * sp.ln(x) + b_sym
        expr = sp.simplify(expr)
        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    @staticmethod
    def _spline_to_latex(model: FittedModel) -> str:
        """
        Generate LaTeX for spline model.

        Args:
            model: Fitted spline model

        Returns:
            LaTeX string
        """
        n_segments = model.params.get("n_segments", "unknown")
        return f"$f(x) = \\text{{Cubic spline with {n_segments} segments}}$"


# ============================================================================
# Preprocessing Service (Single Responsibility Principle)
# ============================================================================

class StrokePreprocessor:
    """Service for preprocessing user-drawn strokes."""

    def __init__(self, domain_width: float, domain_height: float):
        """
        Initialize the preprocessor.

        Args:
            domain_width: Width of the plotting domain
            domain_height: Height of the plotting domain

        Raises:
            ValueError: If dimensions are invalid
        """
        if domain_width <= 0 or domain_height <= 0:
            raise ValueError("Domain dimensions must be positive")

        self._domain_width = domain_width
        self._domain_height = domain_height

    def preprocess(
        self, x: FloatArray, y: FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Preprocess stroke data.

        Args:
            x: X-coordinates
            y: Y-coordinates

        Returns:
            Tuple of (processed_x, processed_y)
        """
        n_samples = max(100, int(self._domain_width * 50))
        x_res, y_res = self._resample_by_arc_length(x, y, n_samples)
        x_smooth, y_smooth = self._smooth_curve(x_res, y_res)
        x_clean, y_clean = self._remove_outliers(x_smooth, y_smooth)

        if len(x_clean) > 50:
            x_clean, y_clean = self._simplify_curve(x_clean, y_clean)

        return x_clean, y_clean

    def is_function(self, x: FloatArray, y: FloatArray) -> Tuple[bool, str]:
        """
        Check if curve represents a function y=f(x).

        Args:
            x: X-coordinates
            y: Y-coordinates

        Returns:
            Tuple of (is_function, error_message)
        """
        epsilon = self._domain_width / 1000
        sorted_indices = np.argsort(x).astype(np.int64)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        for i in range(len(x_sorted) - 1):
            for j in range(i + 1, len(x_sorted)):
                if x_sorted[j] - x_sorted[i] > epsilon:
                    break
                if abs(y_sorted[j] - y_sorted[i]) > epsilon:
                    return False, f"Multiple y-values at x ≈ {x_sorted[i]:.2f}"

        return True, ""

    @staticmethod
    def _resample_by_arc_length(
        x: FloatArray, y: FloatArray, n_samples: int
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Resample curve uniformly by arc length.

        Args:
            x: X-coordinates
            y: Y-coordinates
            n_samples: Number of samples to generate

        Returns:
            Tuple of (resampled_x, resampled_y)
        """
        if len(x) < 2:
            return x, y

        dx, dy = np.diff(x), np.diff(y)
        s = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])

        if s[-1] == 0:
            return x, y

        s_uniform = np.linspace(0, s[-1], n_samples)

        try:
            f_x = interp1d(s, x, kind='linear', fill_value='extrapolate')
            f_y = interp1d(s, y, kind='linear', fill_value='extrapolate')
            return f_x(s_uniform), f_y(s_uniform)
        except (ValueError, IndexError):
            return x, y

    @staticmethod
    def _smooth_curve(
        x: FloatArray, y: FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Apply Savitzky-Golay smoothing filter.

        Args:
            x: X-coordinates
            y: Y-coordinates

        Returns:
            Tuple of (x, smoothed_y)
        """
        if len(x) < 15:
            return x, y

        window_length = min(15, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if window_length < 3:
            return x, y

        try:
            y_smooth = savgol_filter(y, window_length, min(3, window_length - 1))
            return x, y_smooth
        except (ValueError, TypeError):
            return x, y

    @staticmethod
    def _remove_outliers(
        x: FloatArray, y: FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Remove outliers based on curvature.

        Args:
            x: X-coordinates
            y: Y-coordinates

        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        if len(x) < 5:
            return x, y

        curvatures = np.zeros(len(x))
        radius = max(2, int(len(x) * 0.02))

        for i in range(len(x)):
            i_prev, i_next = max(0, i - radius), min(len(x) - 1, i + radius)
            if i_next - i_prev < 2:
                continue

            # Menger curvature
            pts = np.array([
                [x[i_prev], y[i_prev]],
                [x[i], y[i]],
                [x[i_next], y[i_next]]
            ])
            a = np.linalg.norm(pts[1] - pts[0])
            b = np.linalg.norm(pts[2] - pts[1])
            c = np.linalg.norm(pts[2] - pts[0])

            if a * b * c > 1e-10:
                area = 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                curvatures[i] = 4 * area / (a * b * c)

        threshold = np.median(curvatures) + 3 * np.std(curvatures)
        return x[curvatures < threshold], y[curvatures < threshold]

    def _simplify_curve(
        self, x: FloatArray, y: FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Simplify using Douglas-Peucker algorithm.

        Args:
            x: X-coordinates
            y: Y-coordinates

        Returns:
            Tuple of (simplified_x, simplified_y)
        """
        points = np.column_stack((x, y))
        epsilon = 0.01 * self._domain_height
        simplified = self._douglas_peucker(points, epsilon)
        return simplified[:, 0], simplified[:, 1]

    def _douglas_peucker(self, points: FloatArray, epsilon: float) -> FloatArray:
        """
        Douglas-Peucker simplification.

        Args:
            points: Nx2 array of coordinates
            epsilon: Distance threshold

        Returns:
            Simplified array of points
        """
        if len(points) < 3:
            return points

        dmax, index = 0.0, 0
        for i in range(1, len(points) - 1):
            d = self._perp_distance(points[i], points[0], points[-1])
            if d > dmax:
                index, dmax = i, d

        if dmax > epsilon:
            rec1 = self._douglas_peucker(points[:index + 1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            return np.vstack((rec1[:-1], rec2))
        return np.array([points[0], points[-1]])

    @staticmethod
    def _perp_distance(
        point: FloatArray, line_start: FloatArray, line_end: FloatArray
    ) -> float:
        """
        Perpendicular distance from point to line.

        Args:
            point: Point coordinates
            line_start: Line start point
            line_end: Line end point

        Returns:
            Perpendicular distance
        """
        if np.allclose(line_start, line_end):
            return float(np.linalg.norm(point - line_start))
        return float(
            np.abs(np.cross(line_end - line_start, line_start - point))
            / np.linalg.norm(line_end - line_start)
        )


# ============================================================================
# GUI Components (Interface Segregation Principle)
# ============================================================================

class SettingsDialog(QDialog):
    """Dialog for editing plot settings."""

    def __init__(
        self, current_settings: PlotSettings, parent: Optional[QWidget] = None
    ):
        """
        Initialize settings dialog.

        Args:
            current_settings: Current plot settings
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self.settings = current_settings
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the user interface components."""
        layout = QGridLayout()

        # Input fields
        self.x_min_edit = QLineEdit(str(self.settings.x_min))
        self.x_max_edit = QLineEdit(str(self.settings.x_max))
        self.y_min_edit = QLineEdit(str(self.settings.y_min))
        self.y_max_edit = QLineEdit(str(self.settings.y_max))
        self.grid_edit = QLineEdit(str(self.settings.grid_spacing))

        self.accuracy_spinbox = QDoubleSpinBox()
        self.accuracy_spinbox.setRange(0.0001, 1.0)
        self.accuracy_spinbox.setSingleStep(0.0001)
        self.accuracy_spinbox.setDecimals(4)
        self.accuracy_spinbox.setValue(self.settings.accuracy)

        # Add labels and inputs
        layout.addWidget(QLabel("X Min:"), 0, 0)
        layout.addWidget(self.x_min_edit, 0, 1)
        layout.addWidget(QLabel("X Max:"), 1, 0)
        layout.addWidget(self.x_max_edit, 1, 1)
        layout.addWidget(QLabel("Y Min:"), 2, 0)
        layout.addWidget(self.y_min_edit, 2, 1)
        layout.addWidget(QLabel("Y Max:"), 3, 0)
        layout.addWidget(self.y_max_edit, 3, 1)
        layout.addWidget(QLabel("Grid Spacing:"), 4, 0)
        layout.addWidget(self.grid_edit, 4, 1)
        layout.addWidget(QLabel("Approximation Accuracy:"), 5, 0)
        layout.addWidget(self.accuracy_spinbox, 5, 1)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout, 6, 0, 1, 2)
        self.setLayout(layout)

    def get_settings(self) -> Optional[PlotSettings]:
        """
        Retrieve validated settings from dialog.

        Returns:
            PlotSettings if valid, None otherwise
        """
        try:
            return PlotSettings(
                x_min=float(self.x_min_edit.text()),
                x_max=float(self.x_max_edit.text()),
                y_min=float(self.y_min_edit.text()),
                y_max=float(self.y_max_edit.text()),
                grid_spacing=float(self.grid_edit.text()),
                accuracy=float(self.accuracy_spinbox.value()),
            )
        except (ValueError, TypeError):
            return None


class DrawingApp(QMainWindow):
    """Main application window for drawing and fitting curves."""

    COLORS = [
        (255, 0, 0), (0, 200, 0), (255, 150, 0),
        (200, 0, 200), (0, 200, 200),
    ]

    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.setWindowTitle("Function Drawer & LaTeX Generator")
        self.setGeometry(100, 100, 1400, 800)

        # Services (Dependency Injection)
        self.model_service = ModelSelectionService()
        self.latex_generator = LaTeXGenerator()

        # State
        self.settings = PlotSettings()
        self.drawing = False
        self.strokes: List[List[Tuple[float, float]]] = []
        self.current_stroke: Optional[List[Tuple[float, float]]] = None
        self.panning = False
        self.pan_start_pos: Optional[QPointF] = None
        self.pan_start_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

        # Display state
        self.fitted_curves: List[Any] = []
        self.drawn_curve: Optional[Any] = None
        self.all_models: List[FittedModel] = []
        self.option_checkboxes: List[QCheckBox] = []
        self.option_widgets: List[QWidget] = []

        # UI elements
        self.plot_widget: pg.PlotWidget
        self.clear_button: QPushButton
        self.fit_button: QPushButton
        self.export_button: QPushButton
        self.settings_button: QPushButton
        self.options_layout: QVBoxLayout
        self.latex_output: QTextEdit
        self.clear_on_new_line_checkbox: QCheckBox

        # Build and configure UI
        self._build_ui()
        self._configure_plot()

    def _build_ui(self) -> None:
        """Build the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: plot
        left_layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(10, 10))
        left_layout.addWidget(self.plot_widget)

        # Buttons
        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear")
        self.fit_button = QPushButton("Fit Curve")
        self.export_button = QPushButton("Copy LaTeX")
        self.settings_button = QPushButton("Settings")

        self.clear_button.clicked.connect(self.clear_drawing)
        self.fit_button.clicked.connect(self.fit_curve)
        self.export_button.clicked.connect(self.copy_latex)
        self.settings_button.clicked.connect(self.show_settings)

        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.settings_button)
        left_layout.addLayout(button_layout)

        main_layout.addLayout(left_layout, 3)

        # Right panel: options and output
        right_layout = QVBoxLayout()

        options_group = QGroupBox("Display Options")
        self.options_layout = QVBoxLayout()
        options_group.setLayout(self.options_layout)

        self.clear_on_new_line_checkbox = QCheckBox("Clear plot on a new line")
        self.clear_on_new_line_checkbox.setChecked(True)
        self.options_layout.addWidget(self.clear_on_new_line_checkbox)

        right_layout.addWidget(options_group)
        right_layout.addWidget(QLabel("LaTeX Output (Top Candidates):"))

        self.latex_output = QTextEdit()
        self.latex_output.setReadOnly(True)
        right_layout.addWidget(self.latex_output)

        main_layout.addLayout(right_layout, 1)

        # Configure plot interaction
        vb = self.plot_widget.plotItem.vb
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=True, y=True)
        self.plot_widget.viewport().installEventFilter(self)

    def _configure_plot(self) -> None:
        """Configure the plot display settings."""
        self.plot_widget.setLabel("left", "y")
        self.plot_widget.setLabel("bottom", "x")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        vb = self.plot_widget.plotItem.vb
        vb.disableAutoRange()
        self.plot_widget.setXRange(self.settings.x_min, self.settings.x_max)
        self.plot_widget.setYRange(self.settings.y_min, self.settings.y_max)

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        """
        Handle mouse events for drawing and panning.

        Args:
            obj: Object receiving the event
            event: Event to handle

        Returns:
            True if event was handled, False otherwise
        """
        if obj == self.plot_widget.viewport():
            if not isinstance(event, QMouseEvent):
                return super().eventFilter(obj, event)

            vb = self.plot_widget.plotItem.vb
            mouse_event = event

            if event.type() == QEvent.Type.MouseButtonPress:
                if mouse_event.button() == Qt.MouseButton.LeftButton:
                    if self.clear_on_new_line_checkbox.isChecked():
                        self.clear_drawing()

                    view_pos = vb.mapSceneToView(mouse_event.pos())
                    self.drawing = True
                    self.current_stroke = [(float(view_pos.x()), float(view_pos.y()))]
                    self.strokes.append(self.current_stroke)
                    return True

                if mouse_event.button() == Qt.MouseButton.RightButton:
                    # Start panning: store initial position and current view range
                    self.panning = True
                    self.pan_start_pos = vb.mapSceneToView(mouse_event.pos())

                    # Store current ranges
                    x_range = vb.viewRange()[0]
                    y_range = vb.viewRange()[1]
                    self.pan_start_range = (
                        (x_range[0], x_range[1]),
                        (y_range[0], y_range[1])
                    )

                    self.plot_widget.viewport().setCursor(
                        QCursor(Qt.CursorShape.ClosedHandCursor)
                    )
                    return True

            if event.type() == QEvent.Type.MouseMove:
                if self.drawing and self.current_stroke is not None:
                    view_pos = vb.mapSceneToView(mouse_event.pos())
                    self.current_stroke.append(
                        (float(view_pos.x()), float(view_pos.y()))
                    )
                    self.update_drawing()
                    return True

                if self.panning and self.pan_start_pos is not None and self.pan_start_range is not None:
                    # Get current mouse position in view coordinates
                    current_pos = vb.mapSceneToView(mouse_event.pos())

                    # Calculate offset in view coordinates (intuitive: drag direction = pan direction)
                    dx = float(self.pan_start_pos.x() - current_pos.x())
                    dy = float(self.pan_start_pos.y() - current_pos.y())

                    # Apply offset to original ranges
                    x_range_start, y_range_start = self.pan_start_range
                    new_x_min = x_range_start[0] + dx
                    new_x_max = x_range_start[1] + dx
                    new_y_min = y_range_start[0] + dy
                    new_y_max = y_range_start[1] + dy

                    # Update view ranges
                    vb.setRange(xRange=(new_x_min, new_x_max), yRange=(new_y_min, new_y_max), padding=0)
                    return True

            if event.type() == QEvent.Type.MouseButtonRelease:
                if mouse_event.button() == Qt.MouseButton.LeftButton:
                    self.drawing = False
                    self.current_stroke = None
                    return True

                if mouse_event.button() == Qt.MouseButton.RightButton:
                    self.panning = False
                    self.pan_start_pos = None
                    self.pan_start_range = None
                    self.plot_widget.viewport().unsetCursor()
                    return True

        return super().eventFilter(obj, event)

    def clear_drawing(self) -> None:
        """Clear all drawn strokes and fitted curves."""
        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.panning = False
        self.pan_start_pos = None
        self.pan_start_range = None

        self.plot_widget.viewport().unsetCursor()

        if self.drawn_curve is not None:
            self.plot_widget.removeItem(self.drawn_curve)
            self.drawn_curve = None

        for curve in self.fitted_curves:
            self.plot_widget.removeItem(curve)
        self.fitted_curves = []

        for widget in self.option_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.option_widgets = []
        self.option_checkboxes = []

        self.latex_output.clear()
        self.all_models = []

    def update_drawing(self) -> None:
        """Update the visual representation of drawn strokes."""
        xs: List[float] = []
        ys: List[float] = []

        for stroke in self.strokes:
            if not stroke:
                continue
            if xs:
                xs.append(np.nan)
                ys.append(np.nan)
            sx, sy = zip(*stroke)
            xs.extend(sx)
            ys.extend(sy)

        if len(xs) < 2:
            if self.drawn_curve is not None:
                self.plot_widget.removeItem(self.drawn_curve)
                self.drawn_curve = None
            return

        x_arr, y_arr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        if self.drawn_curve is None:
            self.drawn_curve = self.plot_widget.plot(
                x_arr, y_arr,
                pen=pg.mkPen((100, 100, 255), width=2),
                name="Drawn curve"
            )
        else:
            self.drawn_curve.setData(x_arr, y_arr)

    def fit_curve(self) -> None:
        """Fit mathematical models to the drawn curve."""
        # Collect all points
        points = []
        for stroke in self.strokes:
            points.extend(stroke)

        if len(points) < 10:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Draw a longer curve (at least 10 points)"
            )
            return

        points_array = np.array(points, dtype=float)
        x_raw, y_raw = points_array[:, 0], points_array[:, 1]

        # Preprocess
        try:
            preprocessor = StrokePreprocessor(
                self.settings.domain_width,
                self.settings.domain_height
            )
            x_proc, y_proc = preprocessor.preprocess(x_raw, y_raw)

            if len(x_proc) < 5:
                QMessageBox.warning(
                    self, "Processing Error",
                    "Not enough points after preprocessing"
                )
                return

            is_func, error_msg = preprocessor.is_function(x_proc, y_proc)
            if not is_func:
                QMessageBox.warning(self, "Not a Function", error_msg)
                return
        # Catch all exceptions to ensure robustness of preprocessing
        except Exception as e:  # pylint: disable=broad-exception-caught
            QMessageBox.critical(
                self, "Error",
                f"Preprocessing failed: {str(e)}"
            )
            return

        # Fit models
        try:
            models = self.model_service.fit_all_models(
                x_proc, y_proc, self.settings.accuracy
            )

            if not models:
                QMessageBox.warning(
                    self, "Fitting Failed",
                    "Could not fit any model to the data"
                )
                return

            best_model = self.model_service.select_best_model(models, y_proc)
        # Catch all exceptions to ensure robustness of model fitting
        except Exception as e:  # pylint: disable=broad-exception-caught
            QMessageBox.critical(
                self, "Error",
                f"Model fitting failed: {str(e)}"
            )
            return

        # Display results
        self._display_fitted_models(models, best_model, x_proc, y_proc)

    def _display_fitted_models(
        self,
        models: List[FittedModel],
        best_model: FittedModel,
        x_proc: FloatArray,
        y_proc: FloatArray
    ) -> None:
        """
        Display fitted models on the plot.

        Args:
            models: List of all fitted models
            best_model: The best selected model
            x_proc: Processed x-coordinates
            y_proc: Processed y-coordinates
        """
        # Sort models by quality
        y_std = np.std(y_proc)
        scores = [
            (m.rmse / y_std if y_std > 0 else m.rmse) + 0.1 * m.aic + m.complexity
            for m in models
        ]
        sorted_indices_int = np.argsort(scores).astype(np.int64)
        top_models = [models[int(i)] for i in sorted_indices_int[:min(5, len(models))]]

        self.all_models = top_models

        # Clear previous
        for curve in self.fitted_curves:
            self.plot_widget.removeItem(curve)
        self.fitted_curves = []

        for widget in self.option_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.option_widgets = []
        self.option_checkboxes = []

        # Display new models
        latex_parts = []
        x_plot = np.linspace(np.min(x_proc), np.max(x_proc), 500)

        for idx, model in enumerate(top_models):
            latex_str = self.latex_generator.generate(model)
            marker = "★ BEST FIT" if model == best_model else ""
            latex_parts.append(
                f"Option {idx + 1} - {model.name} {marker}\n"
                f"RMSE: {model.rmse:.6f}\n{latex_str}\n"
            )

            # Plot curve
            try:
                y_plot = model.evaluate(x_plot)
                color = self.COLORS[idx % len(self.COLORS)]
                pen_style = (
                    Qt.PenStyle.SolidLine if model == best_model
                    else Qt.PenStyle.DashLine
                )
                pen_width = 3 if model == best_model else 2

                curve = self.plot_widget.plot(
                    x_plot, y_plot,
                    pen=pg.mkPen(color, width=pen_width, style=pen_style),
                    name=f"Option {idx + 1}: {model.name}"
                )
                self.fitted_curves.append(curve)
            # Catch all exceptions to ensure one model failure doesn't stop display
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            # Add checkbox
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, i=idx: self.toggle_option(i, checked)
            )

            label = QLabel(f"Option {idx + 1}: {model.name} {marker}")
            label.setStyleSheet(f"color: rgb{color}; font-weight: bold;")

            row_layout.addWidget(checkbox)
            row_layout.addWidget(label)
            row_layout.addStretch(1)

            self.options_layout.addWidget(row)
            self.option_widgets.append(row)
            self.option_checkboxes.append(checkbox)

        self.latex_output.setPlainText("\n".join(latex_parts))

    def toggle_option(self, index: int, checked: bool) -> None:
        """
        Toggle visibility of a fitted curve.

        Args:
            index: Index of the curve to toggle
            checked: Whether to show or hide the curve
        """
        if index < len(self.fitted_curves):
            self.fitted_curves[index].setVisible(checked)

    def copy_latex(self) -> None:
        """Copy LaTeX output to clipboard."""
        latex_text = self.latex_output.toPlainText()
        if latex_text:
            QApplication.clipboard().setText(latex_text)
            QMessageBox.information(self, "Copied", "LaTeX copied to clipboard")

    def show_settings(self) -> None:
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            if new_settings:
                try:
                    self.settings = new_settings
                    self._configure_plot()
                    self.clear_drawing()
                except ValueError as e:
                    QMessageBox.critical(self, "Invalid Settings", str(e))


# ============================================================================
# Application Entry Point
# ============================================================================

def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    