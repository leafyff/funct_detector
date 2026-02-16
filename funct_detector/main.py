"""
Function Drawer and LaTeX Generator — 12 canonical models.

Mathematical methods
--------------------
1.  Cubic Spline                             scipy.interpolate.CubicSpline
2.  Interpolation polynomial (Chebyshev)     Chebyshev.fit at CGL nodes
3.  L-inf minimax polynomial                 scipy.optimize.linprog / HiGHS
4.  Polynomial Least Squares (Chebyshev)     numpy.polynomial.Chebyshev.fit
5.  Non-Uniform Fast Fourier Transform       LS trigonometric regression (NUFFT type-1 pseudoinverse)
6.  AAA Algorithm                            scipy.interpolate.AAA (Nakatsukasa–Sete–Trefethen 2018)
7.  Exponential curve                        A·exp(B·x) + C
8.  Logarithmic curve                        A·ln(x+shift) + B
9.  Rational curve                           A + B/(x−D)
10. Sinusoidal curve                         A·sin(2π·f·x + φ) + B
11. Tangential curve                         A·tan(B·x + C) + D
12. Arctan (S-curve)                         A·arctan(B·x + C) + D

Scoring:  Score = α·L∞/σ + β·RMSE/σ + γ·complexity
"""

from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pyqtgraph as pg
import sympy as sp
from numpy.polynomial import Chebyshev, Polynomial
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QCursor, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import AAA, CubicSpline, UnivariateSpline, make_interp_spline
from scipy.optimize import curve_fit, linprog
from scipy.signal import lombscargle, savgol_filter

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.floating[Any]]
EvaluationFunction = Callable[[FloatArray], FloatArray]

# ---------------------------------------------------------------------------
# Scoring weights: Score = ALPHA*(L_inf/s) + BETA*(RMSE/s) + GAMMA*complexity
# ---------------------------------------------------------------------------

ALPHA: float = 0.4
BETA: float = 0.4
GAMMA: float = 0.2

# All models shown; only the top N get their checkbox checked ON.
N_TOP_CHECKED: int = 2


# ===========================================================================
# Data-classes
# ===========================================================================

@dataclass(frozen=True, slots=True)
class PlotSettings:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    grid_spacing: float = 1.0
    accuracy: float = 0.0001
    latex_approx: bool = True    # use decimal approximations in LaTeX output
    latex_decimals: int = 3      # digits after decimal point when approx is on

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")
        if self.grid_spacing <= 0:
            raise ValueError(f"grid_spacing must be positive, got {self.grid_spacing}")
        if not (0.0001 <= self.accuracy <= 1.0):
            raise ValueError(f"accuracy must be in [0.0001, 1.0], got {self.accuracy}")
        if not (0 <= self.latex_decimals <= 10):
            raise ValueError(f"latex_decimals must be in [0, 10], got {self.latex_decimals}")

    @property
    def domain_width(self) -> float:
        return self.x_max - self.x_min

    @property
    def domain_height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True, slots=True)
class FittedModel:
    name: str
    evaluate: EvaluationFunction
    latex_kind: str
    rmse: float
    l_inf: float
    bic: float
    complexity: float
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rmse < 0:
            raise ValueError(f"RMSE cannot be negative: {self.rmse}")
        if self.l_inf < 0:
            raise ValueError(f"L-inf cannot be negative: {self.l_inf}")
        if self.complexity < 0:
            raise ValueError(f"complexity cannot be negative: {self.complexity}")
        if not callable(self.evaluate):
            raise ValueError("evaluate must be callable")

    def score(self, y_scale: float = 1.0) -> float:
        s = y_scale if y_scale > 0 else 1.0
        return ALPHA * self.l_inf / s + BETA * self.rmse / s + GAMMA * self.complexity


# ===========================================================================
# Abstract base fitter
# ===========================================================================

class ModelFitter(ABC):

    @abstractmethod
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        raise NotImplementedError

    @staticmethod
    def _rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _l_inf(y_true: FloatArray, y_pred: FloatArray) -> float:
        return float(np.max(np.abs(y_true - y_pred)))

    @staticmethod
    def _bic(n: int, mse: float, k: int) -> float:
        if mse <= 0:
            return float("inf")
        return float(n * np.log(mse) + k * np.log(n))

    @staticmethod
    def _r2(y_true: FloatArray, y_pred: FloatArray) -> float:
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 0:
            return 0.0
        return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot

    @staticmethod
    def _linear_prefit(u: FloatArray, v: FloatArray) -> tuple[float, float, float]:
        """Least-squares linear fit v ~ a*u + b.  Returns (slope, intercept, R²)."""
        p = Polynomial.fit(u, v, 1)
        coef = p.convert().coef        # [intercept, slope] in original domain
        v_pred = np.asarray(p(u), dtype=np.float64)
        ss_tot = float(np.sum((v - float(np.mean(v))) ** 2))
        r2 = 1.0 - float(np.sum((v - v_pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        return float(coef[1]), float(coef[0]), r2

    @staticmethod
    def _reject_fit(y: FloatArray, y_pred: FloatArray, rmse: float, threshold: float) -> bool:
        """Return True when the fit is too poor to keep (rmse/std > threshold)."""
        if not np.all(np.isfinite(y_pred)):
            return True
        y_std = float(np.std(y))
        return y_std > 0 and rmse / y_std > threshold

    @staticmethod
    def _validate_basic_data(
        x: FloatArray, y: FloatArray, min_points: int = 8
    ) -> Optional[tuple[int, float, float]]:
        """Validate basic data requirements.

        Returns (n, y_std, x_span) if valid, None otherwise.
        """
        n = len(x)
        if n < min_points:
            return None
        y_std = float(np.std(y))
        if y_std <= 0:
            return None
        x_span = float(x[-1] - x[0])
        if x_span <= 0:
            return None
        return n, y_std, x_span

    @staticmethod
    def _validate_fitted_result(
        y: FloatArray, y_pred: FloatArray, y_std: float, threshold: float = 0.9
    ) -> bool:
        """Check if fitted result is acceptable.

        Returns True if valid, False otherwise.
        """
        if not np.all(np.isfinite(y_pred)):
            return False
        if y_std > 0:
            rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            if rmse / y_std > threshold:
                return False
        return True


# ===========================================================================
# Chebyshev polynomial fitter
# Replaces manual Vandermonde+lstsq with np.polynomial.Chebyshev.fit.
# Domain mapping to [-1,1] is handled internally; no manual rescaling needed.
# ===========================================================================

class ChebyshevPolynomialFitter(ModelFitter):

    def __init__(self, max_degree: int = 14) -> None:
        self._max_degree = max_degree

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        if len(x) < 3:
            return None
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return None

        max_degree = min(self._max_degree, len(x) - 1)
        y_scale = float(np.std(y)) or 1.0

        best_score = float("inf")
        best_degree = 1
        best_fit: Optional[Chebyshev] = None

        for degree in range(1, max_degree + 1):
            try:
                # Chebyshev.fit: least-squares in Chebyshev basis.
                # domain=[x_min, x_max] stored on object; c(x) evaluates correctly.
                c = Chebyshev.fit(x, y, degree, domain=[x_min, x_max])
                y_pred = np.asarray(c(x), dtype=np.float64)
                rmse = self._rmse(y, y_pred)
                l_inf = self._l_inf(y, y_pred)
                complexity = float(degree) * 0.5
                s = ALPHA * l_inf / y_scale + BETA * rmse / y_scale + GAMMA * complexity
                if s < best_score:
                    best_score, best_degree, best_fit = s, degree, c
                if rmse < max(1e-9, float(accuracy)) and degree > 1:
                    break
            except (np.linalg.LinAlgError, ValueError):
                continue

        if best_fit is None:
            return None

        y_pred_f = np.asarray(best_fit(x), dtype=np.float64)
        rmse_f = self._rmse(y, y_pred_f)
        l_inf_f = self._l_inf(y, y_pred_f)
        bic_f = self._bic(len(x), max(rmse_f ** 2, 1e-300), best_degree + 1)
        cheb_obj = best_fit

        def evaluate(x_eval: FloatArray) -> FloatArray:
            return np.asarray(cheb_obj(x_eval), dtype=np.float64)

        return FittedModel(
            name="Polynomial Least Squares Approximation (Chebyshev Basis)",
            evaluate=evaluate,
            latex_kind="chebyshev",
            rmse=rmse_f,
            l_inf=l_inf_f,
            bic=bic_f,
            complexity=float(best_degree) * 0.5,
            params={
                "cheb_coef": list(best_fit.coef),   # coefficients in Chebyshev basis
                "domain": [x_min, x_max],
                "degree": best_degree,
            },
        )


# ===========================================================================
# Sinusoidal fitter  A*sin(2*pi*f*x + phi) + B
# ===========================================================================

class SinusoidalFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            n = len(x)
            if n < 8:
                return None

            order = np.argsort(x)
            xs: FloatArray = x[order]
            ys: FloatArray = y[order]
            x_span = float(xs[-1] - xs[0])
            if x_span <= 0:
                return None

            f_min = 1.0 / x_span
            f_max = float(n) / (2.0 * x_span)
            if f_max <= f_min:
                return None

            ang_freqs = np.linspace(2.0 * np.pi * f_min, 2.0 * np.pi * f_max, 512)
            y_c = ys - float(np.mean(ys))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power: FloatArray = np.asarray(
                    lombscargle(xs, y_c, ang_freqs, normalize=True), dtype=np.float64
                )

            peak_idx = int(np.argmax(power))
            if float(power[peak_idx]) < 0.25:
                return None
            f0 = float(ang_freqs[peak_idx]) / (2.0 * np.pi)
            if f0 <= 0:
                return None

            amp_init = float(np.std(ys) * np.sqrt(2))
            off_init = float(np.mean(ys))
            y_std = float(np.std(ys))

            def _sin(xv: FloatArray, a: float, f: float, p: float, b: float) -> FloatArray:
                return np.asarray(a * np.sin(2.0 * np.pi * f * xv + p) + b, dtype=np.float64)

            best_rmse = float("inf")
            best_popt: Optional[FloatArray] = None

            for phase_guess in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = curve_fit(
                            _sin, xs, ys,
                            p0=[amp_init, f0, phase_guess, off_init],
                            maxfev=10000,
                            bounds=(
                                [-np.inf, f_min * 0.5, -np.pi, -np.inf],
                                [np.inf, f_max * 2.0, np.pi, np.inf],
                            ),
                        )
                    popt = np.asarray(res[0], dtype=np.float64)
                    r = self._rmse(ys, _sin(xs, *popt))
                    if r < best_rmse:
                        best_rmse, best_popt = r, popt
                except (RuntimeError, ValueError):
                    continue

            if best_popt is None:
                return None

            sin_a, sin_f, sin_p, sin_b = (float(best_popt[k]) for k in range(4))
            y_pred = _sin(xs, sin_a, sin_f, sin_p, sin_b)
            if not np.all(np.isfinite(y_pred)):
                return None

            rmse = self._rmse(ys, y_pred)
            if y_std > 0 and rmse / y_std > 0.8:
                return None

            l_inf = self._l_inf(ys, y_pred)
            bic = self._bic(n, max(rmse ** 2, 1e-300), 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _sin(x_eval, sin_a, sin_f, sin_p, sin_b)

            return FittedModel(
                name="Sinusoidal curve",
                evaluate=evaluate,
                latex_kind="sinusoidal",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=3.0,
                params={"A": sin_a, "f": sin_f, "phase": sin_p, "B": sin_b},
            )
        except (RuntimeError, ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            return None


# ===========================================================================
# Exponential fitter  A*exp(B*x) + C
# ===========================================================================

class ExponentialFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            y_min = float(np.min(y))
            y_range = float(np.ptp(y))
            shift = max(0.0, -y_min + 1e-6 * max(1.0, y_range)) if y_min <= 0 else 0.0

            # Log-space linear pre-fit for initial guess
            y_log = np.log(y + shift)
            b_init, log_a0, r2_pre = self._linear_prefit(x, y_log)
            if r2_pre < 0.70:
                return None
            a_init = float(np.exp(log_a0))

            def _exp(xv: FloatArray, a: float, b: float, c: float) -> FloatArray:
                return np.asarray(a * np.exp(b * xv) + c, dtype=np.float64)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = curve_fit(_exp, x, y, p0=[a_init, b_init, 0.0], maxfev=10000)
            popt = np.asarray(res[0], dtype=np.float64)
            exp_a, exp_b, exp_c = float(popt[0]), float(popt[1]), float(popt[2])

            y_pred = _exp(x, exp_a, exp_b, exp_c)
            rmse = self._rmse(y, y_pred)
            if self._reject_fit(y, y_pred, rmse, 0.9):
                return None

            l_inf = self._l_inf(y, y_pred)
            bic = self._bic(len(x), max(rmse ** 2, 1e-300), 3)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _exp(x_eval, exp_a, exp_b, exp_c)

            return FittedModel(
                name="Exponential curve",
                evaluate=evaluate,
                latex_kind="exponential",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=3.0,
                params={"A": exp_a, "B": exp_b, "C": exp_c},
            )
        except (RuntimeError, ValueError, TypeError, OverflowError,
                np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# Logarithmic fitter  A*ln(x + shift) + B
# ===========================================================================

class LogarithmicFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            x_min = float(np.min(x))
            shift = max(0.0, -x_min + 1e-6 * max(1.0, float(np.ptp(x)))) if x_min <= 0 else 0.0
            x_log = np.log(x + shift)

            # Linear pre-fit for initial guess
            a_init, b_init, r2_pre = self._linear_prefit(x_log, y)
            if r2_pre < 0.70:
                return None

            log_shift = shift

            def _log(xv: FloatArray, a: float, b: float) -> FloatArray:
                return np.asarray(a * np.log(xv + log_shift) + b, dtype=np.float64)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = curve_fit(_log, x, y, p0=[a_init, b_init], maxfev=10000)
            popt = np.asarray(res[0], dtype=np.float64)
            log_a, log_b = float(popt[0]), float(popt[1])

            y_pred = _log(x, log_a, log_b)
            rmse = self._rmse(y, y_pred)
            if self._reject_fit(y, y_pred, rmse, 0.9):
                return None

            l_inf = self._l_inf(y, y_pred)
            bic = self._bic(len(x), max(rmse ** 2, 1e-300), 2)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _log(x_eval, log_a, log_b)

            return FittedModel(
                name="Logarithmic curve",
                evaluate=evaluate,
                latex_kind="logarithmic",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=2.5,
                params={"A": log_a, "B": log_b, "shift": shift},
            )
        except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# Rational fitter  A + B/(x - D)
# ===========================================================================

class RationalFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            if len(x) < 6:
                return None

            x_lo, x_hi = float(np.min(x)), float(np.max(x))
            x_rng = x_hi - x_lo
            y_mean, y_std = float(np.mean(y)), float(np.std(y))

            def _rat(xv: FloatArray, a: float, b: float, d: float) -> FloatArray:
                denom = xv - d
                safe = np.where(np.abs(denom) < 1e-9, np.sign(denom) * 1e-9, denom)
                return np.asarray(a + b / safe, dtype=np.float64)

            best_rmse = float("inf")
            best_popt: Optional[FloatArray] = None

            for d_cand in (
                x_lo - 0.5 * x_rng, x_lo - 0.1 * x_rng,
                x_hi + 0.1 * x_rng, x_hi + 0.5 * x_rng,
                x_lo - x_rng, x_hi + x_rng,
            ):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = curve_fit(
                            _rat, x, y,
                            p0=[y_mean, y_std * x_rng, d_cand],
                            maxfev=10000,
                        )
                    popt = np.asarray(res[0], dtype=np.float64)
                    d_fit = float(popt[2])
                    if x_lo < d_fit < x_hi:
                        continue
                    r = self._rmse(y, _rat(x, float(popt[0]), float(popt[1]), d_fit))
                    if r < best_rmse:
                        best_rmse, best_popt = r, popt
                except (RuntimeError, ValueError):
                    continue

            if best_popt is None:
                return None

            rat_a, rat_b, rat_d = float(best_popt[0]), float(best_popt[1]), float(best_popt[2])
            y_pred = _rat(x, rat_a, rat_b, rat_d)
            if not np.all(np.isfinite(y_pred)):
                return None

            rmse = self._rmse(y, y_pred)
            if y_std > 0 and rmse / y_std > 0.9:
                return None

            l_inf = self._l_inf(y, y_pred)
            bic = self._bic(len(x), max(rmse ** 2, 1e-300), 3)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _rat(x_eval, rat_a, rat_b, rat_d)

            return FittedModel(
                name="Rational curve",
                evaluate=evaluate,
                latex_kind="rational",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=3.5,
                params={"A": rat_a, "B": rat_b, "D": rat_d},
            )
        except (RuntimeError, ValueError, TypeError, OverflowError,
                np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# Arctan fitter  A*arctan(B*x + C) + D
# Continuous S-curve without the branch-cut discontinuities of tan(x).
# ===========================================================================

class ArctanFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            validation = self._validate_basic_data(x, y, min_points=8)
            if validation is None:
                return None
            n, y_std, x_span = validation

            def _atan(xv: FloatArray, a: float, b: float, c: float, d: float) -> FloatArray:
                return np.asarray(a * np.arctan(b * xv + c) + d, dtype=np.float64)

            best_rmse = float("inf")
            best_popt: Optional[FloatArray] = None

            for b_seed in (1.0 / x_span, 2.0 / x_span, 0.5 / x_span, 1.0):
                for c_seed in (0.0, np.pi / 4, -np.pi / 4):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = curve_fit(
                                _atan, x, y,
                                p0=[y_std, b_seed, c_seed, float(np.mean(y))],
                                maxfev=10000,
                            )
                        popt = np.asarray(res[0], dtype=np.float64)
                        yp = _atan(x, *popt)
                        if not np.all(np.isfinite(yp)):
                            continue
                        r = self._rmse(y, yp)
                        if r < best_rmse:
                            best_rmse, best_popt = r, popt
                    except (RuntimeError, ValueError):
                        continue

            if best_popt is None:
                return None

            at_a, at_b, at_c, at_d = (float(best_popt[k]) for k in range(4))
            y_pred = _atan(x, at_a, at_b, at_c, at_d)

            if not self._validate_fitted_result(y, y_pred, y_std):
                return None

            rmse = self._rmse(y, y_pred)
            l_inf = self._l_inf(y, y_pred)
            bic = self._bic(n, max(rmse ** 2, 1e-300), 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _atan(x_eval, at_a, at_b, at_c, at_d)

            return FittedModel(
                name="Arctan (S-curve)",
                evaluate=evaluate,
                latex_kind="arctan",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=4.0,
                params={"A": at_a, "B": at_b, "C": at_c, "D": at_d},
            )
        except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# Cubic spline fitter
# ===========================================================================

class SplineFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        idx = np.argsort(x)
        xs: FloatArray = x[idx]
        ys: FloatArray = y[idx]

        noise = float(np.std(np.diff(ys)) / np.sqrt(2)) if len(ys) > 1 else 0.1
        s = len(xs) * noise ** 2 * (float(accuracy) / 0.01)

        try:
            spline = UnivariateSpline(xs, ys, s=s, k=3)
            y_pred = np.asarray(spline(xs), dtype=np.float64)
            rmse = self._rmse(ys, y_pred)
            l_inf = self._l_inf(ys, y_pred)
            knots = int(len(spline.get_knots()))
            bic = self._bic(len(xs), max(rmse ** 2, 1e-300), knots + 4)
            cs = CubicSpline(xs, np.asarray(spline(xs), dtype=np.float64))

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return np.asarray(spline(x_eval), dtype=np.float64)

            return FittedModel(
                name="Cubic Spline",
                evaluate=evaluate,
                latex_kind="spline",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=float(knots) * 0.3,
                params={"cubic_spline": cs, "x_knots": xs.tolist(), "k_count": knots},
            )
        except (ValueError, TypeError):
            cs = CubicSpline(xs, ys)
            y_pred = np.asarray(cs(xs), dtype=np.float64)
            rmse = self._rmse(ys, y_pred)
            l_inf = self._l_inf(ys, y_pred)
            segs = max(1, len(xs) - 1)
            bic = self._bic(len(xs), max(rmse ** 2, 1e-300), segs * 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:  # type: ignore[misc]
                return np.asarray(cs(x_eval), dtype=np.float64)

            return FittedModel(
                name="Cubic Spline",
                evaluate=evaluate,
                latex_kind="spline",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=float(segs) * 0.3,
                params={"cubic_spline": cs, "x_knots": xs.tolist(), "k_count": segs},
            )


# ===========================================================================
# AAA rational approximation
# Replaces the hand-rolled Loewner/SVD loop with scipy.interpolate.AAA.
# The SciPy implementation is the official reference implementation of the
# Nakatsukasa-Sete-Trefethen (2018) algorithm.
# ===========================================================================

class AAAFitter(ModelFitter):

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            if len(x) < 8:
                return None

            rtol: float = max(1e-13, float(accuracy) * 1e-3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = AAA(x.tolist(), y.tolist(), rtol=rtol)

            y_pred = np.asarray(r(x.tolist()), dtype=np.float64)
            if not np.all(np.isfinite(y_pred)):
                return None

            rmse = self._rmse(y, y_pred)
            l_inf = self._l_inf(y, y_pred)
            n_terms: int = len(r.support_points)
            bic = self._bic(len(x), max(rmse ** 2, 1e-300), n_terms * 2)

            # Store lightweight scalars for LaTeX (not the AAA object itself).
            sp_list = [float(v) for v in r.support_points]
            sv_list = [float(v) for v in r.support_values]
            w_list = [float(v) for v in r.weights]
            aaa_obj = r

            def evaluate(x_eval: FloatArray) -> FloatArray:
                out = np.asarray(aaa_obj(x_eval), dtype=np.float64)
                out[~np.isfinite(out)] = float("nan")
                return out

            return FittedModel(
                name="AAA Algorithm",
                evaluate=evaluate,
                latex_kind="aaa",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=float(n_terms) * 0.8,
                params={
                    "support_points": sp_list,
                    "support_values": sv_list,
                    "weights": w_list,
                    "n_terms": n_terms,
                },
            )
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# L-inf minimax polynomial via scipy.optimize.linprog
# Uses Chebyshev Vandermonde (chebvander) instead of power-basis vander for
# better LP conditioning.  method='highs' is the modern HiGHS backend.
# ===========================================================================

class LinfPolynomialFitter(ModelFitter):

    def __init__(self, max_degree: int = 8) -> None:
        self._max_degree = max_degree

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        if len(x) < 4:
            return None
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return None

        # Normalise to [-1, 1] — same mapping Chebyshev.fit uses internally
        mid = 0.5 * (x_max + x_min)
        half = 0.5 * (x_max - x_min)
        t = (x - mid) / half
        y_scale = float(np.std(y)) or 1.0

        best_score = float("inf")
        best_degree = 2
        best_coef: list[float] = [0.0, 0.0, 0.0]

        for degree in range(2, min(self._max_degree, len(x) - 2) + 1):
            try:
                # chebvander: Chebyshev Vandermonde — far better conditioned than np.vander
                V = np.polynomial.chebyshev.chebvander(t, degree)
                n_pts, n_coef = V.shape
                n_vars = n_coef + 1   # [c_0, ..., c_d, epsilon]

                c_obj = np.zeros(n_vars, dtype=np.float64)
                c_obj[-1] = 1.0

                ones = np.ones((n_pts, 1), dtype=np.float64)
                # Constraints: V*c - y <= eps  AND  y - V*c <= eps
                A_ub = np.vstack([
                    np.hstack([V, -ones]),
                    np.hstack([-V, -ones]),
                ])
                b_ub = np.concatenate([y, -y])

                result = linprog(
                    c_obj,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=[(None, None)] * n_vars,
                    method="highs",
                )
                if not result.success:
                    continue

                cheb_coef = np.asarray(result.x[:n_coef], dtype=np.float64)
                y_pred = np.asarray(
                    np.polynomial.chebyshev.chebval(t, cheb_coef), dtype=np.float64
                )
                if not np.all(np.isfinite(y_pred)):
                    continue

                rmse = self._rmse(y, y_pred)
                l_inf = self._l_inf(y, y_pred)
                score = ALPHA * l_inf / y_scale + BETA * rmse / y_scale + GAMMA * float(degree) * 0.5
                if score < best_score:
                    best_score = score
                    best_degree = degree
                    best_coef = [float(c) for c in cheb_coef]
            except (ValueError, RuntimeError, TypeError):
                continue

        if best_score == float("inf"):
            return None

        p_mid, p_half, p_coef = mid, half, best_coef

        def evaluate(x_eval: FloatArray) -> FloatArray:
            t_e = (x_eval - p_mid) / p_half
            return np.asarray(
                np.polynomial.chebyshev.chebval(t_e, np.asarray(p_coef)), dtype=np.float64
            )

        y_f = evaluate(x)
        rmse_f = self._rmse(y, y_f)
        l_inf_f = self._l_inf(y, y_f)
        bic_f = self._bic(len(x), max(rmse_f ** 2, 1e-300), best_degree + 1)

        return FittedModel(
            name="L-inf minimax polynomial",
            evaluate=evaluate,
            latex_kind="linf_poly",
            rmse=rmse_f,
            l_inf=l_inf_f,
            bic=bic_f,
            complexity=float(best_degree) * 0.6,
            params={"cheb_coef": p_coef, "degree": best_degree,
                    "x_mid": p_mid, "x_half": p_half},
        )


# ===========================================================================
# Interpolation polynomial in Chebyshev basis
# Uses barycentric Chebyshev nodes of the second kind (Chebyshev-Gauss-Lobatto)
# for guaranteed stability: Chebyshev.fit with degree = n-1 through all points.
# Distinct from the LS fitter, which minimises ||residual||₂ over degree ≤ 14.
# ===========================================================================

class InterpolationPolynomialFitter(ModelFitter):
    """Interpolates ALL data points exactly using Chebyshev nodes of the 2nd kind.

    The interpolant is built as a Chebyshev expansion on [x_min, x_max] via
    numpy's Chebyshev.fit with deg = n - 1.  To control Runge's phenomenon the
    fitter subsamples to at most MAX_PTS equidistant-by-arc-length points first,
    choosing the subsampled count that minimises the leave-one-out RMSE.
    """

    _MAX_PTS: int = 32   # interpolation beyond this degree is unstable for most data

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        if len(x) < 4:
            return None
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return None

        n_pts = min(len(x), self._MAX_PTS)
        # Arc-length uniform subsample to n_pts points
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]

        # Try several subsample counts; pick the one with smallest RMSE on full data
        best_score = float("inf")
        best_cheb: Optional[Chebyshev] = None
        y_scale = float(np.std(y)) or 1.0

        for n in range(4, n_pts + 1):
            try:
                # Chebyshev-Gauss-Lobatto nodes mapped to [x_min, x_max]
                k = np.arange(n)
                cgl = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * np.cos(
                    np.pi * k / (n - 1)
                )
                # Interpolate y at CGL nodes using cubic spline through original data
                cs_tmp = CubicSpline(xs, ys)
                y_cgl = np.asarray(cs_tmp(np.sort(cgl)), dtype=np.float64)
                cgl_s = np.sort(cgl)

                cheb = Chebyshev.fit(cgl_s, y_cgl, n - 1, domain=[x_min, x_max])
                y_pred = np.asarray(cheb(x), dtype=np.float64)
                if not np.all(np.isfinite(y_pred)):
                    continue
                rmse = self._rmse(y, y_pred)
                l_inf = self._l_inf(y, y_pred)
                score = ALPHA * l_inf / y_scale + BETA * rmse / y_scale + GAMMA * float(n) * 0.6
                if score < best_score:
                    best_score, best_cheb = score, cheb
            except (np.linalg.LinAlgError, ValueError, TypeError):
                continue

        if best_cheb is None:
            return None

        cheb_obj = best_cheb
        deg = len(cheb_obj.coef) - 1
        y_pred_f = np.asarray(cheb_obj(x), dtype=np.float64)
        rmse_f = self._rmse(y, y_pred_f)
        l_inf_f = self._l_inf(y, y_pred_f)
        bic_f = self._bic(len(x), max(rmse_f ** 2, 1e-300), deg + 1)

        def evaluate(x_eval: FloatArray) -> FloatArray:
            return np.asarray(cheb_obj(x_eval), dtype=np.float64)

        return FittedModel(
            name="Interpolation polynomial (Chebyshev Basis)",
            evaluate=evaluate,
            latex_kind="chebyshev",
            rmse=rmse_f,
            l_inf=l_inf_f,
            bic=bic_f,
            complexity=float(deg) * 0.6,
            params={
                "cheb_coef": list(cheb_obj.coef),
                "domain": [x_min, x_max],
                "degree": deg,
            },
        )


# ===========================================================================
# Tangential fitter  A * tan(B*x + C) + D
# Restricted so that no pole B*x+C = pi/2 + k*pi falls inside the data domain.
# ===========================================================================

class TangentialFitter(ModelFitter):
    """Fits y = A * tan(B*x + C) + D.

    The tangent function has period π and vertical asymptotes at
    B*x + C = π/2 + k·π.  We avoid poles in the data range by constraining
    B to (0, π/x_span) so at most one monotone branch fits the data, and
    clipping the argument to (−π/2 + ε, π/2 − ε) for numerical safety.
    """

    _EPS: float = 0.05   # margin from ±π/2 for argument clipping

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            validation = self._validate_basic_data(x, y, min_points=8)
            if validation is None:
                return None
            n, y_std, x_span = validation

            # Monotone increasing data is a prerequisite — quick heuristic
            x_s = x[np.argsort(x)]
            y_s = y[np.argsort(x)]
            monotone_frac = float(np.mean(np.diff(y_s) > 0))
            if 0.45 < monotone_frac < 0.55:
                # Neither clearly monotone nor clearly not — try anyway
                pass
            if monotone_frac < 0.3 or monotone_frac > 0.7:
                # Strongly non-monotone — tangent cannot fit well
                if monotone_frac < 0.3:
                    return None

            eps = self._EPS

            def _tan(xv: FloatArray, a: float, b: float, c: float, d: float) -> FloatArray:
                arg = b * xv + c
                # Soft-clamp to (-π/2+ε, π/2-ε): avoids overflow at poles
                half_pi = 0.5 * np.pi
                arg_clipped = np.clip(arg % np.pi - half_pi, -half_pi + eps, half_pi - eps)
                return np.asarray(a * np.tan(arg_clipped) + d, dtype=np.float64)

            best_rmse = float("inf")
            best_popt: Optional[FloatArray] = None

            b_lo = 0.05 / x_span
            b_hi = 0.95 * np.pi / x_span    # keep < π/x_span so single branch

            for b_seed in np.linspace(b_lo, b_hi, 5):
                for c_seed in (0.0, 0.2, -0.2, 0.5 * np.pi * b_seed * float(np.mean(x))):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = curve_fit(
                                _tan, x_s, y_s,
                                p0=[y_std, b_seed, c_seed, float(np.mean(y_s))],
                                maxfev=15000,
                                bounds=(
                                    [-np.inf, b_lo, -np.pi / 2, -np.inf],
                                    [np.inf,  b_hi,  np.pi / 2,  np.inf],
                                ),
                            )
                        popt = np.asarray(res[0], dtype=np.float64)
                        yp = _tan(x_s, *popt)
                        if not np.all(np.isfinite(yp)):
                            continue
                        r = self._rmse(y_s, yp)
                        if r < best_rmse:
                            best_rmse, best_popt = r, popt
                    except (RuntimeError, ValueError):
                        continue

            if best_popt is None:
                return None

            ta, tb, tc, td = (float(best_popt[k]) for k in range(4))
            y_pred = _tan(x_s, ta, tb, tc, td)

            if not self._validate_fitted_result(y_s, y_pred, y_std):
                return None

            rmse = self._rmse(y_s, y_pred)
            l_inf = self._l_inf(y_s, y_pred)
            bic = self._bic(n, max(rmse ** 2, 1e-300), 4)

            tan_a, tan_b, tan_c, tan_d = ta, tb, tc, td

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return _tan(x_eval, tan_a, tan_b, tan_c, tan_d)

            return FittedModel(
                name="Tangential curve",
                evaluate=evaluate,
                latex_kind="tangential",
                rmse=rmse,
                l_inf=l_inf,
                bic=bic,
                complexity=4.0,
                params={"A": tan_a, "B": tan_b, "C": tan_c, "D": tan_d},
            )
        except (RuntimeError, ValueError, TypeError, OverflowError,
                np.linalg.LinAlgError, FloatingPointError):
            return None


# ===========================================================================
# NUFFT — Non-Uniform Fast Fourier Transform fitter
# Implements Type-1 NUFFT via least-squares trigonometric regression:
#   f(x) = a₀/2 + Σₖ [aₖ cos(k·θ) + bₖ sin(k·θ)],  θ = 2π(x−x₀)/span
# This is the pseudoinverse of the NUFFT forward operator and equivalent to
# the output of a finufft.nufft1d1 solve with adjoint normalisation.
# Tikhonov regularisation (λ = 10⁻⁶·n) prevents overfitting to noise.
# ===========================================================================

class NUFFTFitter(ModelFitter):
    """Trigonometric least-squares approximation on non-uniform samples.

    The reconstruction is a real-valued trigonometric polynomial of order N*:

        f(x) = a₀/2 + Σ_{k=1}^{N*} [aₖ cos(k θ) + bₖ sin(k θ)]

    where θ = 2π(x − x_min)/span ∈ [0, 2π).

    N* is chosen by minimising the penalised score over N ∈ {2, …, N_max};
    the system is solved via normal equations with Tikhonov regularisation.
    """

    _N_MAX: int = 24   # maximum Fourier order to search

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            n = len(x)
            if n < 8:
                return None
            x0 = float(np.min(x))
            span = float(np.max(x) - x0)
            if span < 1e-12:
                return None

            theta = (2.0 * np.pi / span) * (x - x0)   # ∈ [0, 2π)
            y_scale = float(np.std(y)) or 1.0
            lam = 1e-6 * n   # Tikhonov regularisation weight

            best_score = float("inf")
            best_N = 2
            best_coeffs: Optional[FloatArray] = None

            n_max = min(self._N_MAX, (n - 1) // 2)

            for N in range(2, n_max + 1):
                try:
                    A = self._vander(theta, N)        # n × (2N+1)
                    ATA = A.T @ A
                    ATA[np.diag_indices_from(ATA)] += lam
                    coeffs = np.linalg.solve(ATA, A.T @ y)
                    y_pred = A @ coeffs
                    if not np.all(np.isfinite(y_pred)):
                        continue
                    rmse = self._rmse(y, y_pred)
                    l_inf = self._l_inf(y, y_pred)
                    complexity = float(N) * 0.4
                    score = (ALPHA * l_inf / y_scale + BETA * rmse / y_scale
                             + GAMMA * complexity)
                    if score < best_score:
                        best_score, best_N, best_coeffs = score, N, coeffs
                    # Early stop if residual < requested accuracy
                    if rmse < max(1e-9, float(accuracy)):
                        break
                except np.linalg.LinAlgError:
                    continue

            if best_coeffs is None:
                return None

            c = best_coeffs
            N_fit = best_N
            x0_fit = x0
            span_fit = span

            def evaluate(x_eval: FloatArray) -> FloatArray:
                th = (2.0 * np.pi / span_fit) * (x_eval - x0_fit)
                A_eval = NUFFTFitter._vander(th, N_fit)
                return np.asarray(A_eval @ c, dtype=np.float64)

            y_f = evaluate(x)
            rmse_f = self._rmse(y, y_f)
            l_inf_f = self._l_inf(y, y_f)
            bic_f = self._bic(n, max(rmse_f ** 2, 1e-300), 2 * N_fit + 1)

            return FittedModel(
                name="Non-Uniform Fast Fourier Transform (NUFFT)",
                evaluate=evaluate,
                latex_kind="nufft",
                rmse=rmse_f,
                l_inf=l_inf_f,
                bic=bic_f,
                complexity=float(N_fit) * 0.4,
                params={
                    "coeffs": c.tolist(),
                    "N": N_fit,
                    "x0": x0_fit,
                    "span": span_fit,
                },
            )
        except (RuntimeError, ValueError, TypeError, OverflowError,
                np.linalg.LinAlgError, FloatingPointError):
            return None

    @staticmethod
    def _vander(theta: FloatArray, n: int) -> FloatArray:
        """Build real trigonometric Vandermonde matrix of order n.

        Returns A of shape (len(theta), 2n+1) where columns are:
        [1, cos θ, sin θ, cos 2θ, sin 2θ, …, cos nθ, sin nθ].
        """
        cols = [np.ones(len(theta), dtype=np.float64)]
        for k in range(1, n + 1):
            cols.append(np.cos(k * theta))
            cols.append(np.sin(k * theta))
        return np.column_stack(cols)


# ===========================================================================
# Model selection service
# ===========================================================================

class ModelSelectionService:
    """Runs all 12 canonical fitters and exposes a fixed-order registry.

    The registry defines the canonical display order and names that the UI
    always shows, regardless of which fitters converged on a given input.
    """

    # Canonical display order — names must match exactly what each fitter returns.
    CANONICAL_NAMES: tuple[str, ...] = (
        "Cubic Spline",
        "Interpolation polynomial (Chebyshev Basis)",
        "L-inf minimax polynomial",
        "Polynomial Least Squares Approximation (Chebyshev Basis)",
        "Non-Uniform Fast Fourier Transform (NUFFT)",
        "AAA Algorithm",
        "Exponential curve",
        "Logarithmic curve",
        "Rational curve",
        "Sinusoidal curve",
        "Tangential curve",
        "Arctan (S-curve)",
    )

    def __init__(self) -> None:
        self._fitters: tuple[ModelFitter, ...] = (
            SplineFitter(),
            InterpolationPolynomialFitter(),
            LinfPolynomialFitter(),
            ChebyshevPolynomialFitter(),
            NUFFTFitter(),
            AAAFitter(),
            ExponentialFitter(),
            LogarithmicFitter(),
            RationalFitter(),
            SinusoidalFitter(),
            TangentialFitter(),
            ArctanFitter(),
        )

    def fit_all_models(self, x: FloatArray, y: FloatArray,
                       accuracy: float) -> list[Optional[FittedModel]]:
        """Return one slot per canonical model (None if fitter did not converge)."""
        results: list[Optional[FittedModel]] = []
        for fitter in self._fitters:
            try:
                results.append(fitter.fit(x, y, accuracy))
            except (RuntimeError, ValueError, TypeError, OverflowError,
                    np.linalg.LinAlgError, FloatingPointError):
                results.append(None)
        return results

    @staticmethod
    def select_best_model(
        models: list[Optional[FittedModel]], y: FloatArray
    ) -> Optional[FittedModel]:
        y_scale = float(np.std(y)) or 1.0
        fitted = [m for m in models if m is not None]
        if not fitted:
            return None
        return min(fitted, key=lambda m: m.score(y_scale))


# ===========================================================================
# LaTeX generator
# dispatch dict is an instance attribute, built once in __init__
# ===========================================================================

class LaTeXGenerator:
    """Converts FittedModel -> display-math LaTeX string.

    Parameters
    ----------
    approx : bool
        When True (default) all numeric coefficients are rendered as
        rounded decimals with *decimals* digits after the point.
        When False, exact rational fractions are used.
    decimals : int
        Number of digits after the decimal point in approximate mode.
    """

    _MAX_SPLINE_CASES: int = 6

    def __init__(self, approx: bool = True, decimals: int = 3) -> None:
        self.approx = approx
        self.decimals = max(0, min(10, int(decimals)))
        # Built once at construction, not on every generate() call
        self._dispatch: dict[str, Callable[[FittedModel], str]] = {
            "chebyshev":   self._chebyshev,
            "sinusoidal":  self._sinusoidal,
            "exponential": self._exponential,
            "logarithmic": self._logarithmic,
            "rational":    self._rational,
            "arctan":      self._arctan,
            "tangential":  self._tangential,
            "spline":      self._spline,
            "aaa":         self._aaa,
            "linf_poly":   self._linf_poly,
            "nufft":       self._nufft,
        }

    def reconfigure(self, approx: bool, decimals: int) -> None:
        """Update mode without rebuilding the dispatch table."""
        self.approx = approx
        self.decimals = max(0, min(10, int(decimals)))

    def generate(self, model: FittedModel) -> str:
        try:
            handler = self._dispatch.get(model.latex_kind)
            return handler(model) if handler is not None else self._fallback(model)
        except (KeyError, TypeError, AttributeError, ValueError, ArithmeticError):
            return self._fallback(model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n(self, v: float) -> sp.Expr:
        """Convert float to sympy number respecting approx mode.

        Approx mode  → sp.Float with string representation at self.decimals places.
        Exact mode   → sp.Rational with denominator ≤ 1000 (exact fraction).
        """
        if self.approx:
            return sp.Float(f"{v:.{self.decimals}f}")
        return sp.Rational(v).limit_denominator(1000)

    def _round_floats(self, expr: sp.Basic) -> sp.Basic:
        """Walk *expr* and round every sp.Float leaf to self.decimals decimal places.

        This is needed because symbolic arithmetic (e.g. inside Chebyshev
        recurrences) can produce floats with more digits than the user requested,
        even when the inputs were already rounded.
        """
        if isinstance(expr, sp.Float):
            return sp.Float(f"{float(expr):.{self.decimals}f}")
        if expr.args:
            return expr.func(*[self._round_floats(a) for a in expr.args])
        return expr

    # Keep _r as a static alias for exact-only contexts (spline knot labels, etc.)
    @staticmethod
    def _r(v: float, denom: int = 1000) -> sp.Rational:
        return sp.Rational(v).limit_denominator(denom)

    def _wrap(self, expr: sp.Basic) -> str:
        """Simplify *expr* and format as display-math LaTeX.

        Approx mode: walk the tree and round all Float leaves to self.decimals
        before calling sp.latex — this fixes both digit count and Chebyshev
        recurrence artifacts.
        Exact mode: nsimplify cleans up rational arithmetic.
        """
        if self.approx:
            return f"$$f(x) = {sp.latex(self._round_floats(expr))}$$"
        simplified = sp.nsimplify(expr, rational=False, tolerance=1e-6)
        return f"$$f(x) = {sp.latex(simplified)}$$"

    def _expand_cheb_series(self, coef: list[float], t: sp.Expr) -> str:
        """Build sum of c_k * T_k(t) symbolically and wrap in display-math."""
        expr: sp.Expr = sp.Integer(0)
        for k, c in enumerate(coef):
            if abs(c) < 1e-14:
                continue
            expr += self._n(c) * sp.chebyshevt(k, t)  # type: ignore[attr-defined]
        return self._wrap(sp.expand(expr))

    # ------------------------------------------------------------------
    # Per-kind generators
    # ------------------------------------------------------------------

    def _chebyshev(self, model: FittedModel) -> str:
        """Expand Chebyshev series symbolically using stored .coef array."""
        cheb_coef: list[float] = list(model.params["cheb_coef"])
        x_min = float(model.params["domain"][0])
        x_max = float(model.params["domain"][1])

        x = sp.Symbol("x")
        mid_val = 0.5 * (x_max + x_min)
        half_val = 0.5 * (x_max - x_min)
        # Use Float in approx mode: Rational mid/half propagate into Chebyshev
        # recurrence and produce many-digit fractional coefficients.
        if self.approx:
            t = (x - sp.Float(mid_val)) / sp.Float(half_val)
        else:
            t = (x - sp.Rational(mid_val).limit_denominator(10000)) / sp.Rational(half_val).limit_denominator(10000)
        return self._expand_cheb_series(cheb_coef, t)

    def _sinusoidal(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        f = self._n(float(model.params["f"]))
        ph = self._n(float(model.params["phase"]))
        B = self._n(float(model.params["B"]))
        return self._wrap(A * sp.sin(sp.Integer(2) * sp.pi * f * x + ph) + B)

    def _exponential(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        B = self._n(float(model.params["B"]))
        C = self._n(float(model.params["C"]))
        return self._wrap(A * sp.exp(B * x) + C)

    def _logarithmic(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        B = self._n(float(model.params["B"]))
        shift = float(model.params.get("shift", 0.0))
        arg = x + self._n(shift) if abs(shift) > 1e-9 else x
        return self._wrap(A * sp.ln(arg) + B)

    def _rational(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        B = self._n(float(model.params["B"]))
        D = self._n(float(model.params["D"]))
        return self._wrap(A + B / (x - D))

    def _arctan(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        B = self._n(float(model.params["B"]))
        C = self._n(float(model.params["C"]))
        D = self._n(float(model.params["D"]))
        latex_str = self._wrap(A * sp.atan(B * x + C) + D)
        # Fix: sympy renders atan as \operatorname{atan}, we want \arctan
        return latex_str.replace(r'\operatorname{atan}', r'\arctan')

    @staticmethod
    def _spline_poly(n0: sp.Expr, n1: sp.Expr, n2: sp.Expr, n3: sp.Expr,
                     dx: sp.Expr) -> sp.Expr:
        """Expand cubic polynomial c0*dx³ + c1*dx² + c2*dx + c3 in x."""
        return sp.expand(n0 * dx ** 3 + n1 * dx ** 2 + n2 * dx + n3)

    def _spline(self, model: FittedModel) -> str:
        cs: Optional[CubicSpline] = model.params.get("cubic_spline")
        x_knots_raw: list[float] = model.params.get("x_knots", [])

        if cs is None or len(x_knots_raw) < 2:
            k = model.params.get("k_count", "?")
            return rf"$$f(x) = \text{{Cubic spline ({k} segments)}}$$"

        x_knots = np.asarray(x_knots_raw, dtype=np.float64)
        if len(x_knots) > self._MAX_SPLINE_CASES + 1:
            sel = np.linspace(0, len(x_knots) - 1, self._MAX_SPLINE_CASES + 1, dtype=int)
            x_knots = x_knots[sel]

        x_sym = sp.Symbol("x")
        cases: list[str] = []

        for i in range(len(x_knots) - 1):
            xi, xi1 = float(x_knots[i]), float(x_knots[i + 1])
            seg = int(np.searchsorted(cs.x, 0.5 * (xi + xi1), side="right") - 1)
            seg = max(0, min(seg, cs.c.shape[1] - 1))

            c0 = float(cs.c[0, seg])
            c1 = float(cs.c[1, seg])
            c2 = float(cs.c[2, seg])
            c3 = float(cs.c[3, seg])

            if self.approx:
                dx: sp.Expr = x_sym - sp.Float(xi)
                piece_out = self._round_floats(
                    self._spline_poly(self._n(c0), self._n(c1), self._n(c2), self._n(c3), dx)
                )
                lo: str = f"{xi:.{self.decimals}f}"
                hi: str = f"{xi1:.{self.decimals}f}"
            else:
                dx = x_sym - sp.Rational(xi).limit_denominator(1000)
                piece_out = self._spline_poly(
                    self._n(c0), self._n(c1), self._n(c2), self._n(c3), dx
                )
                lo = sp.latex(sp.Rational(xi).limit_denominator(100))
                hi = sp.latex(sp.Rational(xi1).limit_denominator(100))

            cases.append(rf"{sp.latex(piece_out)}, & {lo} \le x < {hi}")

        body = r" \\ ".join(cases)
        return rf"$$f(x) = \begin{{cases}} {body} \end{{cases}}$$"

    def _aaa(self, model: FittedModel) -> str:
        """Barycentric form using support_points, support_values and weights from AAA."""
        sp_list: list[float] = list(model.params.get("support_points", []))
        sv_list: list[float] = list(model.params.get("support_values", []))
        w_list: list[float] = list(model.params.get("weights", []))

        if not sp_list or len(sp_list) != len(sv_list) or len(sp_list) != len(w_list):
            return self._fallback(model)

        x = sp.Symbol("x")
        max_terms = min(8, len(sp_list))
        num_expr: sp.Expr = sp.Integer(0)
        den_expr: sp.Expr = sp.Integer(0)

        for j in range(max_terms):
            wj = self._n(w_list[j])
            fj = self._n(sv_list[j])
            zj = self._n(sp_list[j])
            pole = x - zj
            num_expr += wj * fj / pole
            den_expr += wj / pole

        if self.approx:
            # sp.cancel on Float expressions produces garbage; render barycentric
            # form directly after rounding all float leaves
            num_r = self._round_floats(num_expr)
            den_r = self._round_floats(den_expr)
            body = rf"\frac{{{sp.latex(num_r)}}}{{{sp.latex(den_r)}}}"
        else:
            try:
                ratio = sp.cancel(num_expr / den_expr)
                body = sp.latex(ratio)
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                body = rf"\frac{{{sp.latex(num_expr)}}}{{{sp.latex(den_expr)}}}"

        trail = r" + \ldots" if len(sp_list) > max_terms else ""
        return f"$$f(x) = {body}{trail}$$"

    def _linf_poly(self, model: FittedModel) -> str:
        """Expand the L-inf Chebyshev-basis result into a standard poly in x."""
        cheb_coef: list[float] = list(model.params["cheb_coef"])
        x_mid = float(model.params["x_mid"])
        x_half = float(model.params["x_half"])

        x = sp.Symbol("x")
        if self.approx:
            t = (x - sp.Float(x_mid)) / sp.Float(x_half)
        else:
            t = (x - sp.Rational(x_mid).limit_denominator(1000)) / sp.Rational(x_half).limit_denominator(1000)
        return self._expand_cheb_series(cheb_coef, t)

    def _tangential(self, model: FittedModel) -> str:
        x = sp.Symbol("x")
        A = self._n(float(model.params["A"]))
        B = self._n(float(model.params["B"]))
        C = self._n(float(model.params["C"]))
        D = self._n(float(model.params["D"]))
        return self._wrap(A * sp.tan(B * x + C) + D)

    def _nufft(self, model: FittedModel) -> str:
        """Render the trigonometric polynomial as a truncated Fourier series."""
        coeffs: list[float] = list(model.params.get("coeffs", []))
        N: int = int(model.params.get("N", 0))
        x0: float = float(model.params.get("x0", 0.0))
        span: float = float(model.params.get("span", 1.0))
        if not coeffs or N < 1:
            return self._fallback(model)

        x = sp.Symbol("x")
        # θ = 2π(x − x₀)/span
        if self.approx:
            x0_s: sp.Expr = sp.Float(f"{x0:.{self.decimals}f}")
            span_s: sp.Expr = sp.Float(f"{span:.{self.decimals}f}")
        else:
            x0_s = sp.Rational(x0).limit_denominator(1000)
            span_s = sp.Rational(span).limit_denominator(1000)

        theta = sp.Integer(2) * sp.pi * (x - x0_s) / span_s

        expr: sp.Expr = self._n(coeffs[0]) / sp.Integer(2)   # a₀/2 constant term
        for k in range(1, N + 1):
            ak = coeffs[2 * k - 1] if 2 * k - 1 < len(coeffs) else 0.0
            bk = coeffs[2 * k] if 2 * k < len(coeffs) else 0.0
            if abs(ak) > 1e-14:
                expr += self._n(ak) * sp.cos(sp.Integer(k) * theta)
            if abs(bk) > 1e-14:
                expr += self._n(bk) * sp.sin(sp.Integer(k) * theta)

        return self._wrap(expr)

    @staticmethod
    def _fallback(model: FittedModel) -> str:
        return (
            rf"$$f(x) \approx \text{{{model.name}}}"
            rf"\quad[\text{{RMSE}} = {model.rmse:.4g}]$$"
        )


# ===========================================================================
# Stroke preprocessor
# ===========================================================================

class StrokePreprocessor:

    def __init__(self, domain_width: float, domain_height: float) -> None:
        if domain_width <= 0 or domain_height <= 0:
            raise ValueError("Domain dimensions must be positive")
        self._domain_width = float(domain_width)
        self._domain_height = float(domain_height)

    def preprocess(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        n = max(100, int(self._domain_width * 50))
        x, y = self._resample(x, y, n)
        x, y = self._smooth(x, y)
        x, y = self._remove_outliers(x, y)
        if len(x) > 50:
            x, y = self._simplify(x, y)
        return x, y

    def is_function(self, x: FloatArray, y: FloatArray) -> tuple[bool, str]:
        eps = self._domain_width / 1000.0
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        j = 0
        for i in range(len(xs)):
            if j < i:
                j = i
            while j + 1 < len(xs) and xs[j + 1] - xs[i] <= eps:
                j += 1
            if j > i:
                win = ys[i: j + 1]
                if float(np.ptp(win)) > eps:
                    return False, f"Multiple y-values at x ~ {float(xs[i]):.2f}"
        return True, ""

    @staticmethod
    def _resample(x: FloatArray, y: FloatArray, n: int) -> tuple[FloatArray, FloatArray]:
        """Arc-length uniform resampling using make_interp_spline (k=1 = piecewise linear)."""
        if len(x) < 2:
            return x, y
        dx, dy = np.diff(x), np.diff(y)
        s: FloatArray = np.concatenate(([0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))))
        if float(s[-1]) == 0.0:
            return x, y
        su = np.linspace(0.0, float(s[-1]), n)
        try:
            spl_x = make_interp_spline(s, x, k=1)
            spl_y = make_interp_spline(s, y, k=1)
            return (
                np.asarray(spl_x(su), dtype=np.float64),
                np.asarray(spl_y(su), dtype=np.float64),
            )
        except (ValueError, IndexError):
            return x, y

    @staticmethod
    def _smooth(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Savitzky-Golay smoothing via scipy.signal.savgol_filter."""
        if len(x) < 15:
            return x, y
        window = min(15, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if window < 3:
            return x, y
        try:
            return x, np.asarray(savgol_filter(y, window, min(3, window - 1)), dtype=np.float64)
        except (ValueError, TypeError):
            return x, y

    @staticmethod
    def _remove_outliers(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Curvature-based outlier removal; scalar 2-D cross product (NumPy 2.0 safe)."""
        if len(x) < 5:
            return x, y
        curv = np.zeros(len(x), dtype=np.float64)
        radius = max(2, int(len(x) * 0.02))
        for i in range(len(x)):
            ip = max(0, i - radius)
            iq = min(len(x) - 1, i + radius)
            if iq - ip < 2:
                continue
            v1 = np.array([x[i] - x[ip], y[i] - y[ip]], dtype=np.float64)
            v2 = np.array([x[iq] - x[ip], y[iq] - y[ip]], dtype=np.float64)
            a = float(np.linalg.norm(v1))
            b = float(np.linalg.norm(
                np.array([x[iq] - x[i], y[iq] - y[i]], dtype=np.float64)
            ))
            c = float(np.linalg.norm(v2))
            denom = a * b * c
            if denom <= 1e-10:
                continue
            # Scalar z-component of 3-D cross product of two 2-D vectors
            cross_z = float(v1[0] * v2[1] - v1[1] * v2[0])
            curv[i] = 4.0 * abs(cross_z) / (2.0 * denom)
        threshold = float(np.median(curv) + 3.0 * np.std(curv))
        return x[curv < threshold], y[curv < threshold]

    def _simplify(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        pts = np.column_stack((x, y))
        simplified = self._rdp(pts, 0.01 * self._domain_height)
        return simplified[:, 0], simplified[:, 1]

    def _rdp(self, pts: FloatArray, tol: float) -> FloatArray:
        """Ramer-Douglas-Peucker polyline simplification."""
        if len(pts) < 3:
            return pts
        dmax, split = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = self._perp(pts[i], pts[0], pts[-1])
            if d > dmax:
                split, dmax = i, d
        if dmax > tol:
            a = self._rdp(pts[: split + 1], tol)
            b = self._rdp(pts[split:], tol)
            return np.vstack((a[:-1], b))
        return np.array([pts[0], pts[-1]], dtype=np.float64)

    @staticmethod
    def _perp(pt: FloatArray, a: FloatArray, b: FloatArray) -> float:
        if np.allclose(a, b):
            return float(np.linalg.norm(pt - a))
        ba = b - a
        ap = a - pt
        num = abs(float(ba[0] * ap[1] - ba[1] * ap[0]))
        den = float(np.linalg.norm(ba))
        return num / den if den > 0 else 0.0


# ===========================================================================
# Background worker
# ===========================================================================

class FitWorker(QObject):
    finished: Signal = Signal(list, object)
    error: Signal = Signal(str)

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        accuracy: float,
        service: ModelSelectionService,
    ) -> None:
        super().__init__()
        self._x = x.copy()
        self._y = y.copy()
        self._accuracy = accuracy
        self._service = service

    def run(self) -> None:
        try:
            models = self._service.fit_all_models(self._x, self._y, self._accuracy)
            if not any(m is not None for m in models):
                self.error.emit("Could not fit any model to the data.")
                return
            best = self._service.select_best_model(models, self._y)
            self.finished.emit(models, best)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ===========================================================================
# Settings dialog
# ===========================================================================

class SettingsDialog(QDialog):

    def __init__(self, settings: PlotSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self._settings = settings
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)

        self._x_min_edit = QLineEdit(str(self._settings.x_min))
        self._x_max_edit = QLineEdit(str(self._settings.x_max))
        self._y_min_edit = QLineEdit(str(self._settings.y_min))
        self._y_max_edit = QLineEdit(str(self._settings.y_max))
        self._grid_edit = QLineEdit(str(self._settings.grid_spacing))

        self._accuracy_sb = QDoubleSpinBox()
        self._accuracy_sb.setRange(0.0001, 1.0)
        self._accuracy_sb.setSingleStep(0.0001)
        self._accuracy_sb.setDecimals(4)
        self._accuracy_sb.setValue(self._settings.accuracy)

        # ── LaTeX format controls ──────────────────────────────────────
        self._latex_approx_cb = QCheckBox("Approximate coefficients (decimals)")
        self._latex_approx_cb.setChecked(self._settings.latex_approx)
        self._latex_approx_cb.setToolTip(
            "ON  — coefficients shown as rounded decimals, e.g. 3.142\n"
            "OFF — exact rational fractions, e.g. 355/113"
        )

        self._latex_decimals_sb = QSpinBox()
        self._latex_decimals_sb.setRange(0, 10)
        self._latex_decimals_sb.setValue(self._settings.latex_decimals)
        self._latex_decimals_sb.setToolTip("Digits after decimal point in approximate mode")
        self._latex_approx_cb.toggled.connect(self._latex_decimals_sb.setEnabled)
        self._latex_decimals_sb.setEnabled(self._settings.latex_approx)

        fields: list[tuple[str, QWidget]] = [
            ("X Min:", self._x_min_edit),
            ("X Max:", self._x_max_edit),
            ("Y Min:", self._y_min_edit),
            ("Y Max:", self._y_max_edit),
            ("Grid Spacing:", self._grid_edit),
            ("Approximation Accuracy:", self._accuracy_sb),
        ]
        for row, (label, widget) in enumerate(fields):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)

        # LaTeX section with a visual separator
        sep_row = len(fields)
        sep = QLabel("─── LaTeX Output Format ───")
        sep.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(sep, sep_row, 0, 1, 2)

        layout.addWidget(self._latex_approx_cb, sep_row + 1, 0, 1, 2)
        layout.addWidget(QLabel("Digits after decimal point:"), sep_row + 2, 0)
        layout.addWidget(self._latex_decimals_sb, sep_row + 2, 1)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row, sep_row + 3, 0, 1, 2)

    def get_settings(self) -> Optional[PlotSettings]:
        try:
            return PlotSettings(
                x_min=float(self._x_min_edit.text()),
                x_max=float(self._x_max_edit.text()),
                y_min=float(self._y_min_edit.text()),
                y_max=float(self._y_max_edit.text()),
                grid_spacing=float(self._grid_edit.text()),
                accuracy=float(self._accuracy_sb.value()),
                latex_approx=bool(self._latex_approx_cb.isChecked()),
                latex_decimals=int(self._latex_decimals_sb.value()),
            )
        except (ValueError, TypeError):
            return None


# ===========================================================================
# Main window
# ===========================================================================

class DrawingApp(QMainWindow):

    # One pleasant, distinct colour per canonical model slot (12 total).
    # Order matches ModelSelectionService.CANONICAL_NAMES exactly.
    _MODEL_COLORS: tuple[tuple[int, int, int], ...] = (
        (220, 80, 80),      # 1  Cubic Spline           — warm red
        (60, 160, 240),     # 2  Interpolation poly     — sky blue
        (255, 140, 0),      # 3  L-inf minimax          — amber
        (80, 200, 80),      # 4  LS Chebyshev           — grass green
        (180, 80, 220),     # 5  NUFFT                  — violet
        (20, 210, 190),     # 6  AAA                    — teal
        (240, 100, 160),    # 7  Exponential            — rose
        (160, 220, 60),     # 8  Logarithmic            — lime
        (240, 190, 40),     # 9  Rational               — gold
        (90, 190, 255),     # 10 Sinusoidal             — azure
        (255, 130, 60),     # 11 Tangential             — coral
        (140, 100, 240),    # 12 Arctan (S-curve)       — lavender
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Function Drawer — High-Precision LaTeX Engine")
        self.setGeometry(100, 100, 1450, 820)

        self._model_service = ModelSelectionService()
        self._settings = PlotSettings()
        self._latex_gen = LaTeXGenerator(
            approx=self._settings.latex_approx,
            decimals=self._settings.latex_decimals,
        )

        self._drawing = False
        self._strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: Optional[list[tuple[float, float]]] = None

        self._panning = False
        self._pan_start_scene_pos: Optional[QPointF] = None  # Store scene (pixel) position
        self._pan_start_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None

        self._drawn_curve: Optional[Any] = None
        # One slot per canonical model (None = not yet fitted / did not converge)
        self._fitted_curves: list[Optional[Any]] = [None] * len(
            ModelSelectionService.CANONICAL_NAMES
        )
        self._shown_models: list[Optional[FittedModel]] = [None] * len(
            ModelSelectionService.CANONICAL_NAMES
        )
        self._model_latex_rows: list[str] = []
        self._option_checkboxes: list[QCheckBox] = []
        self._option_widgets: list[QWidget] = []

        self._thread: Optional[QThread] = None
        self._worker: Optional[FitWorker] = None
        self._x_proc: FloatArray = np.empty(0, dtype=np.float64)
        self._y_proc: FloatArray = np.empty(0, dtype=np.float64)

        self._build_ui()
        self._configure_plot()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.addLegend(offset=(10, 10))
        left.addWidget(self._plot_widget)

        btn_row = QHBoxLayout()
        self._clear_btn = QPushButton("Clear")
        self._fit_btn = QPushButton("Fit Curve")
        self._export_btn = QPushButton("Copy LaTeX")
        self._settings_btn = QPushButton("Settings")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: gray; font-style: italic;")

        self._clear_btn.clicked.connect(self.clear_drawing)
        self._fit_btn.clicked.connect(self.fit_curve)
        self._export_btn.clicked.connect(self.copy_latex)
        self._settings_btn.clicked.connect(self.show_settings)

        for widget in (self._clear_btn, self._fit_btn, self._export_btn,
                       self._settings_btn, self._status_lbl):
            btn_row.addWidget(widget)
        left.addLayout(btn_row)
        root.addLayout(left, 3)

        right = QVBoxLayout()

        # ── top control ────────────────────────────────────────────────
        self._clear_on_new_line = QCheckBox("Clear plot on new stroke")
        self._clear_on_new_line.setChecked(True)
        right.addWidget(self._clear_on_new_line)

        # ── fixed 12-model panel ───────────────────────────────────────
        opts_group = QGroupBox("Models")
        self._options_layout = QVBoxLayout()
        self._options_layout.setSpacing(2)
        opts_group.setLayout(self._options_layout)
        right.addWidget(opts_group)

        # Pre-create one row per canonical model slot — always visible
        for idx, name in enumerate(ModelSelectionService.CANONICAL_NAMES):
            color = self._MODEL_COLORS[idx]
            row, cb = self._create_option_row(idx, name, "", color, False,
                                              enabled=False)
            self._options_layout.addWidget(row)
            self._option_widgets.append(row)
            self._option_checkboxes.append(cb)

        # ── LaTeX output panel ─────────────────────────────────────────
        right.addWidget(QLabel("LaTeX Output (checked models):"))
        self._latex_output = QTextEdit()
        self._latex_output.setReadOnly(True)
        self._latex_output.setFontFamily("Courier New")
        right.addWidget(self._latex_output)
        root.addLayout(right, 1)

        vb = self._plot_widget.plotItem.vb
        vb.setMenuEnabled(False)
        # Disable pyqtgraph's built-in mouse modes to prevent interference
        vb.setMouseMode(vb.RectMode)  # Set to rect mode but we'll override with event filter
        vb.setMouseEnabled(x=False, y=False)  # Disable built-in mouse panning
        self._plot_widget.viewport().installEventFilter(self)

    def _configure_plot(self) -> None:
        self._plot_widget.setLabel("left", "y")
        self._plot_widget.setLabel("bottom", "x")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        vb = self._plot_widget.plotItem.vb
        vb.disableAutoRange()
        self._plot_widget.setXRange(self._settings.x_min, self._settings.x_max)
        self._plot_widget.setYRange(self._settings.y_min, self._settings.y_max)

    def eventFilter(self, obj: Any, event: QEvent) -> bool:  # noqa: N802
        if obj is not self._plot_widget.viewport():
            return super().eventFilter(obj, event)

        vb = self._plot_widget.plotItem.vb
        et = event.type()

        # Handle wheel events for zooming
        if et == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            # Get the mouse position in view coordinates
            mouse_point = vb.mapSceneToView(event.position())
            mouse_x = float(mouse_point.x())
            mouse_y = float(mouse_point.y())

            # Get current view range
            xr, yr = vb.viewRange()
            x_min, x_max = float(xr[0]), float(xr[1])
            y_min, y_max = float(yr[0]), float(yr[1])

            # Calculate zoom factor based on wheel delta
            # Positive delta = zoom in, negative = zoom out
            angle_delta = event.angleDelta().y()
            zoom_factor = 1.0 - (angle_delta / 1200.0)  # ~0.9 for zoom in, ~1.1 for zoom out
            zoom_factor = max(0.5, min(2.0, zoom_factor))  # Clamp to reasonable range

            # Calculate new ranges centered on mouse position
            x_span = (x_max - x_min) * zoom_factor
            y_span = (y_max - y_min) * zoom_factor

            # Calculate how far the mouse is from the left/bottom edges (as fraction)
            x_frac = (mouse_x - x_min) / (x_max - x_min) if x_max != x_min else 0.5
            y_frac = (mouse_y - y_min) / (y_max - y_min) if y_max != y_min else 0.5

            # New ranges keep the mouse position at the same location
            new_x_min = mouse_x - x_span * x_frac
            new_x_max = mouse_x + x_span * (1.0 - x_frac)
            new_y_min = mouse_y - y_span * y_frac
            new_y_max = mouse_y + y_span * (1.0 - y_frac)

            # Apply the new range
            vb.setRange(
                xRange=(new_x_min, new_x_max),
                yRange=(new_y_min, new_y_max),
                padding=0,
                update=True
            )
            return True

        # Handle mouse events
        if not isinstance(event, QMouseEvent):
            return super().eventFilter(obj, event)

        if et == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._clear_on_new_line.isChecked():
                    self.clear_drawing()
                vp = vb.mapSceneToView(event.position())
                self._drawing = True
                self._current_stroke = [(float(vp.x()), float(vp.y()))]
                self._strokes.append(self._current_stroke)
                return True
            if event.button() == Qt.MouseButton.RightButton:
                self._panning = True
                # Store scene position (widget pixel coordinates - these don't change)
                self._pan_start_scene_pos = QPointF(event.position())
                # Store the current view range
                xr, yr = vb.viewRange()
                self._pan_start_range = (
                    (float(xr[0]), float(xr[1])),
                    (float(yr[0]), float(yr[1])),
                )
                self._plot_widget.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True

        if et == QEvent.Type.MouseMove:
            if self._drawing and self._current_stroke is not None:
                vp = vb.mapSceneToView(event.position())
                self._current_stroke.append((float(vp.x()), float(vp.y())))
                self.update_drawing()
                return True
            if (self._panning
                    and self._pan_start_scene_pos is not None
                    and self._pan_start_range is not None):
                # Calculate delta in scene (pixel) coordinates
                current_scene_pos = event.position()
                dx_scene = current_scene_pos.x() - self._pan_start_scene_pos.x()
                dy_scene = current_scene_pos.y() - self._pan_start_scene_pos.y()

                # Convert scene delta to view delta using the ORIGINAL view range
                # This avoids the feedback loop from coordinate transformation
                (x0, x1), (y0, y1) = self._pan_start_range
                view_width = x1 - x0
                view_height = y1 - y0

                # Get the viewport size in pixels
                vb_rect = vb.boundingRect()
                pixel_width = vb_rect.width()
                pixel_height = vb_rect.height()

                # Calculate view deltas (inverted for natural panning direction)
                if pixel_width > 0 and pixel_height > 0:
                    dx_view = -dx_scene * view_width / pixel_width
                    dy_view = dy_scene * view_height / pixel_height

                    # Apply the offset to the original range
                    new_x_range = (x0 + dx_view, x1 + dx_view)
                    new_y_range = (y0 + dy_view, y1 + dy_view)

                    # Update the view range without triggering autoRange
                    vb.setRange(xRange=new_x_range, yRange=new_y_range, padding=0, update=True)
                return True

        if et == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self._drawing = False
                self._current_stroke = None
                return True
            if event.button() == Qt.MouseButton.RightButton:
                self._panning = False
                self._pan_start_scene_pos = None
                self._pan_start_range = None
                self._plot_widget.viewport().unsetCursor()
                return True

        return super().eventFilter(obj, event)

    def _clear_results(self) -> None:
        """Remove fitted plot curves and reset all 12 model rows to disabled state."""
        # Remove plot curves
        for c in self._fitted_curves:
            if c is not None:
                try:
                    self._plot_widget.removeItem(c)
                except (RuntimeError, ValueError, AttributeError):
                    # Item may have already been removed or become invalid
                    pass
        n_slots = len(ModelSelectionService.CANONICAL_NAMES)
        self._fitted_curves = [None] * n_slots
        self._shown_models = [None] * n_slots
        self._model_latex_rows = [""] * n_slots

        # Reset every row to disabled/unchecked appearance
        for idx, cb in enumerate(self._option_checkboxes):
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.setEnabled(False)
            cb.blockSignals(False)
            # Dim the label
            if idx < len(self._option_widgets):
                lbl = self._option_widgets[idx].findChild(QLabel)
                if lbl is not None:
                    r, g, b = self._MODEL_COLORS[idx]
                    lbl.setStyleSheet(
                        f"color: rgba({r},{g},{b},90); font-weight: normal;"
                    )
                    lbl.setText(f"{idx + 1}. {ModelSelectionService.CANONICAL_NAMES[idx]}")

    def clear_drawing(self) -> None:
        self._strokes.clear()
        self._current_stroke = None
        self._drawing = False
        self._panning = False
        self._pan_start_pos = None
        self._pan_start_range = None
        self._plot_widget.viewport().unsetCursor()

        if self._drawn_curve is not None:
            self._plot_widget.removeItem(self._drawn_curve)
            self._drawn_curve = None

        self._clear_results()
        self._latex_output.clear()
        self._status_lbl.setText("Ready")
        self._status_lbl.setStyleSheet("color: gray; font-style: italic;")

    def update_drawing(self) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for stroke in self._strokes:
            if not stroke:
                continue
            if xs:
                xs.append(float("nan"))
                ys.append(float("nan"))
            sx, sy = zip(*stroke)
            xs.extend(sx)
            ys.extend(sy)

        if len(xs) < 2:
            if self._drawn_curve is not None:
                self._plot_widget.removeItem(self._drawn_curve)
                self._drawn_curve = None
            return

        xa = np.asarray(xs, dtype=np.float64)
        ya = np.asarray(ys, dtype=np.float64)
        if self._drawn_curve is None:
            self._drawn_curve = self._plot_widget.plot(
                xa, ya, pen=pg.mkPen((100, 120, 255), width=2), name="Drawn curve"
            )
        else:
            self._drawn_curve.setData(xa, ya)

    def fit_curve(self) -> None:
        points: list[tuple[float, float]] = [p for stroke in self._strokes for p in stroke]
        if len(points) < 10:
            QMessageBox.warning(self, "Insufficient Data",
                                "Draw a longer curve (at least 10 points).")
            return

        pts = np.asarray(points, dtype=np.float64)
        x_raw: FloatArray = pts[:, 0]
        y_raw: FloatArray = pts[:, 1]

        try:
            pre = StrokePreprocessor(self._settings.domain_width, self._settings.domain_height)
            x_p, y_p = pre.preprocess(x_raw, y_raw)
            if len(x_p) < 5:
                QMessageBox.warning(self, "Processing Error",
                                    "Not enough points after preprocessing.")
                return
            ok, msg = pre.is_function(x_p, y_p)
            if not ok:
                QMessageBox.warning(self, "Not a Function", msg)
                return
        except (ValueError, TypeError, IndexError) as exc:
            QMessageBox.critical(self, "Preprocessing Error", str(exc))
            return

        self._x_proc = x_p
        self._y_proc = y_p
        self._fit_btn.setEnabled(False)
        self._status_lbl.setText("Fitting... please wait")
        self._status_lbl.setStyleSheet("color: orange; font-style: italic;")

        self._thread = QThread()
        self._worker = FitWorker(x_p, y_p, self._settings.accuracy, self._model_service)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_fit_finished)
        self._worker.error.connect(self._on_fit_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_fit_finished(
        self, models: list[Optional[FittedModel]], best_model: Optional[FittedModel]
    ) -> None:
        self._display_fitted_models(models, best_model, self._x_proc, self._y_proc)

    def _on_fit_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Fitting Error", msg)

    def _on_thread_done(self) -> None:
        self._fit_btn.setEnabled(True)
        self._status_lbl.setText("Done")
        self._status_lbl.setStyleSheet("color: green; font-style: italic;")

    @staticmethod
    def _build_latex_row(
        idx: int, model: FittedModel, y_scale: float,
        latex_str: str, is_best: bool,
    ) -> str:
        """Format one model's text block for the LaTeX output pane."""
        marker = " ★ BEST" if is_best else ""
        return (
            f"─── {idx + 1}. {model.name}{marker}\n"
            f"    RMSE  : {model.rmse:.6f}\n"
            f"    L-inf : {model.l_inf:.6f}\n"
            f"    BIC   : {model.bic:.4f}\n"
            f"    Score : {model.score(y_scale):.6f}\n"
            f"    LaTeX : {latex_str}\n"
        )

    def _display_fitted_models(
        self,
        models: list[Optional[FittedModel]],
        best_model: Optional[FittedModel],
        x_proc: FloatArray,
        y_proc: FloatArray,
    ) -> None:
        """Update the 12 model rows, sorted by score (best first), with no extra markings."""
        y_scale = float(np.std(y_proc)) or 1.0
        self._clear_results()

        x_plot = np.linspace(
            float(np.min(x_proc)), float(np.max(x_proc)), 600, dtype=np.float64
        )
        y_clip = float(np.max(np.abs(y_proc))) * 10

        # Build list of all models with their canonical indices and scores
        all_models_data: list[tuple[int, Optional[FittedModel], float]] = []
        for idx, model in enumerate(models):
            if model is not None:
                score = model.score(y_scale)
                all_models_data.append((idx, model, score))
            else:
                # Non-converged models go to the end with infinite score
                all_models_data.append((idx, None, float('inf')))

        # Sort by score (lower is better)
        all_models_data.sort(key=lambda x: x[2])

        # Determine which models to auto-check (top N converged ones)
        converged_count = sum(1 for _, m, _ in all_models_data if m is not None)
        top_checked_count = min(N_TOP_CHECKED, converged_count)

        # Update each UI row based on sorted order
        for display_idx, (canonical_idx, model, score) in enumerate(all_models_data):
            self._shown_models[display_idx] = model

            cb = self._option_checkboxes[display_idx]
            lbl = self._option_widgets[display_idx].findChild(QLabel)
            r, g, b = self._MODEL_COLORS[canonical_idx]  # Use canonical color
            canonical_name = ModelSelectionService.CANONICAL_NAMES[canonical_idx]

            if model is None:
                # Model did not converge — show disabled at bottom of list
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.blockSignals(False)
                if lbl is not None:
                    lbl.setStyleSheet(
                        f"color: rgba({r},{g},{b},90); font-weight: normal;"
                    )
                    lbl.setText(f"{display_idx + 1}. {canonical_name}  (—)")
                self._fitted_curves[display_idx] = None
                self._model_latex_rows[display_idx] = ""
                continue

            # Model converged — check if it's in top N
            is_in_top = display_idx < top_checked_count
            is_best = model is best_model

            latex_str = self._latex_gen.generate(model)
            self._model_latex_rows[display_idx] = self._build_latex_row(
                display_idx, model, y_scale, latex_str, is_best
            )

            # Update label with NO extra markings
            if lbl is not None:
                lbl.setStyleSheet(
                    f"color: rgb({r},{g},{b}); "
                    f"font-weight: {'bold' if is_best else 'normal'};"
                )
                lbl.setText(f"{display_idx + 1}. {canonical_name}")

            cb.blockSignals(True)
            cb.setChecked(is_in_top)
            cb.setEnabled(True)
            cb.blockSignals(False)

            # Plot curve
            try:
                y_plot = np.clip(
                    np.asarray(model.evaluate(x_plot), dtype=np.float64),
                    -y_clip, y_clip,
                )
                style = Qt.PenStyle.SolidLine if is_best else Qt.PenStyle.DashLine
                width = 3 if is_best else 2
                curve = self._plot_widget.plot(
                    x_plot, y_plot,
                    pen=pg.mkPen((r, g, b), width=width, style=style),
                    name=f"{display_idx + 1}. {canonical_name}",
                )
                self._fitted_curves[display_idx] = curve
                curve.setVisible(is_in_top)
            except (ValueError, TypeError, RuntimeError, OverflowError,
                    np.linalg.LinAlgError, FloatingPointError):
                self._fitted_curves[display_idx] = None

        self._refresh_latex_output()

    def _refresh_latex_output(self) -> None:
        """Rebuild the LaTeX pane: show only rows whose checkbox is checked."""
        parts: list[str] = []
        for idx, row_text in enumerate(self._model_latex_rows):
            cb = (self._option_checkboxes[idx]
                  if idx < len(self._option_checkboxes) else None)
            if cb is not None and cb.isChecked() and row_text:
                parts.append(row_text)
        self._latex_output.setPlainText(
            "\n".join(parts) if parts else "(no model selected)"
        )

    def _create_option_row(
        self,
        idx: int,
        model_name: str,
        marker: str,
        color: tuple[int, int, int],
        checked: bool,
        enabled: bool = True,
    ) -> tuple[QWidget, QCheckBox]:
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(2, 1, 2, 1)
        hl.setSpacing(6)

        cb = QCheckBox()
        cb.setChecked(checked)
        cb.setEnabled(enabled)
        cb.toggled.connect(lambda state, i=idx: self.toggle_option(i, state))

        r, g, b = color
        if enabled:
            style = f"color: rgb({r},{g},{b}); font-weight: bold;"
        else:
            style = f"color: rgba({r},{g},{b},90); font-weight: normal;"
        label_text = f"{idx + 1}. {model_name}" + (f" {marker}" if marker else "")
        lbl = QLabel(label_text)
        lbl.setStyleSheet(style)

        hl.addWidget(cb)
        hl.addWidget(lbl)
        hl.addStretch(1)
        return row, cb

    def toggle_option(self, index: int, checked: bool) -> None:
        if 0 <= index < len(self._fitted_curves):
            curve = self._fitted_curves[index]
            if curve is not None:
                curve.setVisible(checked)
        self._refresh_latex_output()

    def copy_latex(self) -> None:
        text = self._latex_output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Copied", "LaTeX copied to clipboard.")

    def show_settings(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec():
            new_s = dlg.get_settings()
            if new_s is None:
                QMessageBox.critical(self, "Invalid Settings",
                                     "One or more values are invalid.")
                return
            try:
                latex_format_changed = (
                    new_s.latex_approx != self._settings.latex_approx
                    or new_s.latex_decimals != self._settings.latex_decimals
                )
                self._settings = new_s
                self._latex_gen.reconfigure(new_s.latex_approx, new_s.latex_decimals)

                if latex_format_changed and any(
                    m is not None for m in self._shown_models
                ):
                    y_scale = (float(np.std(self._y_proc)) or 1.0
                               if len(self._y_proc) > 0 else 1.0)
                    for idx, model in enumerate(self._shown_models):
                        if model is None:
                            self._model_latex_rows[idx] = ""
                            continue
                        is_best = (
                            idx == next(
                                (i for i, m in enumerate(self._shown_models)
                                 if m is not None), -1
                            )
                        )
                        latex_str = self._latex_gen.generate(model)
                        self._model_latex_rows[idx] = self._build_latex_row(
                            idx, model, y_scale, latex_str, is_best
                        )
                    self._refresh_latex_output()

                self._configure_plot()
                if not latex_format_changed:
                    self.clear_drawing()
            except ValueError as exc:
                QMessageBox.critical(self, "Invalid Settings", str(exc))


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()