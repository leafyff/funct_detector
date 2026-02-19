# Funct Detector

Interactive function drawing and automatic analytical approximation.

**Latest release:** v0.3  
**Dev logs:** Available inside the repository

Function Detector is a PySide6 + NumPy + SciPy application that lets you draw a function on a coordinate plane and automatically approximates it using multiple canonical mathematical models.

#### How to use:
You draw the curve.  
The program analyzes it.  
It gives you ranked analytical approximations.

---

## Overview

When the application starts, you see a coordinate grid.
You draw a curve using the mouse.
The program preprocesses the data and runs twelve independent approximation engines.
Each model is evaluated and scored using normalized error metrics and a complexity penalty.
The best approximations are highlighted and their analytical expressions are generated in LaTeX form.

---

## Implemented Approximation Methods

The following canonical models are implemented:

1. Cubic Spline
2. Interpolation polynomial (Chebyshev Basis)
3. L∞ minimax polynomial
4. Polynomial Least Squares Approximation (Chebyshev Basis)
5. Non-Uniform Fast Fourier Transform (NUFFT)
6. AAA Algorithm
7. Exponential curve
8. Logarithmic curve
9. Rational curve
10. Sinusoidal curve
11. Tangential curve
12. Arctan (S-curve)
---

## Model Scoring

Each fitted model is evaluated using:

$$
\text{Score} = \alpha \frac{L_\infty}{\sigma} + \beta \frac{\text{RMSE}}{\sigma} + \gamma \cdot \text{complexity}  
$$

Where:
- ( $L_\infty$ ) is the maximum absolute error
- RMSE is the root-mean-square error
- ( $\sigma$ ) is the standard deviation of the drawn data
- complexity penalizes overfitting

The system favors models that balance accuracy and simplicity.

---

## Mathematical Techniques Used

- Cubic smoothing splines
- Chebyshev polynomial bases
- Linear programming (HiGHS backend) for minimax approximation
- Trigonometric least squares regression (NUFFT-style)
- Rational approximation via the AAA algorithm
- Nonlinear curve fitting (SciPy `curve_fit`)
- Symbolic LaTeX generation via SymPy

---

## Architecture

The project is built around:
- `ModelFitter` abstract base class
- Independent fitter implementations
- `ModelSelectionService` for orchestration and ranking
- Immutable `FittedModel` data class
- `LaTeXGenerator` for symbolic equation output

Each approximation method is isolated and extendable.
New models can be added without modifying the selection pipeline.

---

## Dependencies

- Python 3.10+
- NumPy
- SciPy
- SymPy
- PySide6
- PyQtGraph

---

## Running

```bash
pip install -r requirements.txt
python main.py
```

---

## Purpose

Function Detector is both:
- An educational tool for approximation theory
- A sandbox for comparing interpolation, regression, and rational methods
- A visual demonstration of overfitting vs structured modeling
- A practical experiment in numerical stability

---
