import sys
import warnings

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QTextEdit, QDialog,
                               QLabel, QLineEdit, QGridLayout, QMessageBox,
                               QDoubleSpinBox, QCheckBox, QGroupBox)
from dataclasses import dataclass
from typing import Optional

from preprocessing import preprocess_stroke, is_function, detect_discontinuities
from fitting import fit_models, select_best_model
from latex_gen import model_to_latex

warnings.filterwarnings('ignore')


@dataclass
class PlotSettings:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    grid_spacing: float = 1.0
    accuracy: float = 0.01


class SettingsDialog(QDialog):
    def __init__(self, current_settings: PlotSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self.settings = current_settings

        layout = QGridLayout()

        self.x_min_edit = QLineEdit(str(self.settings.x_min))
        self.x_max_edit = QLineEdit(str(self.settings.x_max))
        self.y_min_edit = QLineEdit(str(self.settings.y_min))
        self.y_max_edit = QLineEdit(str(self.settings.y_max))
        self.grid_edit = QLineEdit(str(self.settings.grid_spacing))

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

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout, 5, 0, 1, 2)
        self.setLayout(layout)

    def get_settings(self) -> Optional[PlotSettings]:
        try:
            return PlotSettings(
                x_min=float(self.x_min_edit.text()),
                x_max=float(self.x_max_edit.text()),
                y_min=float(self.y_min_edit.text()),
                y_max=float(self.y_max_edit.text()),
                grid_spacing=float(self.grid_edit.text()),
                accuracy=self.settings.accuracy
            )
        except ValueError:
            return None


class DrawingApp(QMainWindow):
    COLORS = [
        (255, 0, 0),
        (0, 200, 0),
        (255, 150, 0),
        (200, 0, 200),
        (0, 200, 200),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Function Drawer & LaTeX Generator")
        self.setGeometry(100, 100, 1400, 800)

        self.settings = PlotSettings()
        self.drawing = False
        self.points = []
        self.fitted_curves = []
        self.drawn_curve = None
        self.all_models = []
        self.x_proc = None
        self.y_proc = None
        self.option_checkboxes = []

        self.plot_widget = None
        self.accuracy_spinbox = None
        self.clear_button = None
        self.fit_button = None
        self.export_button = None
        self.settings_button = None
        self.options_layout = None
        self.latex_output = None

        self.setup_ui()
        self.setup_plot()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(10, 10))
        left_layout.addWidget(self.plot_widget)

        accuracy_layout = QHBoxLayout()
        accuracy_layout.addWidget(QLabel("Approximation Accuracy:"))

        self.accuracy_spinbox = QDoubleSpinBox()
        self.accuracy_spinbox.setRange(0.0001, 1.0)
        self.accuracy_spinbox.setSingleStep(0.001)
        self.accuracy_spinbox.setDecimals(4)
        self.accuracy_spinbox.setValue(self.settings.accuracy)
        self.accuracy_spinbox.valueChanged.connect(self.on_accuracy_changed)
        accuracy_layout.addWidget(self.accuracy_spinbox)

        accuracy_layout.addWidget(QLabel("(lower = higher quality)"))
        accuracy_layout.addStretch()

        left_layout.addLayout(accuracy_layout)

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

        right_layout = QVBoxLayout()

        options_group = QGroupBox("Display Options")
        self.options_layout = QVBoxLayout()
        options_group.setLayout(self.options_layout)
        right_layout.addWidget(options_group)

        right_layout.addWidget(QLabel("LaTeX Output (Top Candidates):"))
        self.latex_output = QTextEdit()
        self.latex_output.setReadOnly(True)
        right_layout.addWidget(self.latex_output)

        main_layout.addLayout(right_layout, 1)

        scene = self.plot_widget.scene()
        scene.sigMouseClicked.connect(self.mouse_clicked)
        scene.sigMouseMoved.connect(self.mouse_moved)

    def setup_plot(self):
        self.plot_widget.setLabel('left', 'y')
        self.plot_widget.setLabel('bottom', 'x')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setXRange(self.settings.x_min, self.settings.x_max)
        self.plot_widget.setYRange(self.settings.y_min, self.settings.y_max)

    def on_accuracy_changed(self, value):
        self.settings.accuracy = max(0.0001, value)
        if value != self.settings.accuracy:
            self.accuracy_spinbox.setValue(self.settings.accuracy)

    def mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                if not self.drawing:
                    self.drawing = True
                    self.points = [(mouse_point.x(), mouse_point.y())]
                else:
                    self.drawing = False

    def mouse_moved(self, pos):
        if self.drawing:
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                self.points.append((mouse_point.x(), mouse_point.y()))
                self.update_drawing()

    def update_drawing(self):
        if self.drawn_curve is not None:
            self.plot_widget.removeItem(self.drawn_curve)

        if len(self.points) > 1:
            points_array = np.array(self.points)
            self.drawn_curve = self.plot_widget.plot(
                points_array[:, 0],
                points_array[:, 1],
                pen=pg.mkPen((100, 100, 255), width=2),
                name="Drawn curve"
            )

    def clear_fitted_curves(self):
        for curve in self.fitted_curves:
            self.plot_widget.removeItem(curve)
        self.fitted_curves = []

    def clear_option_checkboxes(self):
        for checkbox in self.option_checkboxes:
            checkbox.setParent(None)
            checkbox.deleteLater()
        self.option_checkboxes = []

    def clear_drawing(self):
        self.points = []
        self.drawing = False
        if self.drawn_curve is not None:
            self.plot_widget.removeItem(self.drawn_curve)
            self.drawn_curve = None

        self.clear_fitted_curves()
        self.clear_option_checkboxes()

        self.all_models = []
        self.x_proc = None
        self.y_proc = None
        self.latex_output.clear()

        self.plot_widget.clear()
        self.plot_widget.addLegend(offset=(10, 10))

    def toggle_option(self, index, checked):
        if index < len(self.fitted_curves):
            self.fitted_curves[index].setVisible(checked)

    def fit_curve(self):
        if len(self.points) < 10:
            QMessageBox.warning(self, "Insufficient Data", "Draw a longer curve (at least 10 points)")
            return

        points_array = np.array(self.points)
        x_raw = points_array[:, 0]
        y_raw = points_array[:, 1]

        domain_width = self.settings.x_max - self.settings.x_min
        domain_height = self.settings.y_max - self.settings.y_min

        x_proc, y_proc = preprocess_stroke(x_raw, y_raw, domain_width, domain_height)

        if len(x_proc) < 5:
            QMessageBox.warning(self, "Processing Error", "Not enough points after preprocessing")
            return

        is_func, error_msg = is_function(x_proc, y_proc, domain_width)

        if not is_func:
            QMessageBox.warning(self, "Not a Function", error_msg)
            return

        self.x_proc = x_proc
        self.y_proc = y_proc

        segments = detect_discontinuities(x_proc, y_proc)

        models = fit_models(x_proc, y_proc, segments, self.settings.accuracy)

        if not models:
            QMessageBox.warning(self, "Fitting Failed", "Could not fit any model to the data")
            return

        best_model = select_best_model(models, x_proc, y_proc)

        y_std = np.std(y_proc)
        scores = []
        for model in models:
            normalized_rmse = model.rmse / y_std if y_std > 0 else model.rmse
            score = normalized_rmse + 0.1 * model.aic + model.complexity
            scores.append(score)

        sorted_indices = np.argsort(scores)
        top_models = [models[i] for i in sorted_indices[:min(5, len(models))]]

        self.all_models = top_models

        self.clear_fitted_curves()
        self.clear_option_checkboxes()

        latex_parts = []
        x_plot = np.linspace(np.min(x_proc), np.max(x_proc), 500)

        for idx, model in enumerate(top_models):
            latex_str = model_to_latex(model)
            marker = "★ BEST FIT" if model == best_model else ""
            latex_parts.append(f"Option {idx + 1} - {model.name} {marker}\nRMSE: {model.rmse:.6f}\n{latex_str}\n")

            y_plot = model.evaluate(x_plot)

            color = self.COLORS[idx % len(self.COLORS)]
            pen_style = Qt.PenStyle.SolidLine if model == best_model else Qt.PenStyle.DashLine
            pen_width = 3 if model == best_model else 2

            curve = self.plot_widget.plot(
                x_plot,
                y_plot,
                pen=pg.mkPen(color, width=pen_width, style=pen_style),
                name=f"Option {idx + 1}: {model.name}"
            )
            self.fitted_curves.append(curve)

            checkbox = QCheckBox(f"Option {idx + 1}: {model.name} {marker}")
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f"color: rgb{color}; font-weight: bold;")
            checkbox.stateChanged.connect(lambda state, i=idx: self.toggle_option(i, state == Qt.CheckState.Checked))
            self.options_layout.addWidget(checkbox)
            self.option_checkboxes.append(checkbox)

        self.latex_output.setPlainText("\n".join(latex_parts))

    def copy_latex(self):
        latex_text = self.latex_output.toPlainText()
        if latex_text:
            QApplication.clipboard().setText(latex_text)
            QMessageBox.information(self, "Copied", "LaTeX copied to clipboard")

    def show_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            if new_settings:
                self.settings = new_settings
                self.accuracy_spinbox.setValue(self.settings.accuracy)
                self.setup_plot()
                self.clear_drawing()


def run_synthetic_test():
    print("=== Synthetic Test Mode ===\n")

    test_functions = [
        ("Polynomial", lambda x: 2 * x ** 2 - 3 * x + 1, -5, 5),
        ("Sine", lambda x: 3 * np.sin(0.5 * x) + 1, -10, 10),
        ("Exponential", lambda x: 2 * np.exp(0.3 * x) - 5, -3, 3),
    ]

    for name, func, x_min, x_max in test_functions:
        print(f"Testing: {name}")

        x_true = np.linspace(x_min, x_max, 100)
        y_true = func(x_true)

        noise = np.random.normal(0, 0.1 * np.std(y_true), size=y_true.shape)
        y_noisy = y_true + noise

        domain_width = x_max - x_min
        domain_height = np.max(y_noisy) - np.min(y_noisy)

        x_proc, y_proc = preprocess_stroke(x_true, y_noisy, domain_width, domain_height)

        is_func, _ = is_function(x_proc, y_proc, domain_width)
        print(f"  Is function: {is_func}")

        segments = detect_discontinuities(x_proc, y_proc)
        print(f"  Segments: {len(segments)}")

        models = fit_models(x_proc, y_proc, segments, accuracy=0.01)

        if models:
            best_model = select_best_model(models, x_proc, y_proc)
            latex = model_to_latex(best_model)
            print(f"  Best model: {best_model.name}")
            print(f"  LaTeX: {latex}")
            print(f"  RMSE: {best_model.rmse:.6f}")

            print(f"  All candidates ({len(models)}):")
            for idx, model in enumerate(models, 1):
                latex_candidate = model_to_latex(model)
                print(f"    {idx}. {model.name} - RMSE: {model.rmse:.6f}")
                print(f"       {latex_candidate}")
            print()
        else:
            print("  No models found\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_synthetic_test()
    else:
        app = QApplication(sys.argv)
        window = DrawingApp()
        window.show()
        sys.exit(app.exec())


if __name__ == '__main__':
    main()
