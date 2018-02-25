#!/usr/bin/env python
# ecg_python/ecg.py

"""
===============================================================================
    Copyright (C) 2015-2018 Rudolf Cardinal (rudolf@pobox.com).

    This file is part of CRATE.

    CRATE is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRATE is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CRATE. If not, see <http://www.gnu.org/licenses/>.
===============================================================================

Based on:
    https://github.com/TDeagan/pyEKGduino
    https://github.com/pbmanis/EKGMonitor
        ... more sophisticated signal processing

"""

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pdb import set_trace
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Dict, List, Optional, Tuple

from biosppy.signals.ecg import ecg as biosppy_ecg
from biosppy.utils import ReturnTuple
from cardinal_pythonlib.argparse_func import RawDescriptionArgumentDefaultsHelpFormatter  # noqa
from cardinal_pythonlib.datetimefunc import format_datetime
from cardinal_pythonlib.dsp import (
    lowpass_filter,
    highpass_filter,
    bandpass_filter,
    notch_filter,
)
from cardinal_pythonlib.json.serialize import JsonClassEncoder, json_decode
from cardinal_pythonlib.maths_py import normal_round_float
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
import numpy as np
from pendulum import Pendulum
from PyQt5.QtCore import (
    pyqtRemoveInputHook,
    # pyqtRestoreInputHook,
    pyqtSignal,
    QObject,
    QRectF,
    Qt,
    QTimer,
)
from PyQt5.QtGui import (
    QColor,
    QPalette,
    QPen,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QWidget,
)
from pyqtgraph import PlotWidget
from pyqtgraph.graphicsItems import PlotItem
from pyqtgraph.exporters import ImageExporter, SVGExporter
from pyqtgraph.parametertree import Parameter
# https://www.reportlab.com/docs/reportlab-userguide.pdf
from reportlab.graphics.shapes import Drawing
from reportlab.platypus.flowables import Flowable, Image
# ... watch out; there are both shapes.Image and flowables.Image
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from serial import (  # pip install pyserial
    EIGHTBITS,
    PARITY_NONE,
    Serial,
    SerialException,
    STOPBITS_ONE
)
from serial.tools.list_ports import comports
from svglib.svglib import svg2rlg

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# -----------------------------------------------------------------------------
# Communication with Arduino
# -----------------------------------------------------------------------------

# All these to match ecg.cpp in one way or another:
CMD_SOURCE_ANALOGUE = "A"
CMD_SOURCE_SINE_GENERATOR = "G"
CMD_SOURCE_TIME = "T"
CMD_SINE_GENERATOR_FREQUENCY = "S"  # with float parameter: freq_hz
CMD_SAMPLE_FREQUENCY = "F"  # with float parameter: freq_hz
CMD_SAMPLE_CONTINUOUS = "C"
CMD_SAMPLE_NUMBER = "N"  # with int parameter: num_samples
CMD_QUIET = "Q"
CMD_TERMINATOR = "\n"
RESP_HELLO = "Hello"
RESP_OK = "OK"
PREFIX_DEBUG = "Debug: "
PREFIX_INFO = "Info: "
PREFIX_WARNING = "Warning: "
PREFIX_ERROR = "Error: "
DEFAULT_BAUD_RATE = 1000000
MIN_INPUT = 0
MAX_INPUT = 1024  # https://www.gammon.com.au/adc

# Characteristic of our serial comms:
SERIAL_ENCODING = "ascii"

# -----------------------------------------------------------------------------
# ECG capture
# -----------------------------------------------------------------------------

# Our fixed preferences:
SAMPLING_FREQ_HZ = 300.0  # twice the "pro" top frequency of 150 Hz (the Nyquist frequency for 150 Hz signals)  # noqa
POLLING_TIMER_PERIOD_MS = 5
MIN_FILTER_FREQ_HZ = 0
MAX_FILTER_FREQ_HZ = SAMPLING_FREQ_HZ
MIN_NUMTAPS = 1
MIN_NOTCH_Q = 1.0  # no idea
MAX_NOTCH_Q = 1000.0  # no idea
MIN_ECG_DURATION_S = 10.0
MAX_ECG_DURATION_S = 300.0

# User preferences:
DEFAULT_ECG_DURATION_S = 30.0
DEFAULT_SINE_FREQUENCY_HZ = 1.0
# Filtering preferences:
DEFAULT_INVERT = False
DEFAULT_CENTRE = False
DEFAULT_USE_HIGHPASS = True
DEFAULT_HIGHPASS_FREQ_HZ = 0.05
# ... professional uses [0.05 - 150 Hz]
#     http://www.ems12lead.com/2014/03/10/understanding-ecg-filtering/
DEFAULT_HIGHPASS_NUMTAPS = 99  # empirically; needs to be high-ish
DEFAULT_USE_LOWPASS = True
DEFAULT_LOWPASS_FREQ_HZ = 40.0  # this makes the most difference!
# ... professional kit uses e.g. a 25, 35, 75, 100, or 150 Hz low-pass filter:
#     https://www.ncbi.nlm.nih.gov/pubmed/24369740
DEFAULT_LOWPASS_NUMTAPS = 99  # empirically; needs to be high-ish
# For proper calculations about the number of taps:
# https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need  # noqa
DEFAULT_USE_NOTCH = True
DEFAULT_NOTCH_FREQ_HZ = 50.0  # UK mains hum
DEFAULT_NOTCH_Q = 25.0  # Q = w0/bw, so for 50Hz +/- 1 Hz (BW 2Hz), Q = 50/2
# See also:
# - http://www.medteq.info/med/ECGFilters

# -----------------------------------------------------------------------------
# Cosmetics
# -----------------------------------------------------------------------------

USE_PDF_EXPORT = True
USE_SVG_EXPORT = False
DTYPE = "float64"
JSON_FILE_FILTER = "JSON Files (*.json);;All files (*)"
BITMAP_FILE_FILTER = "PNG files (*.png);;All files (*)"
SVG_FILE_FILTER = "SVG files (*.svg);;All files (*)"
PDF_FILE_FILTER = "PDF files (*.pdf);;All files (*)"
UPDATE_ECG_EVERY_MS = 20  # 50 Hz
UPDATE_ANALYTICS_EVERY_MS = 500  # 0.5 Hz
MICROSECONDS_PER_S = 1000000
ECG_PEN = QPen(QColor(255, 0, 0))
ECG_PEN.setCosmetic(True)  # don't scale the pen width!
ECG_PEN.setWidth(1)
DATETIME_FORMAT = "%a %d %B %Y, %H:%M"  # e.g. Wed 24 July 2013, 20:04


# =============================================================================
# Helper functions
# =============================================================================

def current_milli_time() -> int:
    # time.time() is in seconds, with fractions (usually)
    # https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python  # noqa
    return int(round(time.time() * 1000))


def fit_image(src_width: float,
              src_height: float,
              available_width: float,
              available_height: float) -> Tuple[float, float]:
    src_aspect_ratio = src_width / src_height
    dest_aspect_ratio = available_width / available_height
    if src_aspect_ratio >= dest_aspect_ratio:
        # Source image is "wider" in aspect ratio than the destination, so
        # its width will be limiting.
        final_width = available_width
        final_height = final_width / src_aspect_ratio
    else:
        # Source image is "taller". Its height is limiting.
        final_height = available_height
        final_width = final_height * src_aspect_ratio
    return final_width, final_height


@contextmanager
def block_qt_signals_from(x: QObject) -> None:
    x.blockSignals(True)
    yield
    x.blockSignals(False)


def debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt."""
    # https://stackoverflow.com/questions/1736015/debugging-a-pyqt4-app
    pyqtRemoveInputHook()
    set_trace()
    # When you've finished stepping: pyqtRestoreInputHook()


# =============================================================================
# Input validation
# =============================================================================

def get_valid_float(x: str, default: float,
                    minimum: float = None, maximum: float = None) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        f = default
    if minimum is not None:
        f = max(minimum, f)
    if maximum is not None:
        f = min(f, maximum)
    return f


def get_valid_int(x: str, default: int,
                  minimum: int = None, maximum: int = None) -> int:
    try:
        f = int(x)
    except (TypeError, ValueError):
        f = default
    if minimum is not None:
        f = max(minimum, f)
    if maximum is not None:
        f = min(f, maximum)
    return f


# =============================================================================
# Advanced analytics window
# =============================================================================

class AdvancedWindow(QDialog):
    # noinspection PyArgumentList
    def __init__(self,
                 ecg_info: ReturnTuple,
                 parent: QWidget = None) -> None:
        """
        ecg_info:
            ts : array
                Signal time axis reference (seconds).
            filtered : array
                Filtered ECG signal.
            rpeaks : array
                R-peak location indices.
            templates_ts : array
                Templates time axis reference (seconds).
            templates : array
                Extracted heartbeat templates.
            heart_rate_ts : array
                Heart rate time axis reference (seconds).
            heart_rate : array
                Instantaneous heart rate (bpm).
        """
        super().__init__(parent=parent)
        self.ecg_info = ecg_info
        t = ecg_info['ts']  # type: np.ndarray
        y = ecg_info['filtered']  # type: np.ndarray
        rr_peak_indices = ecg_info['rpeaks']  # type: np.ndarray
        r_peak_t = np.take(t, rr_peak_indices)
        r_peak_y = np.take(y, rr_peak_indices)

        layout = QGridLayout()

        filtered_lbl = QLabel(
            "(Re-)Filtered ECG used for analysis, with R peaks")
        self.plot_filtered = PlotWidget(
            labels={
                "bottom": "time (s)",
                "left": "Filtered ECG signal (arbitrary units)"
            }
        )
        self.plot_filtered.plot(
            x=ecg_info['ts'],
            y=ecg_info['filtered']
        )
        self.plot_filtered.plot(
            x=r_peak_t,
            y=r_peak_y,
            pen=(200, 200, 200),
            symbolBrush=(255, 0, 0),
            symbolPen='w'
        )

        hr_lbl = QLabel("Heart rate over time")
        self.plot_hr = PlotWidget(
            labels={
                "bottom": "time (s)",
                "left": "Heart rate (bpm)"
            }
        )
        self.plot_hr.plot(
            x=ecg_info['heart_rate_ts'],
            y=ecg_info['heart_rate']
        )
        if t.size >= 2:
            # Set the same X scale as the ECG.
            # (Otherwise it's out of alignment, as HR isn't calculated over
            # the same time range as the ECG itself -- slightly shorter.)
            self.plot_hr.setXRange(t[0], t[-1])

        poincare_lbl = QLabel("Poincaré plot")
        self.plot_poincare = PlotWidget(
            labels={
                "bottom": "R-R interval (s)",  # x
                "left": "Next R-R interval (s)",  # y
            }
        )
        rr_intervals = np.diff(r_peak_t, n=1)
        rr_intervals_t_plus_1 = rr_intervals[1:]
        rr_intervals_t = rr_intervals[:-1]
        self.plot_poincare.plot(
            x=rr_intervals_t,
            y=rr_intervals_t_plus_1,
            pen=(200, 200, 200),
            symbolBrush=(255, 0, 0),
            symbolPen='w'
        )

        big_plot_width = 10
        small_plot_width = 5

        filter_row = 0
        filter_ecg_height = 5
        layout.addWidget(filtered_lbl, filter_row, 0)
        layout.addWidget(self.plot_filtered, filter_row + 1, 0,
                         filter_ecg_height, big_plot_width)

        hr_row = filter_row + 1 + filter_ecg_height
        hr_graph_height = 5
        layout.addWidget(hr_lbl, hr_row, 0)
        layout.addWidget(self.plot_hr, hr_row + 1, 0,
                         hr_graph_height, big_plot_width)

        poincare_row = hr_row + 1 + hr_graph_height
        poincare_graph_height = 5
        layout.addWidget(poincare_lbl, poincare_row, 0)
        layout.addWidget(self.plot_poincare, poincare_row + 1, 0,
                         poincare_graph_height, small_plot_width)

        self.setLayout(layout)


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QWidget):
    # To suppress lots of: "Cannot find reference 'connect' in 'function'":
    # noinspection PyUnresolvedReferences,PyArgumentList
    def __init__(self,
                 app: "EcgApplication",
                 parent: QWidget = None,
                 hr_dp: int = 0) -> None:
        super().__init__(parent=parent)
        self.app = app
        self.ecg = app.ecg
        self.hr_dp = hr_dp

        layout = QGridLayout()
        self.plot_ecg = PlotWidget(
            labels={"bottom": "time (s)"},
            background='w'  # white
        )
        self.curve = self.plot_ecg.plot(pen=ECG_PEN)
        
        # Person

        name_lbl = QLabel("Name")
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_person_changed)
        details_lbl = QLabel("Details")
        self.details_edit = QLineEdit()
        self.details_edit.textChanged.connect(self.on_person_changed)
        comment_lbl = QLabel("Comments")
        self.comment_edit = QLineEdit()
        self.comment_edit.textChanged.connect(self.on_person_changed)
        
        # Capture

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.quit_button)

        save_data_btn = QPushButton("Save data")
        save_data_btn.clicked.connect(self.on_save_data)

        load_data_btn = QPushButton("Load data")
        load_data_btn.clicked.connect(self.on_load_data)

        save_picture_btn = QPushButton("Save picture")
        save_picture_btn.clicked.connect(self.on_save_picture)

        self.continuous_btn = QPushButton("Continuous")
        self.continuous_btn.setEnabled(False)  # until Arduino awake
        self.continuous_btn.clicked.connect(self.on_continuous)

        self.capture_time_btn = QPushButton("Capture buffer length")
        self.capture_time_btn.setEnabled(False)  # until Arduino awake
        self.capture_time_btn.clicked.connect(self.on_capture_for_time)

        self.stop_btn = QPushButton("Stop capture")
        self.stop_btn.setEnabled(False)  # until Arduino awake
        self.stop_btn.clicked.connect(self.on_stop)

        buffer_duration_lbl = QLabel("Buffer duration (s)")
        self.buffer_duration_edit = QLineEdit()
        self.buffer_duration_edit.textChanged.connect(self.on_buffer_duration)

        # Filters

        self.invert_btn = QCheckBox("Invert")
        self.invert_btn.stateChanged.connect(self.on_invert)

        self.centre_btn = QCheckBox("Centre on mean")
        self.centre_btn.stateChanged.connect(self.on_centre)

        self.highpass_btn = QCheckBox("High-pass filter (eliminate LF)")
        self.highpass_btn.stateChanged.connect(self.on_highpass)
        self.highpass_cutoff_lbl = QLabel("Highpass cutoff freq. (Hz)")
        self.highpass_cutoff_edit = QLineEdit()
        self.highpass_cutoff_edit.textChanged.connect(self.on_highpass)
        self.highpass_numtaps_lbl = QLabel("Highpass [or bandpass] #taps")
        self.highpass_numtaps_edit = QLineEdit()
        self.highpass_numtaps_edit.textChanged.connect(self.on_highpass)

        self.notch_btn = QCheckBox("Notch filter")
        self.notch_btn.stateChanged.connect(self.on_notch)
        self.notch_freq_lbl = QLabel("Notch freq. (Hz)")
        self.notch_freq_edit = QLineEdit()
        self.notch_freq_edit.textChanged.connect(self.on_notch)
        self.notch_q_lbl = QLabel("Notch Q[uality] factor")
        self.notch_q_edit = QLineEdit()
        self.notch_q_edit.textChanged.connect(self.on_notch)

        self.lowpass_btn = QCheckBox("Low-pass filter (eliminate HF)")
        self.lowpass_btn.stateChanged.connect(self.on_lowpass)
        self.lowpass_cutoff_lbl = QLabel("Lowpass cutoff freq. (Hz)")
        self.lowpass_cutoff_edit = QLineEdit()
        self.lowpass_cutoff_edit.textChanged.connect(self.on_lowpass)
        self.lowpass_numtaps_lbl = QLabel("Lowpass #taps")
        self.lowpass_numtaps_edit = QLineEdit()
        self.lowpass_numtaps_edit.textChanged.connect(self.on_lowpass)

        # Source mode

        self.ecg_mode_btn = QRadioButton("ECG mode")
        self.ecg_mode_btn.clicked.connect(self.on_ecg_mode)

        sine_mode_btn = QRadioButton("Sine wave test mode")
        sine_mode_btn.clicked.connect(self.on_sine_mode)

        time_mode_btn = QRadioButton("Time test mode")
        time_mode_btn.clicked.connect(self.on_time_mode)

        self.sine_freq_lbl = QLabel("Sine wave frequency (Hz)")
        self.sine_freq_edit = QLineEdit()
        self.sine_freq_edit.textChanged.connect(self.on_sine_mode)

        # Rescale, clear, advanced, quite

        rescale_btn = QPushButton("Rescale")
        rescale_btn.clicked.connect(self.rescale)

        clear_btn = QPushButton("Clear ECG")
        clear_btn.clicked.connect(self.on_clear)

        analytics_btn = QPushButton("View advanced plots")
        analytics_btn.clicked.connect(self.view_advanced)

        # Live analytics

        red_palette = QPalette()
        red_palette.setColor(QPalette.Foreground, Qt.red)

        hr_static_lbl = QLabel("Average heart rate (bpm)")
        self.hr_lbl = QLabel()
        self.hr_lbl.setPalette(red_palette)

        # Layout

        person_n_cols = 4

        name_row = 0
        layout.addWidget(name_lbl, name_row, 0)
        layout.addWidget(self.name_edit, name_row, 1, 1, person_n_cols)

        details_row = name_row + 1
        layout.addWidget(details_lbl, details_row, 0)
        layout.addWidget(self.details_edit, details_row, 1, 1, person_n_cols)

        comment_row = details_row + 1
        layout.addWidget(comment_lbl, comment_row, 0)
        layout.addWidget(self.comment_edit, comment_row, 1, 1, person_n_cols)

        action_row = comment_row + 1
        layout.addWidget(self.continuous_btn, action_row, 0)
        layout.addWidget(self.capture_time_btn, action_row, 1)
        layout.addWidget(self.stop_btn, action_row, 2)
        layout.addWidget(buffer_duration_lbl, action_row, 3)
        layout.addWidget(self.buffer_duration_edit, action_row, 4)

        simple_transform_row = action_row + 1
        layout.addWidget(self.invert_btn, simple_transform_row, 0)
        layout.addWidget(self.centre_btn, simple_transform_row, 1)

        highpass_row = simple_transform_row + 1
        layout.addWidget(self.highpass_btn, highpass_row, 0)
        layout.addWidget(self.highpass_cutoff_lbl, highpass_row, 1)
        layout.addWidget(self.highpass_cutoff_edit, highpass_row, 2)
        layout.addWidget(self.highpass_numtaps_lbl, highpass_row, 3)
        layout.addWidget(self.highpass_numtaps_edit, highpass_row, 4)

        notch_row = highpass_row + 1
        layout.addWidget(self.notch_btn, notch_row, 0)
        layout.addWidget(self.notch_freq_lbl, notch_row, 1)
        layout.addWidget(self.notch_freq_edit, notch_row, 2)
        layout.addWidget(self.notch_q_lbl, notch_row, 3)
        layout.addWidget(self.notch_q_edit, notch_row, 4)

        lowpass_row = notch_row + 1
        layout.addWidget(self.lowpass_btn, lowpass_row, 0)
        layout.addWidget(self.lowpass_cutoff_lbl, lowpass_row, 1)
        layout.addWidget(self.lowpass_cutoff_edit, lowpass_row, 2)
        layout.addWidget(self.lowpass_numtaps_lbl, lowpass_row, 3)
        layout.addWidget(self.lowpass_numtaps_edit, lowpass_row, 4)

        mode_row = lowpass_row + 1
        layout.addWidget(self.ecg_mode_btn, mode_row, 0)
        layout.addWidget(sine_mode_btn, mode_row, 1)
        layout.addWidget(time_mode_btn, mode_row, 2)
        layout.addWidget(self.sine_freq_lbl, mode_row, 3)
        layout.addWidget(self.sine_freq_edit, mode_row, 4)

        save_load_row = mode_row + 1
        layout.addWidget(save_data_btn, save_load_row, 0)
        layout.addWidget(load_data_btn, save_load_row, 1)
        layout.addWidget(save_picture_btn, save_load_row, 2)

        clear_quit_row = save_load_row + 1
        layout.addWidget(rescale_btn, clear_quit_row, 0)
        layout.addWidget(clear_btn, clear_quit_row, 1)
        layout.addWidget(analytics_btn, clear_quit_row, 2)
        layout.addWidget(quit_btn, clear_quit_row, 4)

        ecg_row = clear_quit_row + 1
        n_ecg_rows = 10
        layout.addWidget(self.plot_ecg, ecg_row, 0, n_ecg_rows, 5)

        summary_hr_row = ecg_row + n_ecg_rows
        layout.addWidget(hr_static_lbl, summary_hr_row, 0)
        layout.addWidget(self.hr_lbl, summary_hr_row, 1)

        self.setLayout(layout)
        self.set_defaults(trigger_ecg_update=False)

    def set_defaults(self, trigger_ecg_update: bool = True) -> None:
        with block_qt_signals_from(self.buffer_duration_edit):
            self.buffer_duration_edit.setText(str(DEFAULT_ECG_DURATION_S))

        with block_qt_signals_from(self.invert_btn):
            self.invert_btn.setChecked(DEFAULT_INVERT)
        with block_qt_signals_from(self.centre_btn):
            self.centre_btn.setChecked(DEFAULT_CENTRE)
        with block_qt_signals_from(self.highpass_btn):
            self.highpass_btn.setChecked(DEFAULT_USE_HIGHPASS)
        with block_qt_signals_from(self.highpass_cutoff_edit):
            self.highpass_cutoff_edit.setText(str(DEFAULT_HIGHPASS_FREQ_HZ))
        with block_qt_signals_from(self.highpass_numtaps_edit):
            self.highpass_numtaps_edit.setText(str(DEFAULT_HIGHPASS_NUMTAPS))
        with block_qt_signals_from(self.notch_btn):
            self.notch_btn.setChecked(DEFAULT_USE_NOTCH)
        with block_qt_signals_from(self.lowpass_btn):
            self.lowpass_btn.setChecked(DEFAULT_USE_LOWPASS)
        with block_qt_signals_from(self.notch_freq_edit):
            self.notch_freq_edit.setText(str(DEFAULT_NOTCH_FREQ_HZ))
        with block_qt_signals_from(self.notch_q_edit):
            self.notch_q_edit.setText(str(DEFAULT_NOTCH_Q))
        with block_qt_signals_from(self.lowpass_cutoff_edit):
            self.lowpass_cutoff_edit.setText(str(DEFAULT_LOWPASS_FREQ_HZ))
        with block_qt_signals_from(self.lowpass_numtaps_edit):
            self.lowpass_numtaps_edit.setText(str(DEFAULT_LOWPASS_NUMTAPS))

        with block_qt_signals_from(self.sine_freq_edit):
            self.sine_freq_edit.setText(str(DEFAULT_SINE_FREQUENCY_HZ))

        self.update_from_settings(trigger_ecg_update=trigger_ecg_update)
        self.ecg_mode_btn.click()

    def on_arduino_awake(self) -> None:
        self.continuous_btn.setEnabled(True)
        self.capture_time_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def update_from_settings(self, trigger_ecg_update: bool = True) -> None:
        self.ecg.enable_updates(False)
        self.on_invert()
        self.on_centre()
        self.on_lowpass()
        self.on_highpass()
        self.on_notch()
        self.ecg.enable_updates(True)
        if trigger_ecg_update:
            self.trigger_ecg_update()

    def update_ecg_graph(self, xdata: np.array, ydata: np.array) -> None:
        self.curve.setData(xdata, ydata)

    def update_analysis(self, ecg_info: Optional[ReturnTuple]) -> None:
        """
        info is the output from biosppy.signals.ecg(); it's a combination
        tuple and dictionary.
        """
        if ecg_info is None:
            self.hr_lbl.setText("?")
            return

        hr_array = ecg_info["heart_rate"]  # type: np.ndarray
        if hr_array.size > 0:
            mean_hr = float(np.mean(hr_array))
            self.hr_lbl.setText(str(normal_round_float(mean_hr, self.hr_dp)))
        else:
            self.hr_lbl.setText("?")

    def on_person_changed(self) -> None:
        self.ecg.set_person_details(
            name=self.name_edit.text(),
            details=self.details_edit.text(),
            comment=self.comment_edit.text(),
        )

    def on_save_data(self) -> None:
        # noinspection PyArgumentList
        filename, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save data as",
            filter=JSON_FILE_FILTER,
        )
        if not filename:
            return
        self.ecg.save_json(filename)

    def on_load_data(self) -> None:
        # noinspection PyArgumentList
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Load data",
            filter=JSON_FILE_FILTER,
        )
        if not filename:
            return
        ecg = self.ecg
        if not ecg.load_json(filename):  # will emit signals to update the ECG graph  # noqa
            # noinspection PyCallByClass,PyArgumentList
            QMessageBox.warning(self, "Load failure",
                                "Failed to load JSON file")

        # We need to block signals, or setting the text will write, via the
        # signals system, to the EcgController object (while we asking it for
        # these values...) -- leads to corruption (e.g. spurious blank boxes)
        # otherwise.

        # DOESN'T WORK PROPERLY:
        # self.setUpdatesEnabled(False)
        # ...
        # self.setUpdatesEnabled(True)

        # Works, but a bit laborious:
        # Person
        with block_qt_signals_from(self.name_edit):
            self.name_edit.setText(ecg.person_name)
        with block_qt_signals_from(self.details_edit):
            self.details_edit.setText(ecg.person_details)
        with block_qt_signals_from(self.comment_edit):
            self.comment_edit.setText(ecg.person_comment)
        # Buffer size
        with block_qt_signals_from(self.buffer_duration_edit):
            self.buffer_duration_edit.setText(str(ecg.buffer_duration_s))
        # Filters
        with block_qt_signals_from(self.invert_btn):
            self.invert_btn.setChecked(ecg.invert)
        with block_qt_signals_from(self.centre_btn):
            self.centre_btn.setChecked(ecg.centre_on_mean)
        with block_qt_signals_from(self.highpass_btn):
            self.highpass_btn.setChecked(ecg.use_highpass_filter)
        with block_qt_signals_from(self.highpass_cutoff_edit):
            self.highpass_cutoff_edit.setText(str(ecg.highpass_cutoff_freq_hz))
        with block_qt_signals_from(self.highpass_numtaps_edit):
            self.highpass_numtaps_edit.setText(str(ecg.highpass_numtaps))
        with block_qt_signals_from(self.notch_btn):
            self.notch_btn.setChecked(ecg.use_notch_filter)
        with block_qt_signals_from(self.notch_freq_edit):
            self.notch_freq_edit.setText(str(ecg.notch_freq_hz))
        with block_qt_signals_from(self.notch_q_edit):
            self.notch_q_edit.setText(str(ecg.notch_quality_factor))
        with block_qt_signals_from(self.lowpass_btn):
            self.lowpass_btn.setChecked(ecg.use_lowpass_filter)
        with block_qt_signals_from(self.lowpass_cutoff_edit):
            self.lowpass_cutoff_edit.setText(str(ecg.lowpass_cutoff_freq_hz))
        with block_qt_signals_from(self.lowpass_numtaps_edit):
            self.lowpass_numtaps_edit.setText(str(ecg.lowpass_numtaps))

        self.update_from_settings(trigger_ecg_update=False)
        # ... deals with e.g. hide/show dialogue boxes

    def on_save_picture(self) -> None:
        if USE_PDF_EXPORT:
            filename_filter = PDF_FILE_FILTER
        elif USE_SVG_EXPORT:
            filename_filter = SVG_FILE_FILTER
        else:
            filename_filter = BITMAP_FILE_FILTER
        # noinspection PyArgumentList
        filename, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save image as",
            filter=filename_filter,
        )
        if not filename:
            return

        if USE_PDF_EXPORT:
            self._save_pdf(filename)
            return

        plotitem = self.plot_ecg.getPlotItem()
        if USE_SVG_EXPORT:
            self._save_svg(plotitem, filename)
        else:
            self._save_bitmap(plotitem, filename)

    def _save_pdf(self, filename: str) -> None:
        log.info("Exporting PDF to {!r}".format(filename))

        margin = 1.5 * cm
        pagesize = landscape(A4)  # width, height
        pagewidth, pageheight = pagesize
        contentwidth = pagewidth - 2 * margin
        contentheight = pageheight - 2 * margin
        max_imagewidth = contentwidth
        max_imageheight = contentheight - 2 * cm  # guess for text height

        with TemporaryDirectory() as tempdir:
            # We keep this tempdir going, because ReportLab sometimes (e.g.
            # with flowables.Image) doesn't read the file when the Image is
            # created, but at PDF write time.

            # Get the ECG as a reportlab Drawing
            plotitem = self.plot_ecg.getPlotItem()
            as_svg = False
            if as_svg:
                # Good things: colour, transparency
                # Bad things: scale goes wrong, including time scale!
                tmp_filename = os.path.join(tempdir, "tmp.svg")
                self._save_svg(plotitem, tmp_filename)
                # ecg_svg_drawing = Image(filename)  # doesn't work
                ecg = svg2rlg(tmp_filename)  # type: Drawing
                # The next bit is just empirical and not great:
                scale_factor = 0.75
                ecg.width *= scale_factor
                ecg.height *= scale_factor
                ecg.scale(scale_factor, scale_factor)
            else:
                tmp_filename = os.path.join(tempdir, "tmp.png")
                width_px, height_px = self._save_bitmap(plotitem, tmp_filename)
                imagewidth, imageheight = fit_image(
                    src_width=width_px,
                    src_height=height_px,
                    available_width=max_imagewidth,
                    available_height=max_imageheight
                )
                ecg = Image(tmp_filename, width=imagewidth, height=imageheight)

            # Make the PDF
            styles = getSampleStyleSheet()
            style = styles["Normal"]
            doc = SimpleDocTemplate(
                filename,
                pagesize=pagesize,
                title=self.ecg.person_name,
                author="Software by Rudolf Cardinal",
                subject="ECG (nondiagnostic; for fun only)",
                creator="RN Cardinal / Arduino ECG demo",
                leftMargin=margin,
                rightMargin=margin,
                topMargin=margin,
                bottomMargin=margin,
            )
            settings = self.ecg.get_settings_description()
            story = [
                # Spacer(1, 0.5 * cm),
                Paragraph(
                    "Name: <b>{}</b> // Details: {} // Comments: {}".format(
                        self.ecg.person_name or "—",
                        self.ecg.person_details or "—",
                        self.ecg.person_comment or "—",
                    ),
                    style
                ),
                Paragraph(
                    "ECG taken when: <b>{}</b>".format(
                        format_datetime(self.ecg.latest_datetime,
                                        DATETIME_FORMAT)),
                    style
                ),
                Paragraph(
                    "<i>Not of diagnostic grade; for fun only. "
                    "Settings: {}</i>".format(settings),
                    style
                ),
                ecg,
            ]  # type: List[Flowable]
            doc.build(story)

    @staticmethod
    def _save_svg(plotitem: PlotItem, filename: str) -> None:
        log.info("Writing SVG to {!r}".format(filename))
        exporter = SVGExporter(plotitem)
        # WATCH OUT. The SVGExporter is a bit buggy (e.g. axes move and
        # are mis-scaled).
        exporter.export(filename)

    @staticmethod
    def _save_bitmap(plotitem: PlotItem, filename: str) -> Tuple[int, int]:
        log.info("Writing bitmap to {!r}".format(filename))
        exporter = ImageExporter(plotitem)
        parameters = exporter.parameters()  # type: Parameter
        # Export it at the same resolution as it is on screen.
        # (This is the least bad option...)
        sourcerect = exporter.getSourceRect()  # type: QRectF
        width = int(sourcerect.width())
        height = int(sourcerect.height())
        # To compensate for crash due to bug
        #   https://github.com/pyqtgraph/pyqtgraph/issues/538
        # (in which, when you set e.g. width, it autosets e.g. height according
        # to the aspect ratio -- sometimes leading to float rather than int,
        # and to a crash...)
        # ... we disable signals.
        # MOREOVER, it won't write the value if it's the same according to
        # "=="; so, we have the problem that, for example, the width starts
        # out as 1837.0, and we can't write 1837 (int). So, this nonsense:
        parameters.param('width').setValue(
            None, blockSignal=exporter.widthChanged)
        parameters.param('width').setValue(
            width, blockSignal=exporter.widthChanged)
        parameters.param('height').setValue(
            None, blockSignal=exporter.heightChanged)
        parameters.param('height').setValue(
            height, blockSignal=exporter.heightChanged)
        exporter.export(filename)
        return width, height

    def rescale(self) -> None:
        self.plot_ecg.autoRange()
        self.plot_ecg.setXRange(0.0, self.ecg.buffer_duration_s)

    def set_graph_ranges_for_ecg(self) -> None:
        self.plot_ecg.setXRange(0.0, self.ecg.buffer_duration_s)
        self.plot_ecg.setYRange(MIN_INPUT, MAX_INPUT)

    def set_graph_ranges_for_time_test(self) -> None:
        self.plot_ecg.autoRange()
        self.plot_ecg.setXRange(0.0, self.ecg.buffer_duration_s)

    def on_continuous(self) -> None:
        self.ecg.sample_continuous()

    def on_capture_for_time(self) -> None:
        duration_s = self.ecg.buffer_duration_s
        n_samples = self.ecg.get_n_samples_from_duration(duration_s)
        self.ecg.sample_number(n_samples)

    def on_stop(self) -> None:
        self.ecg.quiet()

    def on_clear(self) -> None:
        self.ecg.clear_data()

    def on_ecg_mode(self) -> None:
        self.sine_freq_lbl.setEnabled(False)
        self.sine_freq_edit.setEnabled(False)
        self.ecg.set_source_analogue()
        self.set_graph_ranges_for_ecg()

    def on_time_mode(self) -> None:
        self.sine_freq_lbl.setEnabled(False)
        self.sine_freq_edit.setEnabled(False)
        self.ecg.set_source_time()
        self.set_graph_ranges_for_time_test()

    def on_sine_mode(
            self,
            default_freq_hz: float = DEFAULT_SINE_FREQUENCY_HZ) -> None:
        freq_hz = get_valid_float(
            self.sine_freq_edit.text(),
            default=default_freq_hz,
            minimum=MIN_FILTER_FREQ_HZ,
            maximum=MAX_FILTER_FREQ_HZ
        )
        self.sine_freq_lbl.setEnabled(True)
        self.sine_freq_edit.setEnabled(True)
        self.ecg.set_sine_frequency(frequency_hz=freq_hz)
        self.ecg.set_source_sine_generator()
        self.set_graph_ranges_for_ecg()

    def on_buffer_duration(self) -> None:
        buffer_duration_s = get_valid_float(
            self.buffer_duration_edit.text(),
            default=DEFAULT_ECG_DURATION_S,
            minimum=MIN_ECG_DURATION_S,
            maximum=MAX_ECG_DURATION_S
        )
        self.ecg.set_processing_options(
            buffer_duration_s=buffer_duration_s
        )

    def on_invert(self) -> None:
        self.ecg.set_processing_options(
            invert=self.invert_btn.isChecked(),
        )

    def trigger_ecg_update(self, with_analytics: bool = True) -> None:
        self.ecg.trigger_update(force_update=with_analytics)

    def on_centre(self) -> None:
        self.ecg.set_processing_options(
            centre_on_mean=self.centre_btn.isChecked()
        )

    def on_lowpass(self) -> None:
        lowpass_cutoff_freq_hz = get_valid_float(
            self.lowpass_cutoff_edit.text(),
            default=DEFAULT_LOWPASS_FREQ_HZ,
            minimum=MIN_FILTER_FREQ_HZ,
            maximum=MAX_FILTER_FREQ_HZ
        )
        lowpass_numtaps = get_valid_int(
            self.lowpass_numtaps_edit.text(),
            default=DEFAULT_LOWPASS_NUMTAPS,
            minimum=MIN_NUMTAPS
        )
        use_lowpass_filter = self.lowpass_btn.isChecked()
        use_highpass_filter = self.highpass_btn.isChecked()
        self.lowpass_cutoff_lbl.setEnabled(use_lowpass_filter)
        self.lowpass_cutoff_edit.setEnabled(use_lowpass_filter)
        self.lowpass_numtaps_lbl.setEnabled(
            use_lowpass_filter and not use_highpass_filter)
        self.lowpass_numtaps_edit.setEnabled(
            use_lowpass_filter and not use_highpass_filter)
        self.ecg.set_processing_options(
            use_lowpass_filter=use_lowpass_filter,
            lowpass_cutoff_freq_hz=lowpass_cutoff_freq_hz,
            lowpass_numtaps=lowpass_numtaps
        )

    def on_highpass(self) -> None:
        highpass_cutoff_freq_hz = get_valid_float(
            self.highpass_cutoff_edit.text(),
            default=DEFAULT_HIGHPASS_FREQ_HZ,
            minimum=MIN_FILTER_FREQ_HZ,
            maximum=MAX_FILTER_FREQ_HZ
        )
        highpass_numtaps = get_valid_int(
            self.highpass_numtaps_edit.text(),
            default=DEFAULT_HIGHPASS_NUMTAPS,
            minimum=MIN_NUMTAPS
        )
        use_lowpass_filter = self.lowpass_btn.isChecked()
        use_highpass_filter = self.highpass_btn.isChecked()
        self.highpass_cutoff_lbl.setEnabled(use_highpass_filter)
        self.highpass_cutoff_edit.setEnabled(use_highpass_filter)
        self.highpass_numtaps_lbl.setEnabled(use_highpass_filter)
        self.highpass_numtaps_edit.setEnabled(use_highpass_filter)
        self.lowpass_numtaps_lbl.setEnabled(
            use_lowpass_filter and not use_highpass_filter)
        self.lowpass_numtaps_edit.setEnabled(
            use_lowpass_filter and not use_highpass_filter)
        self.ecg.set_processing_options(
            use_highpass_filter=use_highpass_filter,
            highpass_cutoff_freq_hz=highpass_cutoff_freq_hz,
            highpass_numtaps=highpass_numtaps
        )

    def on_notch(self) -> None:
        notch_freq_hz = get_valid_float(
            self.notch_freq_edit.text(),
            default=DEFAULT_NOTCH_FREQ_HZ,
            minimum=MIN_FILTER_FREQ_HZ,
            maximum=MAX_FILTER_FREQ_HZ
        )
        notch_q = get_valid_float(
            self.notch_q_edit.text(),
            default=DEFAULT_NOTCH_Q,
            minimum=MIN_NOTCH_Q,
            maximum=MAX_NOTCH_Q
        )
        use_notch_filter = self.notch_btn.isChecked()
        self.notch_freq_lbl.setEnabled(use_notch_filter)
        self.notch_freq_edit.setEnabled(use_notch_filter)
        self.notch_q_lbl.setEnabled(use_notch_filter)
        self.notch_q_edit.setEnabled(use_notch_filter)
        self.ecg.set_processing_options(
            use_notch_filter=use_notch_filter,
            notch_freq_hz=notch_freq_hz,
            notch_quality_factor=notch_q
        )

    def quit_button(self) -> None:
        self.close()

    def view_advanced(self) -> None:
        ecg_info = self.app.get_ecg_info()
        if ecg_info is None:
            # noinspection PyCallByClass,PyArgumentList
            QMessageBox.warning(self, "Advanced plots", "Insufficient data")
            return
        aw = AdvancedWindow(ecg_info=ecg_info, parent=self)
        aw.exec_()  # modal


# =============================================================================
# ECG controller
# =============================================================================

class EcgController(QObject):
    # Signals
    data_changed = pyqtSignal(bool)  # parameter is: "force analytics update?"
    arduino_awake = pyqtSignal()

    SAVED_ATTRS = {  # with their defaults
        'times': [],
        'data': [],

        'person_name': '',
        'person_details': '',
        'person_comment': '',

        'start_datetime': None,
        'latest_datetime': None,

        'invert': DEFAULT_INVERT,
        'centre_on_mean': DEFAULT_CENTRE,

        'use_lowpass_filter': DEFAULT_USE_LOWPASS,
        'lowpass_cutoff_freq_hz': DEFAULT_LOWPASS_FREQ_HZ,
        'lowpass_numtaps': DEFAULT_LOWPASS_NUMTAPS,

        'use_highpass_filter': DEFAULT_USE_HIGHPASS,
        'highpass_cutoff_freq_hz': DEFAULT_HIGHPASS_FREQ_HZ,
        'highpass_numtaps': DEFAULT_HIGHPASS_NUMTAPS,

        'use_notch_filter': DEFAULT_USE_NOTCH,
        'notch_freq_hz': DEFAULT_NOTCH_FREQ_HZ,
        'notch_quality_factor': DEFAULT_NOTCH_Q,

        'sampling_freq_hz': SAMPLING_FREQ_HZ,
        'buffer_duration_s': DEFAULT_ECG_DURATION_S,
    }

    def __init__(self,
                 port_device: str,
                 baud: int,
                 buffer_duration_s: float = DEFAULT_ECG_DURATION_S,
                 sampling_freq_hz: float = SAMPLING_FREQ_HZ,
                 parent: QObject = None) -> None:
        super().__init__(parent=parent)

        self.updates_enabled = True

        # Processing options
        self.invert = False
        self.centre_on_mean = False
        
        self.use_lowpass_filter = True
        self.lowpass_cutoff_freq_hz = DEFAULT_LOWPASS_FREQ_HZ
        self.lowpass_numtaps = DEFAULT_LOWPASS_NUMTAPS

        self.use_highpass_filter = True
        self.highpass_cutoff_freq_hz = DEFAULT_HIGHPASS_FREQ_HZ
        self.highpass_numtaps = DEFAULT_HIGHPASS_NUMTAPS

        self.use_notch_filter = True
        self.notch_freq_hz = DEFAULT_NOTCH_FREQ_HZ
        self.notch_quality_factor = DEFAULT_NOTCH_Q

        # Sampling options
        self.sampling_freq_hz = sampling_freq_hz
        self.sampling_period_s = 1.0 / sampling_freq_hz
        self.buffer_duration_s = buffer_duration_s

        # Person
        self.person_name = ""
        self.person_details = ""
        self.person_comment = ""

        # Data
        self.start_datetime = None  # type: Pendulum
        self.latest_datetime = None  # type: Pendulum
        self.buffer_size = self.get_n_samples_from_duration(buffer_duration_s)
        self.cached_len = None  # type: int
        self.cached_time_axis = None  # type: np.ndarray
        self.times = []  # type: List[int]
        self.data = []  # type: List[int]
        self.clear_data()

        log.debug(
            "EcgController: "
            "sampling_freq_hz={sampling_freq_hz}, "
            "sampling_period_s={sampling_period_s}, "
            "buffer_duration_s={buffer_duration_s}, "
            "buffer_size={buffer_size}".format(
                sampling_freq_hz=sampling_freq_hz,
                sampling_period_s=self.sampling_period_s,
                buffer_duration_s=self.buffer_duration_s,
                buffer_size=self.buffer_size,
            ))

        # Set up the Arduino
        self.arduino_awake.connect(self.on_arduino_awake)
        try:
            self.port = Serial(
                port=port_device,
                baudrate=baud,
                bytesize=EIGHTBITS,
                parity=PARITY_NONE,
                stopbits=STOPBITS_ONE,
            )
        except SerialException as e:
            log.critical("Failed to open serial port {!r}:\n{}".format(
                port_device, e))
            self.port = None

    def on_arduino_awake(self) -> None:
        self.reset_sample_frequency()

    def get_n_samples_from_duration(self, duration_s: float) -> int:
        return int(duration_s / self.sampling_period_s) + 1

    def set_person_details(self, name: str, details: str,
                           comment: str) -> None:
        self.person_name = name
        self.person_details = details
        self.person_comment = comment

    def save_json(self, filename: str) -> None:
        d = {}  # type: Dict[str, Any]
        for attr in self.SAVED_ATTRS.keys():
            d[attr] = getattr(self, attr)
        log.info("Saving to {!r}".format(filename))
        with open(filename, "w") as f:
            f.write(json.dumps(d, sort_keys=True,
                               cls=JsonClassEncoder))
        log.info("... saved")

    def load_json(self, filename: str) -> bool:
        log.info("Loading from {!r}".format(filename))
        with open(filename) as f:
            j = f.read()
        d = json_decode(j)
        if d is None:
            return False
        for attr, default in self.SAVED_ATTRS.items():
            if attr not in d:
                log.warning(
                    "When loading from {!r}, attribute {!r} was "
                    "missing".format(filename, attr))
                value = default
            else:
                value = d.get(attr, default)
            setattr(self, attr, value)
        log.info("... loaded")

        # Ensure we have sensible values
        self.data = self.data or []

        # Reset other things
        self.sampling_period_s = 1.0 / self.sampling_freq_hz
        self.set_buffer_duration(self.buffer_duration_s)  # triggers update

        return True

    def set_processing_options(
            self,
            invert: bool = None,
            centre_on_mean: bool = None,
            use_lowpass_filter: bool = None,
            lowpass_cutoff_freq_hz: float = None,
            lowpass_numtaps: int = None,
            use_highpass_filter: bool = None,
            highpass_cutoff_freq_hz: float = None,
            highpass_numtaps: int = None,
            use_notch_filter: bool = None,
            notch_freq_hz: float = None,
            notch_quality_factor: float = None,
            trigger_update: bool = True,
            buffer_duration_s: float = None) -> None:

        if invert is not None:
            self.invert = invert
        if centre_on_mean is not None:
            self.centre_on_mean = centre_on_mean

        if use_lowpass_filter is not None:
            self.use_lowpass_filter = use_lowpass_filter
        if lowpass_cutoff_freq_hz is not None:
            self.lowpass_cutoff_freq_hz = lowpass_cutoff_freq_hz
        if lowpass_numtaps is not None:
            self.lowpass_numtaps = lowpass_numtaps

        if use_highpass_filter is not None:
            self.use_highpass_filter = use_highpass_filter
        if highpass_cutoff_freq_hz is not None:
            self.highpass_cutoff_freq_hz = highpass_cutoff_freq_hz
        if highpass_numtaps is not None:
            self.highpass_numtaps = highpass_numtaps

        if use_notch_filter is not None:
            self.use_notch_filter = use_notch_filter
        if notch_freq_hz is not None:
            self.notch_freq_hz = notch_freq_hz
        if notch_quality_factor is not None:
            self.notch_quality_factor = notch_quality_factor

        if buffer_duration_s is not None:
            self.set_buffer_duration(buffer_duration_s)

        if trigger_update:
            self.trigger_update(force_update=True)

    def get_settings_description(self) -> str:
        d = [
            "Sampling frequency: {} Hz.".format(self.sampling_freq_hz)
        ]
        if self.use_highpass_filter and self.use_lowpass_filter:
            d.append("Bandpass filter [{}, {}] Hz.".format(
                self.highpass_cutoff_freq_hz, self.lowpass_cutoff_freq_hz
            ))
        elif self.use_highpass_filter:
            d.append("High-pass filter [{}, +∞] Hz.".format(
                self.highpass_cutoff_freq_hz
            ))
        elif self.use_lowpass_filter:
            d.append("Low-pass filter [0, {}] Hz.".format(
                self.lowpass_cutoff_freq_hz
            ))
        if self.use_notch_filter:
            d.append("Notch filter removing {n} Hz.".format(
                n=self.notch_freq_hz
            ))
        return " ".join(d)

    def set_buffer_duration(self, duration_s: float) -> None:
        self.buffer_duration_s = duration_s
        self.buffer_size = self.get_n_samples_from_duration(duration_s)
        current_n = len(self.data)
        # log.debug(
        #     "set_buffer_duration({}): "
        #     "buffer_size {}, "
        #     "current_n {}".format(
        #         duration_s,
        #         self.buffer_size,
        #         current_n))
        if current_n > self.buffer_size:
            # lose old data, not new
            n_to_drop = current_n - self.buffer_size
            self.times = self.times[n_to_drop:]
            self.data = self.data[n_to_drop:]
            # ... if there are e.g. 5 too many, this does data[5:], which drops
            # 5 items
            self.cached_len = None  # type: int
            self.cached_time_axis = None  # type: np.ndarray
        self.trigger_update(True)

    def clear_data(self) -> None:
        # Don't store a bunch of zeroes; funny things happen with the filter.
        # Beware empty data: filters may crash.
        self.cached_len = None  # type: int
        self.cached_time_axis = None  # type: np.ndarray
        self.times = []  # type: List[int]
        self.data = []  # type: List[int]
        self.start_datetime = None
        self.latest_datetime = None
        self.trigger_update(True)

    def enable_updates(self, enable: bool) -> None:
        self.updates_enabled = enable

    def trigger_update(self, force_update: bool = False) -> None:
        if self.updates_enabled:
            self.data_changed.emit(force_update)

    def get_raw_data(self) -> List[int]:
        return self.data

    def get_time_axis_s(self) -> np.array:
        # times_s = np.array(self.times) / MICROSECONDS_PER_S
        # if len(times_s) > 0:
        #     times_s = times_s - times_s[0]
        # return times_s
        n_samples = len(self.data)
        if n_samples != self.cached_len:
            self.cached_len = n_samples
            if n_samples == 0:
                self.cached_time_axis = np.array([], dtype=DTYPE)
            else:
                self.cached_time_axis = np.linspace(
                    start=0.0,
                    stop=(n_samples - 1) * self.sampling_period_s,
                    num=n_samples,
                    dtype=DTYPE)
        return self.cached_time_axis

    def get_data(self) -> np.array:
        data = np.array(self.data, dtype=DTYPE)
        if not self.data:  # this truth-test doesn't work for np.array()
            return data  # filters will crash on empty data
        if self.invert:
            data = MAX_INPUT - data
        if self.centre_on_mean:
            data = data - np.mean(data)
        if self.use_lowpass_filter and not self.use_highpass_filter:
            try:
                data = lowpass_filter(
                    data=data,
                    sampling_freq_hz=self.sampling_freq_hz,
                    cutoff_freq_hz=self.lowpass_cutoff_freq_hz,
                    numtaps=self.lowpass_numtaps
                )
            except ValueError as e:
                log.warning("Low-pass filter error: {}".format(e))
        if self.use_highpass_filter and not self.use_lowpass_filter:
            try:
                data = highpass_filter(
                    data=data,
                    sampling_freq_hz=self.sampling_freq_hz,
                    cutoff_freq_hz=self.lowpass_cutoff_freq_hz,
                    numtaps=self.lowpass_numtaps
                )
            except ValueError as e:
                log.warning("High-pass filter error: {}".format(e))
        if self.use_lowpass_filter and self.use_highpass_filter:
            try:
                data = bandpass_filter(
                    data=data,
                    sampling_freq_hz=self.sampling_freq_hz,
                    lower_freq_hz=self.highpass_cutoff_freq_hz,  # highpass = "not lower"  # noqa
                    upper_freq_hz=self.lowpass_cutoff_freq_hz,  # lowpass = "not higher"  # noqa
                    numtaps=self.highpass_numtaps
                )
            except ValueError as e:
                log.warning("Band-pass filter error: {}".format(e))
        if self.use_notch_filter:
            try:
                data = notch_filter(
                    data=data,
                    sampling_freq_hz=self.sampling_freq_hz,
                    notch_freq_hz=self.notch_freq_hz,
                    quality_factor=self.notch_quality_factor
                )
            except ValueError as e:
                log.warning("Notch filter error: {}".format(e))
        return data

    def shutdown(self) -> None:
        self.port.close()

    def read_data(self, greedy: bool = True) -> None:
        if not self.port:
            return
        data_changed = False
        while self.port.inWaiting() > 0:
            # noinspection PyArgumentList
            binary_line = self.port.readline()
            try:
                line = binary_line.decode(SERIAL_ENCODING).strip()
            except UnicodeDecodeError:
                log.warning(
                    "Bad contents from Arduino: {!r}".format(binary_line))
                continue
            if line == RESP_HELLO:
                log.info("Arduino is awake")
                self.arduino_awake.emit()
            elif line == RESP_OK:
                pass
            elif line.startswith(PREFIX_DEBUG):
                log.debug("Arduino: " + line)
            elif line.startswith(PREFIX_INFO):
                log.info("Arduino: " + line)
            elif line.startswith(PREFIX_WARNING):
                log.warning("Arduino: " + line)
            elif line.startswith(PREFIX_ERROR):
                log.error("Arduino: " + line)
            elif not line:
                # You get the occasional blank line when it turns on.
                pass
            else:
                try:
                    time_str, value_str = line.split(",")
                    # log.critical(line)
                    time_microsec = int(time_str)
                    value = int(value_str)
                    if len(self.data) >= self.buffer_size:
                        # Buffer is full; slide old values along
                        self.times = self.times[1:] + [time_microsec]
                        self.data = self.data[1:] + [value]
                    else:
                        # Buffer is not full yet
                        self.times.append(time_microsec)
                        self.data.append(value)
                    data_changed = True
                except ValueError:
                    log.warning("Bad line: {!r}".format(line))
            if not greedy:
                break  # read only one at a time
        if data_changed:
            now = Pendulum.now()
            if self.start_datetime is None:
                self.start_datetime = now
            self.latest_datetime = now
            self.trigger_update(False)

    def send_command(self, cmd: str) -> None:
        if not self.port:
            return
        log.debug("EcgController: send_command: {}".format(cmd))
        self.port.write((cmd + CMD_TERMINATOR).encode(SERIAL_ENCODING))
        self.port.flush()

    def set_source_analogue(self) -> None:
        self.send_command(CMD_SOURCE_ANALOGUE)

    def set_source_sine_generator(self) -> None:
        self.send_command(CMD_SOURCE_SINE_GENERATOR)

    def set_source_time(self) -> None:
        self.send_command(CMD_SOURCE_TIME)

    def set_sine_frequency(self, frequency_hz: float) -> None:
        self.send_command("{} {}".format(CMD_SINE_GENERATOR_FREQUENCY,
                                         frequency_hz))

    def set_sample_frequency(self, frequency_hz: float) -> None:
        self.send_command("{} {}".format(CMD_SAMPLE_FREQUENCY, frequency_hz))

    def reset_sample_frequency(self) -> None:
        self.set_sample_frequency(self.sampling_freq_hz)

    def sample_continuous(self) -> None:
        self.reset_sample_frequency()
        self.send_command(CMD_SAMPLE_CONTINUOUS)

    def sample_number(self, n_samples: int) -> None:
        assert n_samples > 0
        self.reset_sample_frequency()
        self.send_command("{} {}".format(CMD_SAMPLE_NUMBER, n_samples))

    def quiet(self) -> None:
        self.send_command(CMD_QUIET)


# =============================================================================
# Application
# =============================================================================

class EcgApplication(QApplication):
    def __init__(
            self,
            argv: List[str],
            port_device: str,
            baud: int,
            update_ecg_every_ms: int = UPDATE_ECG_EVERY_MS,
            update_analytics_every_ms: int = UPDATE_ANALYTICS_EVERY_MS
    ) -> None:
        super().__init__(argv)
        self.ecg = EcgController(port_device=port_device, baud=baud,
                                 parent=self)
        self.mw = MainWindow(app=self)
        self.mw.show()
        self.timer = QTimer(parent=self)
        self.update_ecg_every_ms = update_ecg_every_ms
        self.update_analytics_every_ms = update_analytics_every_ms
        self.last_ecg_update_at = current_milli_time()
        self.last_analytics_update_at = self.last_ecg_update_at
        # noinspection PyUnresolvedReferences
        self.timer.timeout.connect(self.ecg.read_data)
        self.ecg.data_changed.connect(self.on_data_changed)
        self.ecg.arduino_awake.connect(self.mw.on_arduino_awake)
        self.start_capture()

    def exec_(self) -> int:
        retval = super().exec_()
        self.stop_capture()
        self.ecg.shutdown()
        return retval

    def start_capture(self, period_ms: int = POLLING_TIMER_PERIOD_MS) -> None:
        log.debug("EcgApplication: start_capture")
        self.timer.start(period_ms)

    def stop_capture(self) -> None:
        log.debug("EcgApplication: stop_capture")
        if self.timer.isActive():
            self.timer.stop()

    def get_ecg_info(self, ydata: np.array = None) -> Optional[ReturnTuple]:
        if ydata is None:
            ydata = self.ecg.get_data()
        try:
            ecg_info = biosppy_ecg(signal=ydata,
                                   sampling_rate=self.ecg.sampling_freq_hz,
                                   show=False)  # type: ReturnTuple
        except ValueError:  # as e:
            # Generally: not enough data (or not interpretable as ECG)
            # log.debug("Can't summarize ECG: {}".format(e))
            ecg_info = None
        return ecg_info

    def on_data_changed(self, force_update: bool = False) -> None:
        # log.debug("EcgApplication: on_data_changed")
        now = current_milli_time()

        delta_t_ecg = now - self.last_ecg_update_at
        delta_t_analytics = now - self.last_analytics_update_at

        update_ecg = delta_t_ecg > self.update_ecg_every_ms or force_update
        update_analytics = (
                delta_t_analytics > self.update_analytics_every_ms or
                force_update
        )

        if update_ecg or update_analytics:
            ydata = self.ecg.get_data()
        else:
            ydata = None  # for type checker

        if update_ecg:
            self.last_ecg_update_at = now
            xdata = self.ecg.get_time_axis_s()
            self.mw.update_ecg_graph(xdata, ydata)

        if update_analytics:
            # log.debug("Updating analytics")
            self.last_analytics_update_at = now
            ecg_info = self.get_ecg_info(ydata)
            self.mw.update_analysis(ecg_info)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    main_only_quicksetup_rootlogger(level=logging.DEBUG)

    available_ports = [portinfo.device for portinfo in sorted(comports())]
    if not available_ports:
        log.warning("No serial ports found! (Is the Arduino plugged in?)")

    parser = argparse.ArgumentParser(
        description="""
Primitive ECG application for Arduino.
By Rudolf Cardinal (rudolf@pobox.com).

(*) Written for the Arduino UNO.
    The accompanying file "ecg_arduino/src/ecg.cpp" should be built and 
    uploaded to the Arduino; the script "ecg_arduino/build_upload" does this
    automatically, and the script "term" allows you to interact with the 
    Arduino manually. Details of the protocol are in ecg.cpp.
(*) The Arduino code expects an ECG device producing inputs in the range 
    0 to +3.3V, on Arduino input pin A1.
(*) Uses a sampling frequency of {fs} Hz.
        """.format(fs=SAMPLING_FREQ_HZ),
        formatter_class=RawDescriptionArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD_RATE,
        help="Serial port baud rate"
    )
    parser.add_argument(
        "--port", type=str, choices=available_ports,
        default=available_ports[0] if available_ports else "",
        help="Serial port device name"
    )
    args = parser.parse_args()

    ecg_app = EcgApplication(argv=sys.argv,
                             port_device=args.port,
                             baud=args.baud)
    sys.exit(ecg_app.exec_())


if __name__ == '__main__':
    main()
