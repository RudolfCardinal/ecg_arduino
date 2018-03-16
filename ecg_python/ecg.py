#!/usr/bin/env python
# ecg_python/ecg.py

"""
===============================================================================
    Copyright (C) 2018-2018 Rudolf Cardinal (rudolf@pobox.com).

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.
===============================================================================

Based on:
    https://github.com/TDeagan/pyEKGduino
        ... manual measurement of QT
    https://github.com/pbmanis/EKGMonitor
        ... more sophisticated signal processing

"""

import argparse
from contextlib import contextmanager
from functools import partial
import json
import logging
import math
import os
from pdb import set_trace
import sys
# import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
from cardio.core.ecg_batch import EcgBatch
from cardio.core.ecg_dataset import EcgDataset
from cardio.dataset.dataset.named_expr import B
from cardio.dataset.dataset.dataset import Dataset
from cardio.dataset.dataset.pipeline import Pipeline
from cardio.dataset.dataset.dsindex import FilesIndex
from cardio.pipelines import (
    # hmm_predict_pipeline,
    hmm_preprocessing_pipeline,
    hmm_train_pipeline,
)
from cardio.models.hmm import HMModel, prepare_hmm_input

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
from pyqtgraph import InfiniteLine, PlotWidget
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
from scipy.io import wavfile
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
DEFAULT_BAUD_RATE = 921600  # see ecg.cpp, arduino_windows_notes.txt
MIN_INPUT = 0
MAX_INPUT = 1024  # https://www.gammon.com.au/adc

# Characteristic of our serial comms:
SERIAL_ENCODING = "ascii"

# Physical constants
MV_PER_V = 1000

# -----------------------------------------------------------------------------
# ECG capture
# -----------------------------------------------------------------------------

DEFAULT_SAMPLING_FREQ_HZ = 1000.0  # see command-line help text
DEFAULT_GAIN = 100  # as per AD8232 docs
MAX_OUTPUT_VOLTAGE_V = 3.3  # fixed in ecg.cpp
POLLING_TIMER_PERIOD_MS = 5
MIN_FILTER_FREQ_HZ = 0
MAX_FILTER_FREQ_HZ = DEFAULT_SAMPLING_FREQ_HZ
MIN_NUMTAPS = 1
MIN_NOTCH_Q = 1.0  # no idea
MAX_NOTCH_Q = 1000.0  # no idea
MIN_ECG_DURATION_S = 10.0
MAX_ECG_DURATION_S = 300.0

DEFAULT_ECG_DURATION_S = 30.0
DEFAULT_SINE_FREQUENCY_HZ = 1.0
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
JSON_FILE_SAVE_FILTER = "JSON Files (*.json)"
JSON_FILE_LOAD_FILTER = "JSON Files (*.json);;All files (*)"
UPDATE_ECG_EVERY_MS = 20  # 50 Hz
UPDATE_ANALYTICS_EVERY_MS = 500  # 0.5 Hz
MICROSECONDS_PER_S = 1000000
ECG_PEN = QPen(QColor(255, 0, 0))
ECG_PEN.setCosmetic(True)  # don't scale the pen width!
ECG_PEN.setWidth(1)
DATETIME_FORMAT = "%a %d %B %Y, %H:%M"  # e.g. Wed 24 July 2013, 20:04

P_COLOUR = QColor(0, 200, 0)
Q_COLOUR = QColor(200, 200, 0)
S_COLOUR = QColor(200, 0, 200)
T_COLOUR = QColor(0, 0, 200)
R1_COLOUR = QColor(200, 0, 0)
R2_COLOUR = QColor(200, 0, 0)
HR_COLOUR = QColor(200, 0, 0)

# -----------------------------------------------------------------------------
# Cardio annotation names
# -----------------------------------------------------------------------------

HMM_ANNOTATION = "hmm_annotation"  # fixed, I think; don't alter it
DEFAULT_QTDB_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir,
                 "physionet_data", "qtdb"))
DEFAULT_QT_PROCESSOR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir,
                 "physionet_data", "ecg_qt_processor.hmm"))


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


def get_save_filename(
        filetype_filter: str,
        default_ext: str,
        parent: QWidget = None,
        caption: str = "",
        directory: str = "",
        initial_filter: str = "",
        options: Union[QFileDialog.Options, QFileDialog.Option] = None) -> str:
    """
    Saving files / default extension via QFileDialog.getSaveFileName():

    We could add default extensions manually, but then we don't benefit from
    Qt's auto-checking to see if we mean that we want to overwrite something
    (or, it's a lot of work to check without double-checking).
    Qt is meant to provide us with a filename using the default extension if
    we preselect the filter using selectedFilter (or in PyQt, initialFilter);
    see
        https://stackoverflow.com/questions/7234381/
    ... but it doesn't.
    Known Qt 5 bug; see
        https://bugreports.qt.io/browse/QTBUG-27186
    Also:
        https://stackoverflow.com/questions/9822177/

    So let's deal with this properly:
    """
    if options is None:
        options = QFileDialog.Options()
    while True:
        # noinspection PyArgumentList
        filename, _ = QFileDialog.getSaveFileName(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=filetype_filter,
            initialFilter=initial_filter,
            options=options,
        )
        if not filename:
            return filename
        if not default_ext:
            return filename
        base, ext = os.path.splitext(filename)
        if ext:
            return filename
        # If we get here, the user has entered a filename with no extension but
        # the caller wants one
        filename = base + default_ext
        # But there is some possibility that we have bypassed the existence
        # check; e.g. the user types "thing", Qt checks that "thing" doesn't
        # exist, but have now renamed filename to "thing.pdf" and that does
        # exist.
        if not os.path.exists(filename):
            # OK, it doesn't exist.
            return filename
        # noinspection PyTypeChecker
        reply = QMessageBox.question(
            parent,
            "Confirm overwrite",
            "Are you sure you want to overwrite {!r}?".format(filename),
            QMessageBox.Yes,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            return filename
        # Otherwise, we cycle around the while loop.


def save_ecg_as_wav(data: np.array,
                    filename: str,
                    sampling_freq_hz: int) -> None:
    log.debug("Writing ECG as WAV file to {!r}".format(filename))
    wavfile.write(filename, sampling_freq_hz, data)


class CardioQtResult(object):
    def __init__(self, batch: EcgBatch) -> None:
        first_index = batch.index.indices[0]
        meta = batch[first_index].meta
        self.hr_bpm = meta["hr"]
        self.pq_s = meta["pq"]
        self.qrs_s = meta["qrs"]
        self.qt_s = meta["qt"]
        log.critical(meta.keys())

    @property
    def rr_s(self) -> float:
        return 60 / self.hr_bpm

    @property
    def qtc_s(self) -> float:
        return self.qt_s / math.sqrt(self.rr_s)


def get_cardio_ecg_batch_with_data_from_files(
        filespec: str,  # e.g. "/tmp/*.wav"; "/some/file.wav"
        index_no_ext: bool = False,
        fmt: str = "wav",
        sort: bool = True,
        pipeline: Pipeline = None,
        show_ecg: bool = False) -> EcgBatch:
    """
    Works now. However, performance not great. Specimen result:
        HR=165.74585635359117 s, PQ=0.078 s, QRS=0.072 s, QT=0.1145 s
    ... for a manual calculation of:
        HR 106 [biosppy]
        RR = 0.558 [manual] => HR 107 bpm for that beat
        QT = 0.29, QTc = 0.388
    So the CardIO analysis is unreliable.
    Its segmentation looks OK-ish, though imperfect; it's not crazily out, but
    it's definitely not good enough.
    """
    log.debug("Reading ECG from file(s) {!r} with format {}".format(
        filespec, fmt))
    index = FilesIndex(path=filespec, no_ext=index_no_ext, sort=sort)
    # log.warning("index.indices: {!r}".format(index.indices))
    first_index = index.indices[0]
    # log.warning("first_index: {!r}".format(index.indices))
    dataset = Dataset(index=index, batch_class=EcgBatch)
    if pipeline is not None:
        with Pipeline() as p:
            full_pipeline = (
                p.load(fmt=fmt, components=['signal', 'meta']) +
                pipeline
            )
        p2 = (dataset >> full_pipeline)  # type: Pipeline
        batch = p2.next_batch(batch_size=1)
        if show_ecg:
            batch.show_ecg(first_index, annot=HMM_ANNOTATION)
    else:
        # https://github.com/analysiscenter/cardio/blob/master/tutorials/III.Models.ipynb  # noqa
        batch = dataset.next_batch(batch_size=1)
        batch = batch.load(fmt=fmt, components=['signal', 'meta'])
        if show_ecg:
            batch.show_ecg(first_index)
    result = CardioQtResult(batch)
    log.info(
        "Calculated parameters: "
        "HR={hr} s, PQ={pq} s, QRS={qrs} s, QT={qt} s".format(
            hr=result.hr_bpm,
            pq=result.pq_s,
            qrs=result.qrs_s,
            qt=result.qt_s,
        )
    )
    # log.debug("batch={!r}; type={!r}".format(batch, type(batch)))
    return batch


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
                 biosppy_ecg_info: ReturnTuple,
                 abs_voltage: bool,
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
        self.ecg_info = biosppy_ecg_info
        t = biosppy_ecg_info['ts']  # type: np.ndarray
        y = biosppy_ecg_info['filtered']  # type: np.ndarray
        rr_peak_indices = biosppy_ecg_info['rpeaks']  # type: np.ndarray
        r_peak_t = np.take(t, rr_peak_indices)
        r_peak_y = np.take(y, rr_peak_indices)

        layout = QGridLayout()

        filtered_lbl = QLabel(
            "(Re-)Filtered ECG used for analysis, with R peaks")
        self.plot_filtered = PlotWidget(
            labels={
                "bottom": "time (s)",
                "left": "Filtered ECG signal ({})".format(
                    "mV" if abs_voltage else "arbitrary units")
            }
        )
        self.plot_filtered.plot(
            x=biosppy_ecg_info['ts'],
            y=biosppy_ecg_info['filtered']
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
            x=biosppy_ecg_info['heart_rate_ts'],
            y=biosppy_ecg_info['heart_rate']
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
            # pen=(200, 200, 200),
            pen=None,
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
            labels={
                "bottom": "time (s)",
                "left": "ECG signal ({})".format(
                    "mV" if self.ecg.abs_voltage else "arbitrary units")
            },
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

        hr_palette = QPalette()
        hr_palette.setColor(QPalette.Foreground, HR_COLOUR)

        hr_static_lbl = QLabel("Average heart rate (bpm)")
        self.hr_lbl = QLabel()
        self.hr_lbl.setPalette(hr_palette)
        
        # Manual analysis
        
        self.toggle_lines_btn = QCheckBox("Show manual lines")
        self.toggle_lines_btn.stateChanged.connect(self.on_toggle_lines)

        self.reset_lines_btn = QPushButton("Reset manual lines")
        self.reset_lines_btn.clicked.connect(self.reset_lines)

        self.p_pen = QPen(P_COLOUR, 0, Qt.SolidLine)
        self.p_line = self.plot_ecg.addLine(
            x=0, pen=self.p_pen, movable=True,
            label="P",
            labelOpts={
                "position": 0.98,
                "color": P_COLOUR,
            }
        )  # type: InfiniteLine
        self.p_line.sigPositionChangeFinished.connect(self.p_move)

        self.q_pen = QPen(Q_COLOUR, 0, Qt.SolidLine)
        self.q_line = self.plot_ecg.addLine(
            x=0.5, pen=self.q_pen, movable=True,
            label="Q",
            labelOpts={
                "position": 0.96,
                "color": Q_COLOUR,
            }
        )  # type: InfiniteLine
        self.q_line.sigPositionChangeFinished.connect(self.q_move)

        self.r1_pen = QPen(R1_COLOUR, 0, Qt.SolidLine)
        self.r1_line = self.plot_ecg.addLine(
            x=1, pen=self.r1_pen, movable=True,
            label="R1",
            labelOpts={
                "position": 0.94,
                "color": R1_COLOUR,
            }
        )  # type: InfiniteLine
        self.r1_line.sigPositionChangeFinished.connect(self.r1_move)

        self.s_pen = QPen(S_COLOUR, 0, Qt.SolidLine)
        self.s_line = self.plot_ecg.addLine(
            x=1.5, pen=self.s_pen, movable=True,
            label="S",
            labelOpts={
                "position": 0.92,
                "color": S_COLOUR,
            }
        )  # type: InfiniteLine
        self.s_line.sigPositionChangeFinished.connect(self.s_move)

        self.t_pen = QPen(T_COLOUR, 0, Qt.SolidLine)
        self.t_line = self.plot_ecg.addLine(
            x=2, pen=self.t_pen, movable=True,
            label="T",
            labelOpts={
                "position": 0.9,
                "color": T_COLOUR,
            }
        )  # type: InfiniteLine
        self.t_line.sigPositionChangeFinished.connect(self.t_move)

        self.r2_pen = QPen(R2_COLOUR, 0, Qt.SolidLine)
        self.r2_line = self.plot_ecg.addLine(
            x=2.5, pen=self.r2_pen, movable=True,
            label="R2",
            labelOpts={
                "position": 0.88,
                "color": R2_COLOUR,
            }
        )  # type: InfiniteLine
        self.r2_line.sigPositionChangeFinished.connect(self.r2_move)

        p_lbl = QLabel('P start = ')
        self.p_val_lbl = QLabel('0')
        p_palette = QPalette()
        p_palette.setColor(QPalette.Foreground, P_COLOUR)
        p_lbl.setPalette(p_palette)

        r1_lbl = QLabel('R1 peak = ')
        self.r1_val_lbl = QLabel('0')
        r1_palette = QPalette()
        r1_palette.setColor(QPalette.Foreground, R1_COLOUR)
        r1_lbl.setPalette(r1_palette)

        r2_lbl = QLabel('R2 peak = ')
        self.r2_val_lbl = QLabel('0')
        r2_palette = QPalette()
        r2_palette.setColor(QPalette.Foreground, R2_COLOUR)
        r2_lbl.setPalette(r2_palette)

        q_lbl = QLabel('Q (QRS start) = ')
        self.q_val_lbl = QLabel('0')
        q_palette = QPalette()
        q_palette.setColor(QPalette.Foreground, Q_COLOUR)
        q_lbl.setPalette(q_palette)

        s_lbl = QLabel('S (QRS end) = ')
        self.s_val_lbl = QLabel('0')
        s_palette = QPalette()
        s_palette.setColor(QPalette.Foreground, S_COLOUR)
        s_lbl.setPalette(s_palette)

        t_lbl = QLabel('T end = ')
        self.t_val_lbl = QLabel('0')
        t_palette = QPalette()
        t_palette.setColor(QPalette.Foreground, T_COLOUR)
        t_lbl.setPalette(t_palette)

        rr_lbl = QLabel('RR (s) = ')
        self.rr_val_lbl = QLabel('0')

        h_lbl = QLabel('Heart rate (bpm) = ')
        self.h_val_lbl = QLabel('0')

        pr_lbl = QLabel('PQ (PR) (s) = ')
        self.pr_val_lbl = QLabel('0')

        qrs_lbl = QLabel('QRS (s) = ')
        self.qrs_val_lbl = QLabel('0')

        qt_lbl = QLabel('QT (s) = ')
        self.qt_val_lbl = QLabel('0')

        qtc_lbl = QLabel('QTc, Bazett formula (s) = ')
        self.qtc_val_lbl = QLabel('0')

        manual_qt_label = QLabel(
            "MANUAL measurement of RR and QT intervals (drag the vertical "
            "lines; values ONLY reflect what you have set)")

        qtlayout = QGridLayout()
        qt_label_row_1 = 1
        qt_label_row_2 = 2
        qt_label_row_3 = 3
        qt_label_row_4 = 4
        qt_label_row_5 = 5
        qtlayout.addWidget(manual_qt_label, qt_label_row_1, 0, 1, 6)

        qtlayout.addWidget(r1_lbl, qt_label_row_2, 0, Qt.AlignRight)
        qtlayout.addWidget(self.r1_val_lbl, qt_label_row_2, 1, Qt.AlignLeft)
        qtlayout.addWidget(r2_lbl, qt_label_row_2, 2, Qt.AlignRight)
        qtlayout.addWidget(self.r2_val_lbl, qt_label_row_2, 3, Qt.AlignLeft)

        qtlayout.addWidget(p_lbl, qt_label_row_3, 0, Qt.AlignRight)
        qtlayout.addWidget(self.p_val_lbl, qt_label_row_3, 1, Qt.AlignLeft)
        qtlayout.addWidget(q_lbl, qt_label_row_3, 2, Qt.AlignRight)
        qtlayout.addWidget(self.q_val_lbl, qt_label_row_3, 3, Qt.AlignLeft)
        qtlayout.addWidget(s_lbl, qt_label_row_3, 4, Qt.AlignRight)
        qtlayout.addWidget(self.s_val_lbl, qt_label_row_3, 5, Qt.AlignLeft)
        qtlayout.addWidget(t_lbl, qt_label_row_3, 6, Qt.AlignRight)
        qtlayout.addWidget(self.t_val_lbl, qt_label_row_3, 7, Qt.AlignLeft)

        qtlayout.addWidget(rr_lbl, qt_label_row_4, 0, Qt.AlignRight)
        qtlayout.addWidget(self.rr_val_lbl, qt_label_row_4, 1, Qt.AlignLeft)
        qtlayout.addWidget(h_lbl, qt_label_row_4, 2, Qt.AlignRight)
        qtlayout.addWidget(self.h_val_lbl, qt_label_row_4, 3, Qt.AlignLeft)

        qtlayout.addWidget(pr_lbl, qt_label_row_5, 0, Qt.AlignRight)
        qtlayout.addWidget(self.pr_val_lbl, qt_label_row_5, 1, Qt.AlignLeft)
        qtlayout.addWidget(qrs_lbl, qt_label_row_5, 2, Qt.AlignRight)
        qtlayout.addWidget(self.qrs_val_lbl, qt_label_row_5, 3, Qt.AlignLeft)
        qtlayout.addWidget(qt_lbl, qt_label_row_5, 4, Qt.AlignRight)
        qtlayout.addWidget(self.qt_val_lbl, qt_label_row_5, 5, Qt.AlignLeft)
        qtlayout.addWidget(qtc_lbl, qt_label_row_5, 6, Qt.AlignRight)
        qtlayout.addWidget(self.qtc_val_lbl, qt_label_row_5, 7, Qt.AlignLeft)

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

        toggle_lines_row = summary_hr_row + 1
        layout.addWidget(self.toggle_lines_btn, toggle_lines_row, 0)
        layout.addWidget(self.reset_lines_btn, toggle_lines_row, 1)

        qt_summary_row = toggle_lines_row + 1
        n_qt_rows = 3
        layout.addLayout(qtlayout, qt_summary_row, 0, n_qt_rows, 5)

        self.setLayout(layout)
        self.set_defaults(trigger_ecg_update=False)
        self.on_toggle_lines()
        
    def on_toggle_lines(self) -> None:
        wanted = self.toggle_lines_btn.isChecked()
        self.p_line.setVisible(wanted)
        self.q_line.setVisible(wanted)
        self.r1_line.setVisible(wanted)
        self.s_line.setVisible(wanted)
        self.t_line.setVisible(wanted)
        self.r2_line.setVisible(wanted)

    def reset_lines(self) -> None:
        ax = self.plot_ecg.getAxis("bottom")
        left, right = ax.range
        width = right - left
        self.p_line.setValue(left + 0.05 * width)
        self.q_line.setValue(left + 0.1 * width)
        self.r1_line.setValue(left + 0.15 * width)
        self.s_line.setValue(left + 0.2 * width)
        self.t_line.setValue(left + 0.25 * width)
        self.r2_line.setValue(left + 0.3 * width)
        self.p_move()
        self.q_move()
        self.r1_move()
        self.s_move()
        self.t_move()
        self.r2_move()

    def p_move(self) -> None:
        p = self.p_line.value()
        self.p_val_lbl.setText(str("%.3f" % p))
        self.calc_qtc()

    def r1_move(self) -> None:
        r1 = self.r1_line.value()
        self.r1_val_lbl.setText(str("%.3f" % r1))
        self.calc_qtc()

    def r2_move(self) -> None:
        r2 = self.r2_line.value()
        self.r2_val_lbl.setText(str("%.3f" % r2))
        self.calc_qtc()

    def q_move(self) -> None:
        q = self.q_line.value()
        self.q_val_lbl.setText(str("%.3f" % q))
        self.calc_qtc()

    def s_move(self) -> None:
        s = self.s_line.value()
        self.s_val_lbl.setText(str("%.3f" % s))
        self.calc_qtc()

    def t_move(self) -> None:
        t = self.t_line.value()
        self.t_val_lbl.setText(str("%.3f" % t))
        self.calc_qtc()

    def calc_qtc(self) -> None:
        p = self.p_line.value()
        r1 = self.r1_line.value()
        r2 = self.r2_line.value()
        q = self.q_line.value()
        s = self.s_line.value()
        t = self.t_line.value()

        # R-R interval: from one R wave to the next (= interbeat interval)
        rr_s = r2 - r1
        self.rr_val_lbl.setText(str("%.3f" % rr_s))

        hr_bpm = 60 / rr_s
        self.h_val_lbl.setText(str("%.3f" % hr_bpm))

        # Q-T interval: from the start of the QRS complex to the end of the T wave  # noqa
        qt_s = t - q
        self.qt_val_lbl.setText(str("%.3f" % qt_s))

        # P-Q ("P-R") interval: from the start of the P to the start of the QRS
        pq_s = q - p
        self.pr_val_lbl.setText(str("%.3f" % pq_s))

        # QRS width: from the start of the QRS complex to its end
        qrs_s = s - q
        self.qrs_val_lbl.setText(str("%.3f" % qrs_s))

        try:
            # Bazett formula:
            qtc_s = qt_s / math.sqrt(rr_s)
            self.qtc_val_lbl.setText(str("%.3f" % qtc_s))
        except ValueError:
            self.qtc_val_lbl.setText("?")

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

        self.r1_move()
        self.r2_move()
        self.q_move()
        self.t_move()

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
            # noinspection PyTypeChecker
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
        filename = get_save_filename(
            caption="Save data as",
            filetype_filter=JSON_FILE_SAVE_FILTER,
            default_ext=".json"
        )
        if not filename:
            return
        self.ecg.save_json(filename)

    def on_load_data(self) -> None:
        # noinspection PyArgumentList
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Load data",
            filter=JSON_FILE_LOAD_FILTER,
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
            filename_filter = "PDF files (*.pdf)"
            default_ext = ".pdf"
        elif USE_SVG_EXPORT:
            filename_filter = "SVG files (*.svg)"
            default_ext = ".svg"
        else:
            filename_filter = "PNG files (*.png)"
            default_ext = ".png"
        filename = get_save_filename(
            caption="Save image as",
            filetype_filter=filename_filter,
            default_ext=default_ext,
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
            #
            # Note also that under Windows (but not so far under Linux!),
            # the image file remains open at the point the "with" statement
            # ends, and you get a crash with:
            #   PermissionError: [WinError 32] The process cannot access the
            #   file because it is being used by another process:
            #   'C:\\Users\\...\\AppData\\Local\\Temp\\tmpfcjs1fiv\\tmp.png'
            # See https://stackoverflow.com/questions/10996558/determining-if-an-image-is-closed-after-including-in-a-reportlab  # noqa

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
            del story  # } [story contains ecg]
            del ecg    # } make sure that file is closed...

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
        # Use
        biosppy_ecg_info = self.app.get_biosppy_ecg_info()
        if biosppy_ecg_info is None:
            # noinspection PyCallByClass,PyArgumentList
            QMessageBox.warning(self, "Advanced plots", "Insufficient data")
            return
        cardio_ecg_info = self.app.get_cardio_ecg_info()
        aw = AdvancedWindow(biosppy_ecg_info=biosppy_ecg_info,
                            abs_voltage=self.ecg.abs_voltage,
                            parent=self)
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

        'ecg_gain': DEFAULT_GAIN,

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

        'sampling_freq_hz': DEFAULT_SAMPLING_FREQ_HZ,
        'buffer_duration_s': DEFAULT_ECG_DURATION_S,
    }

    def __init__(self,
                 port_device: str = "",
                 baud: int = DEFAULT_BAUD_RATE,
                 abs_voltage: bool = False,
                 ecg_gain: float = DEFAULT_GAIN,
                 buffer_duration_s: float = DEFAULT_ECG_DURATION_S,
                 sampling_freq_hz: float = DEFAULT_SAMPLING_FREQ_HZ,
                 parent: QObject = None,
                 qtdb_dir: str = "",
                 qt_pipeline_filename: str = "") -> None:
        super().__init__(parent=parent)
        self.abs_voltage = abs_voltage
        self.ecg_gain = ecg_gain
        self.qtdb_dir = qtdb_dir
        self.qt_pipeline_filename = qt_pipeline_filename
        self._qt_segmentation_pipeline = None

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

    def get_data(self, debug: bool = True, prefilter: bool = True) -> np.array:
        data = np.array(self.data, dtype=DTYPE)
        if not self.data:  # this truth-test doesn't work for np.array()
            return data  # filters will crash on empty data
        if self.abs_voltage:
            # Convert to voltage in mV
            mv_per_int = (
                    MV_PER_V * MAX_OUTPUT_VOLTAGE_V /
                    (self.ecg_gain * MAX_INPUT)
            )
            if debug:
                log.critical("Raw data [{}-{}]: min {}, max {}".format(
                    MIN_INPUT, MAX_INPUT, np.min(data), np.max(data)))
                log.critical("mv_per_int: {}".format(mv_per_int))
            data = data * mv_per_int
            if debug:
                log.critical("In mV: min {}, max {}".format(
                    np.min(data), np.max(data)))
        if not prefilter:
            return data
        # Invert? Centre?
        if self.invert:
            data = MAX_INPUT - data
        if self.centre_on_mean:
            data = data - np.mean(data)
        # Filters
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
        if self.port is not None:
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

    def get_biosppy_ecg_info(self,
                             ydata: np.array = None) -> Optional[ReturnTuple]:
        """
        Parse an ECG through biosppy and return its analysis.
        http://biosppy.readthedocs.io/en/stable/
        """
        if ydata is None:
            ydata = self.get_data()
        try:
            ecg_info = biosppy_ecg(signal=ydata,
                                   sampling_rate=self.sampling_freq_hz,
                                   show=False)  # type: ReturnTuple
        except ValueError:  # as e:
            # Generally: not enough data (or not interpretable as ECG)
            # log.debug("Can't summarize ECG: {}".format(e))
            ecg_info = None
        return ecg_info

    def get_cardio_ecg_info(self, ydata: np.array = None,
                            show_ecg: bool = False,
                            prefilter: bool = True):
        """
        Parse an ECG through cardio
        
        https://github.com/analysiscenter/cardio
        https://medium.com/data-analysis-center/cardio-framework-for-deep-research-of-electrocardiograms-2a38a0673b8e
        https://github.com/analysiscenter/cardio/blob/master/tutorials/I.CardIO.ipynb
        """  # noqa
        if ydata is None:
            ydata = self.get_data(prefilter=prefilter)
        filename_stem = "ecg"
        filename_with_ext = filename_stem + ".wav"
        with TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename_with_ext)
            sampling_freq_hz = int(self.sampling_freq_hz)
            if sampling_freq_hz != self.sampling_freq_hz:
                log.warning(
                    "Rounding non-integer sampling frequency from {} "
                    "to {}".format(self.sampling_freq_hz, sampling_freq_hz))
            save_ecg_as_wav(data=ydata,
                            filename=filename,
                            sampling_freq_hz=sampling_freq_hz)
            batch = get_cardio_ecg_batch_with_data_from_files(
                filespec=filename,
                fmt="wav",
                # pipeline=None,
                pipeline=self.qt_segmentation_pipeline,
                show_ecg=show_ecg
            )
        return batch

    @property
    def qt_segmentation_pipeline(self):
        assert self.qt_pipeline_filename, "Need pipeline filename!"
        if self._qt_segmentation_pipeline is None:
            if not os.path.exists(self.qt_pipeline_filename):
                # -------------------------------------------------------------
                # Train one. This can be slow.
                # -------------------------------------------------------------
                # https://github.com/analysiscenter/cardio/blob/master/tutorials/III.Models.ipynb  # noqa
                log.warning("Training QT processing pipeline")
                signals_mask = os.path.join(self.qtdb_dir, "*.hea")
                log.debug("signals_mask: {!r}".format(signals_mask))
                qt_dataset = EcgDataset(path=signals_mask,
                                        no_ext=True, sort=True)
                pipeline = hmm_preprocessing_pipeline()
                ppl_inits = (qt_dataset >> pipeline).run()
                pipeline = hmm_train_pipeline(ppl_inits)
                ppl_train = (qt_dataset >> pipeline).run()
                # Now save it for next time
                log.info("Saving QT pipeline to {!r}".format(
                    self.qt_pipeline_filename))
                ppl_train.save_model("HMM", path=self.qt_pipeline_filename)

            # -----------------------------------------------------------------
            # Load a pre-trained pipeline
            # -----------------------------------------------------------------
            log.info("Loading QT pipeline from {!r}".format(
                self.qt_pipeline_filename))
            # Default method -- but the hmm_predict_pipeline() function creates
            # a pipeline that insists on "wfdb" format, and we want "wav".
            #
            # self._qt_segmentation_pipeline = hmm_predict_pipeline(
            #     self.qt_pipeline_filename, annot=HMM_ANNOTATION)
            #
            # Instead:
            model_path = self.qt_pipeline_filename
            batch_size = 20
            features = "hmm_features"
            channel_ix = 0
            annot = HMM_ANNOTATION
            model_name = "HMM"

            config_predict = {
                'build': False,
                'load': {'path': model_path}
            }
            self._qt_segmentation_pipeline = (
                Pipeline()
                    .init_model("static", HMModel, model_name,
                                config=config_predict)
                    .load(fmt="wav", components=["signal", "meta"])
                    .cwt(src="signal", dst=features, scales=[4, 8, 16],
                         wavelet="mexh")
                    .standardize(axis=-1, src=features, dst=features)
                    .predict_model(model_name,
                                   make_data=partial(prepare_hmm_input,
                                                     features=features,
                                                     channel_ix=channel_ix),
                                   save_to=B(annot), mode='w')
                    .calc_ecg_parameters(src=annot)
                    .run(batch_size=batch_size, shuffle=False, drop_last=False,
                         n_epochs=1, lazy=True)
            )

            log.info("... loaded")

        return self._qt_segmentation_pipeline


# =============================================================================
# Application
# =============================================================================

class EcgApplication(QApplication):
    def __init__(
            self,
            argv: List[str],
            port_device: str,
            baud: int,
            sampling_freq_hz: float,
            abs_voltage: bool,
            ecg_gain: float,
            update_ecg_every_ms: int = UPDATE_ECG_EVERY_MS,
            update_analytics_every_ms: int = UPDATE_ANALYTICS_EVERY_MS,
            qtdb_dir: str = "",
            qt_pipeline_filename: str = ""
    ) -> None:
        super().__init__(argv)
        self.ecg = EcgController(port_device=port_device,
                                 baud=baud,
                                 sampling_freq_hz=sampling_freq_hz,
                                 abs_voltage=abs_voltage,
                                 ecg_gain=ecg_gain,
                                 parent=self,
                                 qtdb_dir=qtdb_dir,
                                 qt_pipeline_filename=qt_pipeline_filename)
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

    def get_biosppy_ecg_info(self,
                             ydata: np.array = None) -> Optional[ReturnTuple]:
        """
        Parse an ECG through biosppy and return its analysis.
        """
        return self.ecg.get_biosppy_ecg_info(ydata)

    def get_cardio_ecg_info(self, ydata: np.array = None,
                            show_ecg: bool = False):
        """
        Parse an ECG through cardio
        """
        return self.ecg.get_cardio_ecg_info(ydata=ydata, show_ecg=show_ecg)

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
            ecg_info = self.get_biosppy_ecg_info(ydata)
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
Primitive ECG application for Arduino. By Rudolf Cardinal (rudolf@pobox.com).
Written for a UK primary school Science Week in March 2018.

(*) Written for the Arduino UNO. See the "ecg_arduino" directory.
    The accompanying file "src/ecg.cpp" should be built and uploaded to the 
    Arduino; the script "build_upload.sh" (Linux) or "build_upload.bat" 
    (Windows) does this automatically. The Linux script "term.sh" or a terminal 
    program (e.g. TeraTerm for Windows) allows you to interact with the Arduino 
    manually. Details of the protocol are in ecg.cpp.
(*) The Arduino code expects an ECG device producing inputs in the range 
    0 to +3.3V, on Arduino input pin A1.

Note also:

- The DFRobot ECG device [1] is a 3-electrode system using the AD8232 chip [2]. 
  The normal way of using three electrodes here is to use the leg electrode for 
  "common-mode rejection" [2, 3]. A differential amplifier is meant to take two 
  inputs V_pos and V_neg and produce an output voltage V_out that's a multiple 
  of the difference [4]: 
        V_out = gain_diff(V_pos - V_neg)
  where "gain_diff" is the differential again. However, the real output is
        V_out = gain_diff(V_pos - V_neg) + 0.5 gain_cm(V_pos + V_neg)  
  where "gain_cm" is the common-mode gain (usually much smaller than 
  gain_diff). One might imagine that the technique of common-mode rejection 
  would be to measure the common-mode signal (V_pos + V_neg) and remove it.
  Instead, it seems that the AD8232 measures the common-mode signal, via the
  right leg electrode, inverts it, and injects it back into the subject ("right 
  leg drive amplifier") [2, 3]. (Re safety: the whole thing is driven by a 5 V 
  supply, so this shouldn't be dangerous, and the AD8232 advises additional 
  safety measures [2] that one would hope the DFRobot device follows.)
   
- So I think that when the DFRobot device is properly connected, its signal
  is (left arm LA - right arm RA) = ECG lead I, and the "foot" (right leg, RL)
  electrode is used for the common-mode signal injection.

- To calculate absolute voltage: we can't use an Arduino reference signal.
  Arduinos can't generate a proper analogue output, or even a pulse-width-
  modulated (PWM) signal in the right voltage range, since the Arduino PWM 
  signal alternates between 0 +5V. So to know an absolute input voltage, we 
  need to know the DFRobot device's gain. It appears that the AD8232 gain is 
  exactly 100 [2]. Since the DFRobot's output range is 0 to +3.3V [1], then the 
  input range is 0 to +30 mV.
  HOWEVER, in practice I think it is not this. A real ECG produced ranges of:
        Arduino integer 266-636
        => Arduino input voltage 0.857 to 2.05 V, range ~1.2 V
        => if gain is 100, that means the input signal had a range of 0.012 V
           = 12 mV, and that is crazily big for an ECG; should be more like 
           1 mV, or maybe 3 mV, as per [5].
  Therefore, abs_voltage is set to false by default and you must be sure of 
  your gain to use it.
  
- Sampling rate: should be at least 1000 Hz for paediatric ECGs; see [5].
  
- EEG considerations: while the normal ECG is up to about 3 mV [5], the EEG
  alpha rhythm from scalp electrodes is of the order of 5 to 100 µV [6], so
  we're talking about a signal 1000 to 30 times smaller. The ECG signal from
  the Arduino spans a range of about 350 integer units, so we'd be talking 
  about an ECG signal spanning 0.3 to 12. Even the latter is probably poor, so
  let's ignore this for Science Week.
  
[1] https://www.dfrobot.com/wiki/index.php/Heart_Rate_Monitor_Sensor_SKU:_SEN0213
[2] http://www.analog.com/media/en/technical-documentation/data-sheets/AD8232.pdf
[3] http://www.ti.com/lit/an/sbaa188/sbaa188.pdf
[4] https://en.wikipedia.org/wiki/Common-mode_rejection_ratios
[5] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1769212/
[6] https://www.ncbi.nlm.nih.gov/pubmed/15036063 """,  # noqa
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
    parser.add_argument(
        "--sampling_freq_hz", type=float, default=DEFAULT_SAMPLING_FREQ_HZ,
        help=(
            "Sampling frequency (Hz); default is {f}, the Nyquist frequency "
            "for signals of interest of up to {half_f} Hz".format(
                f=DEFAULT_SAMPLING_FREQ_HZ,
                half_f=DEFAULT_SAMPLING_FREQ_HZ / 2
            )
        )
    )
    parser.add_argument(
        "--abs_voltage", action="store_true",
        help="Show absolute voltage, not arbitrary Arduino range. If you use "
             "this, you must be sure that your gain is set correctly."
    )
    parser.add_argument(
        "--gain", type=float, default=DEFAULT_GAIN,
        help="If abs_voltage is True: "
             "Gain (voltage multiple) from the ECG sensor (e.g. 0 to about 3 "
             "mV) to the Arduino (0 to +{}V)".format(MAX_OUTPUT_VOLTAGE_V)
    )
    parser.add_argument(
        "--qtdb_dir", type=str, default=DEFAULT_QTDB_DIR,
        help="Root directory of the Physionet 'qtdb' database of annotated "
             "QT intervals, to train our QT parser"
    )
    parser.add_argument(
        "--qt_pipeline", type=str, default=DEFAULT_QT_PROCESSOR,
        help="Saved QT processing pipeline"
    )
    args = parser.parse_args()

    ecg_app = EcgApplication(argv=sys.argv,
                             port_device=args.port,
                             baud=args.baud,
                             sampling_freq_hz=args.sampling_freq_hz,
                             abs_voltage=args.abs_voltage,
                             ecg_gain=args.gain,
                             qtdb_dir=args.qtdb_dir,
                             qt_pipeline_filename=args.qt_pipeline)
    sys.exit(ecg_app.exec_())


if __name__ == '__main__':
    main()
