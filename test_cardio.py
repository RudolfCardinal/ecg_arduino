#!/usr/bin/env python
# ecg_python/test_cardio.py

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
"""

import argparse
import logging
import os

from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger

from ecg_python.ecg import (
    DEFAULT_QTDB_DIR,
    DEFAULT_QT_PROCESSOR,
    EcgController,
)

log = logging.getLogger(__name__)


def test_cardio() -> None:
    main_only_quicksetup_rootlogger(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str
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
    log.critical("qtdb_dir: {!r}".format(args.qtdb_dir))

    c = EcgController(
        qtdb_dir=args.qtdb_dir,
        qt_pipeline_filename=args.qt_pipeline,
    )
    c.load_json(args.filename)
    i = c.get_biosppy_ecg_info()
    x = c.get_cardio_ecg_info(show_ecg=True, prefilter=True)
    # log.info("{!r}".format(i))
    # log.info("{!r}".format(x))


if __name__ == '__main__':
    test_cardio()
