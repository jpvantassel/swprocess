# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Import class definitions into the swprocess package namespace."""

import logging

from .meta import __version__
from .activetimeseries import ActiveTimeSeries
from .sensor1c import Sensor1C
from .source import Source
from .array1d import Array1D
from .masw import Masw
# from .spaccurve import SpacCurve
# from .spaccurvesuite import SpacCurveSuite
from .peaks import Peaks
from .peakssuite import PeaksSuite

logging.getLogger('swprocess').addHandler(logging.NullHandler())
