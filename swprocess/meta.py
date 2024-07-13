# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Metadata for swprocess."""

__version__ = "0.3.0"

SUPPORTED_GEOPSY_VERSIONS = ["3.2.0"]

def check_geopsy_version(version):            
    if version not in SUPPORTED_GEOPSY_VERSIONS:
        msg = f"geopsy version {version} is not supported; "
        msg += f"use {SUPPORTED_GEOPSY_VERSIONS} instead."
        raise ValueError(msg)
