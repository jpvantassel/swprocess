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

"""Registry class definition."""

from abc import ABC
import logging

logger = logging.getLogger("swprocess.register")


class AbstractRegistry(ABC):

    @classmethod
    def register(cls, name):

        def wrapper(class_to_wrap):
            logger.info(f"Registering {name} ...")
            if name in cls._register:
                msg = f"Register entry {name} already exists, replacing ..."
                logger.warning(msg)
            cls._register[name] = class_to_wrap
            return class_to_wrap

        return wrapper

    @classmethod
    def create_instance(cls, name, *args, **kwargs):
        _class = cls.create_class(name)
        return _class(*args, **kwargs)

    @classmethod
    def create_class(cls, name):
        return cls._register[name]


class WavefieldTransformRegistry(AbstractRegistry):

    _register = {}


class MaswWorkflowRegistry(AbstractRegistry):

    _register = {}
