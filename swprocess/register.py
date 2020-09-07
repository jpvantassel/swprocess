"""Registry class definition."""

from abc import ABC
import logging

logger = logging.getLogger(__name__)


class AbstractRegistry(ABC):

    @classmethod
    def register(cls, name):

        def wrapper(class_to_wrap):
            logging.info(f"Registering {name} ...")
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
