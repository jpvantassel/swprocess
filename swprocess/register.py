"""Registry class definition."""

from abc import ABCMeta
import logging

logger = logging.getLogger(__name__)

class AbstractRegistry(ABCMeta):

    @classmethod
    def register(cls, name):

        def wrapper(class_to_wrap):
            if name in cls._register:
                logger.warning(f"Key {name} already exists, replacing ...")
            cls._register[name] = class_to_wrap
            return class_to_wrap

        return wrapper

    @classmethod
    def create_instance(cls, name, *args, **kwargs):
        instance = cls._register[name]
        return instance(*args, **kwargs)

class WavefieldTransformRegistry(AbstractRegistry):

    _register = {}

    def __init__(self, name):
        super.__init__(name)

class MaswWorkflowRegistry(AbstractRegistry):

    _register = {}

    def __init__(self, name):
        super.__init__(name)
