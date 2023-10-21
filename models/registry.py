import os
import inspect 
import types

from backbones import *
from necks import *
from heads import *
from losses import *


class Model:
    """
    Base class for model backbone, neck, head and loss
    """

    def __init__(self, config, type_module) -> None:
        self.list_module = [x[0] for x in os.walk('/home/models/')]
        self.config = config

        assert type_module in self.list_module

        self.list_model = [x[0]
                           for x in os.walk(f'/home/models/{type_module}')]
        name_model = config['TYPE']
        assert self.name_model in self.list_module

        module = f'{type_module}.{name_model}'

        # TODO
        # Current
        # check name of model exist
        # Next Step
        # Import Class in file
        # Get Args of class and match with file config
module = __import__('backbones.res2net')
print( module.__dict__.values())
classes = [print(getattr(module, x)) 
           for x in dir(module) if inspect.isclass(getattr(module, x))]
print(classes)

