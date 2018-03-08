"""

    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1)
"""
import torch.nn as nn


class Model(nn.Module):
    """Base class for all models"""

    def __init__(self, params):
        super(Model, self).__init__()
        for param in ['name', 'input_shape', 'num_classes', 'phase', 'dtype']:
            assert param in params, "Missing mandatory neural net parameter '%s'" % param
        assert params['phase'] in ['inference', 'training'],\
               "Invalid phase: '%s'. Expecting 'inference' or 'training'" % (params['phase'])
        self.__name = params['name']
        self.__input_shape = params['input_shape']
        self.__num_classes = params['num_classes']
        self.__phase = params['phase']
        self.__dtype = params['dtype']

    @staticmethod
    def check_parameters(params, default_params):
        """Ensures `params` dictionary contains all keys in `default_params`

        Args:
            params (dict): Dictionary to check.
            default_params (dict): Values with these keys must present in `params`.
        """
        for param, value in default_params.items():
            if param not in params:
                params[param] = value

    @property
    def name(self):
        """Get model name"""
        return self.__name

    @property
    def input_shape(self):
        """Get input shape excluding batch size dimension"""
        return self.__input_shape if isinstance(self.__input_shape, tuple)\
               else (self.__input_shape,)

    @property
    def num_classes(self):
        """Get number of classes"""
        return self.__num_classes

    @property
    def phase(self):
        """Get current phase ('training' or 'inference')"""
        return self.__phase

    @property
    def dtype(self):
        """Get type of data ('float32' or 'float16' or 'int8')"""
        return self.__dtype
