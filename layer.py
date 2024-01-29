import cupy as cp
import numpy as np


class Layer:
    """Base class for all layers"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Vanilla:
    def __call__(self, tensor):
        return cp.trace(tensor.reshape(1024, 1024))

class GlobbalAveragePooling:
    def __call__(self, tensor):
        return cp.mean(tensor)