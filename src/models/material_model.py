import abc
import numpy as np


class MaterialModel(abc.ABC):

    def __init__(self, parameters: np.ndarray):
        if parameters.shape != self.params_scaling().shape:
            raise ValueError("Invalid number of parameters")
        self.__params = parameters

    @property
    def params(self):
        return self.__params * self.params_scaling()

    @classmethod
    @abc.abstractmethod
    def params_scaling(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def labels(cls):
        pass

    @property
    def json(self):
        return dict([(label, value) for label, value in zip(self.labels(), self.params)])

    @abc.abstractmethod
    def __call__(self, strain: float, strain_rate: float, temperature: float):
        pass
