import abc
import numpy as np
import warnings


class MaterialModel(abc.ABC):

    # TODO: Implement analytical and/or numerical derivatives for each model parameter

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
    def _upper_bounds(cls):
        return np.ones_like(cls.params_scaling()) * np.inf

    @classmethod
    def _lower_bounds(cls):
        return np.ones_like(cls.params_scaling()) * (-np.inf)

    def is_within_bounds(self):
        params: np.ndarray = self.params
        upper: np.ndarray = self._upper_bounds()
        lower: np.ndarray = self._lower_bounds()
        within_upper = (upper >= params).all()
        within_lower = (lower <= params).all()
        return within_lower and within_upper

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

    def derivatives(self, strain: float, strain_rate: float, temperature: float):
        warnings.warn("Numerical central derivatives will be calculated, which is inefficient and may be unreliable "
                      "under certain circumstances. For more reliable results, override this method with analytical "
                      "formulas.")
        delta = 1e-9

        params = self.__params
        empty = np.zeros(params.shape)
        scaling = self.params_scaling()
        derivatives = {}
        args = (strain, strain_rate, temperature)
        for key, index in zip(self.labels(), range(params.shape[0])):
            deltas = empty.copy()
            deltas[index] = delta
            forward = params + deltas
            backward = params - deltas
            forward_model = self.__class__(forward)
            backward_model = self.__class__(backward)
            distance = scaling[index] * delta * 2.0
            derivative = (forward_model(*args) - backward_model(*args)) / distance
            derivatives[key] = derivative
        return derivatives
