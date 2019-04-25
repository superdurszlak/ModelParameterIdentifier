import abc
from typing import Type

import numpy as np
from pandas import DataFrame

from src.models.material_model import MaterialModel


class BaseSensitivityAnalysis(abc.ABC):

    def __init__(self,
                 parameters: np.ndarray,
                 model: Type[MaterialModel],
                 max_deviation: np.ndarray,
                 samples: int = 50,
                 relative_deviations: bool = True,
                 minimum_sensitivity: float = 1e-3,
                 relative_sensitivity: bool = True):
        if parameters.shape != model.params_scaling().shape:
            raise ValueError("parameters' shape must match that of model's parameters")

        if samples < 1:
            raise ValueError("samples must be positive value")

        if max_deviation.shape != model.params_scaling().shape:
            raise ValueError("max_deviation's shape must match that of model's parameters")

        if max_deviation.min() < 0.0:
            raise ValueError("max_deviation must consist of non-negative values")

        if minimum_sensitivity <= 0.0:
            raise ValueError("minimum_sensitivity must be positive value")

        if relative_deviations and np.abs(parameters).min() == 0.0:
            raise ValueError("relative_deviations cannot be set to True if any of the parameters is zero")

        self._parameters = parameters
        self._model = model
        self._samples = samples

        self._max_deviation = max_deviation
        self._relative_deviations = relative_deviations

        self._minimum_sensitivity = minimum_sensitivity
        self._relative_sensitivity = relative_sensitivity

        self._completed = False

        self._success = False
        self._maximum_sensitivity = np.zeros(parameters.shape)
        self._deviation_at_minimum_sensitivity = max_deviation.copy()

        self._incomplete_analysis_error = "Analysis has not been carried out yet"
        self._reference_error = None

    @property
    def completed(self):
        return self._completed

    @property
    def success(self):
        if not self._completed:
            raise RuntimeError(self._incomplete_analysis_error)
        return self._success

    @property
    def maximum_sensitivity(self):
        if not self._completed:
            raise RuntimeError(self._incomplete_analysis_error)
        return np.abs(self._maximum_sensitivity)

    @property
    def threshold_sensitivity(self):
        return self._minimum_sensitivity

    @property
    def deviation_at_minimum_sensitivity(self):
        if not self.completed:
            raise RuntimeError(self._incomplete_analysis_error)
        return self._deviation_at_minimum_sensitivity

    def run(self, goal_function: callable, data: DataFrame):
        if self.completed:
            raise RuntimeError("Analysis is already completed")
        self._reference_error = goal_function(self._parameters, data, self._model)

        if self._relative_sensitivity:
            self._minimum_sensitivity = self._minimum_sensitivity * self._reference_error

        for deviation, index in self._get_deviations():
            relative_deviation = deviation

            if self._relative_deviations:
                relative_deviation = relative_deviation * self._parameters

            parameters = self._parameters + relative_deviation
            error = goal_function(parameters, data, self._model)
            sensitivity = error - self._reference_error

            if sensitivity > self._maximum_sensitivity[index]:
                self._maximum_sensitivity[index] = sensitivity

            if sensitivity >= self._minimum_sensitivity \
                    and abs(relative_deviation[index]) < abs(self._deviation_at_minimum_sensitivity[index]):
                self._deviation_at_minimum_sensitivity[index] = relative_deviation[index]

        self._success = self._maximum_sensitivity.min() >= self._minimum_sensitivity
        self._completed = True

    def _get_deviations(self):
        it = np.nditer(self._max_deviation, flags=['multi_index'])
        with it:
            while not it.finished:
                index = it.multi_index
                for matrix in self._get_deviation_matrices(index):
                    yield matrix, index
                    yield np.negative(matrix), index
                it.iternext()

    def _get_deviation_matrices(self, index):
        base_deviation = self._max_deviation[index]
        base_matrix = np.zeros(self._max_deviation.shape)
        for deviation_step in self._get_deviation_steps():
            matrix_copy = base_matrix.copy()
            matrix_copy[index] = base_deviation * deviation_step
            yield matrix_copy

    @abc.abstractmethod
    def _get_deviation_steps(self):
        pass
