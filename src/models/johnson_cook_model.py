import math

import numpy as np

from src.models.material_model import MaterialModel


class JohnsonCookModel(MaterialModel):

    @classmethod
    def _upper_bounds(cls):
        return np.array([np.inf, np.inf, np.inf, np.inf, 1.0])

    @classmethod
    def _lower_bounds(cls):
        return np.array([0.0, 0.0, 0.0, -np.inf, 0.0])

    @classmethod
    def labels(cls):
        return ['A', 'B', 'n', 'C', 'm']

    @classmethod
    def params_scaling(cls):
        return np.array([1e9, 1e10, 1.0, 1.0, 1.0])

    def __call__(self, strain, strain_rate, temperature):
        parameters = self.params

        r_ref = 1e-3
        t_ref = 293.15
        t_melt = 1425 + 273.15
        t_h = (temperature - t_ref) / (t_melt - t_ref)
        r_h = strain_rate / r_ref

        A = parameters[0]
        B = parameters[1]
        n = parameters[2]
        C = parameters[3]
        m = parameters[4]

        return (A + B * (strain ** n)) * (1 + C * math.log(r_h)) * (1 - (t_h ** m))