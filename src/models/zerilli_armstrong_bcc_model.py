import math

import numpy as np

from src.models.material_model import MaterialModel


class ZerilliArmstrongBCCModel(MaterialModel):

    @classmethod
    def _lower_bounds(cls):
        return np.array([0.0, -np.inf, -np.inf, 0.0, 0.0, 0.0])

    @classmethod
    def labels(cls):
        return ['C1', 'C3', 'C4', 'C5', 'n', 'C6']

    @classmethod
    def params_scaling(cls):
        return np.array([1e9, 1e-2, 1e-2, 1e9, 1e-2, 1e9])

    def __call__(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params

        r_ref = 1e-3
        r_h = strain_rate / r_ref

        C1 = parameters[0]
        C3 = parameters[1]
        C4 = parameters[2]
        C5 = parameters[3]
        n = parameters[4]
        C6 = parameters[5]

        exponent = -C3 + C4 * math.log(r_h)
        return C1 * math.exp(temperature * exponent) + C6 + C5 * strain ** n