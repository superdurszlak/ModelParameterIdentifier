import math

import numpy as np

from src.models.material_model import MaterialModel


class ZerilliArmstrongFCCModel(MaterialModel):

    @classmethod
    def labels(cls):
        return ['C2', 'C3', 'C4', 'C6']

    @classmethod
    def params_scaling(cls):
        return np.array([1e9, 1.0, 1.0, 1e9])

    def __call__(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params

        r_ref = 1e-3
        r_h = strain_rate / r_ref

        C2 = parameters[0]
        C3 = parameters[1]
        C4 = parameters[2]
        C6 = parameters[3]

        exponent = -C3 + C4 * math.log(r_h)
        return C2 * (strain ** 0.5) * math.exp(exponent) + C6