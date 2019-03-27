import math

import numpy as np

from src.models.material_model import MaterialModel


class ModifiedJohnsonCookModel(MaterialModel):

    @classmethod
    def labels(cls):
        return ['A1', 'n1', 'b1', 'b2', 'b3', 'L1', 'L2']

    @classmethod
    def params_scaling(cls):
        return np.array([1e9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def __call__(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params

        r_ref = 1e-3
        t_ref = 293.15
        t_melt = 1425 + 273.15
        t_h = (temperature - t_ref) / (t_melt - t_ref)
        r_h = strain_rate / r_ref

        A1 = parameters[0]
        n1 = parameters[1]
        b1 = parameters[2]
        b2 = parameters[3]
        b3 = parameters[4]
        L1 = parameters[5]
        L2 = parameters[6]

        str_rate_dep = b1 + strain * (b2 + strain * b3)
        temp_dep = L1 + L2 * strain

        return (A1 * (strain ** n1)) * (1 + str_rate_dep * math.log(r_h)) * math.exp(temp_dep * t_h)