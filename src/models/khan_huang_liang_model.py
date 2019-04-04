import math
import numpy as np

from src.models.material_model import MaterialModel


class KhanHuangLiangModel(MaterialModel):

    @classmethod
    def labels(cls):
        return ['A', 'B', 'n0', 'n1', 'C', 'm']

    @classmethod
    def params_scaling(cls):
        return np.array([1e9, 1e9, 1, 1, 1, 1])

    def __call__(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params

        t_ref = 293.15
        t_melt = 1425 + 273.15
        t_h = (temperature - t_ref) / (t_melt - t_ref)
        D0 = 1e6
        D_log = math.log(D0)

        A = parameters[0]
        B = parameters[1]
        n0 = parameters[2]
        n1 = parameters[3]
        C = parameters[4]
        m = parameters[5]

        rate_exp = strain_rate ** C
        softening = (1 - t_h ** m)
        hardening = ((1 - (math.log(strain_rate) / D_log)) ** n1) * (strain ** n0)

        return (A + B * hardening) * softening * rate_exp
