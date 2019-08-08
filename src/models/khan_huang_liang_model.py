import math
import numpy as np

from src.models.material_model import MaterialModel


class KhanHuangLiangModel(MaterialModel):

    @classmethod
    def _upper_bounds(cls):
        return np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1.0])

    @classmethod
    def _lower_bounds(cls):
        return np.array([0.0, 0.0, -np.inf, -np.inf, 0.0, 0.0])

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

    def derivatives(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params
        labels = self.labels()

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

        s_safe = max(strain, 1e-9)
        sr_safe = max(strain_rate, 1e-9)
        th_safe = max(t_h, 1e-9)

        rate_exp = strain_rate ** C
        softening = (1 - t_h ** m)
        ln_diff = (1 - (math.log(sr_safe) / D_log))
        hardening = (ln_diff ** n1) * (strain ** n0)
        full_strain_hardening = (A + B * hardening)

        derivatives = {
            labels[0]: rate_exp * softening,
            labels[1]: rate_exp * softening * hardening,
            labels[2]: B * hardening * math.log(s_safe) * softening * rate_exp,
            labels[3]: B * (ln_diff ** n1) * math.log(ln_diff) * softening * rate_exp,
            labels[4]: full_strain_hardening * softening * rate_exp * math.log(sr_safe),
            labels[5]: - full_strain_hardening * rate_exp * (t_h ** m) * math.log(th_safe)
        }

        return derivatives
