import math

import numpy as np

from src.models.material_model import MaterialModel


class ModifiedJohnsonCookModel(MaterialModel):

    @classmethod
    def _lower_bounds(cls):
        return np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])

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

    def derivatives(self, strain: float, strain_rate: float, temperature: float):
        parameters = self.params
        labels = self.labels()

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

        s_safe = max(strain, 1e-9)
        sr_safe = max(r_h, 1e-9)

        str_rate_dep = b1 + strain * (b2 + strain * b3)
        temp_dep = L1 + L2 * strain
        base_stress = (A1 * (strain ** n1))
        rate_dependent = (1 + str_rate_dep * math.log(r_h))
        thermal_dependent = math.exp(temp_dep * t_h)
        base_rate_component_derivative = base_stress * math.log(sr_safe) * thermal_dependent
        base_thermal_component_derivative = base_stress * rate_dependent * thermal_dependent * t_h

        derivatives = {
            labels[0]: (strain ** n1) * rate_dependent * thermal_dependent,
            labels[1]: math.log(s_safe) * base_stress * rate_dependent * thermal_dependent,
            labels[2]: base_rate_component_derivative,
            labels[3]: base_rate_component_derivative * strain,
            labels[4]: base_rate_component_derivative * (strain ** 2.0),
            labels[5]: base_thermal_component_derivative,
            labels[6]: base_thermal_component_derivative * strain
        }

        return derivatives
