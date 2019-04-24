import numpy as np

from src.sensitivity.base_sensitivity_analysis import BaseSensitivityAnalysis


class LinearSensitivityAnalysis(BaseSensitivityAnalysis):
    def _get_deviation_steps(self):
        return (x for x in np.linspace(1.0, 0.0, self._samples, endpoint=False))

