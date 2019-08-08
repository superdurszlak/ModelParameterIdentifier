import math
from typing import Type

import numpy as np
import pandas as pd

from src.models.material_model import MaterialModel


def goal_function(parameters: np.ndarray, data_frame: pd.DataFrame, material_model_class: Type[MaterialModel]):
    try:
        model = material_model_class(parameters)
        if not model.is_within_bounds():
            return math.inf
        data_frame = data_frame.copy(deep=True)
        data_frame['comp_stress'] = data_frame.apply(
            lambda row: model(row['strain'], row['strain_rate'], row['temperature']),
            axis=1)
        data_frame['error'] = ((data_frame['comp_stress'] - data_frame['stress']) / data_frame['stress']) ** 2
        return data_frame['error'].mean()
    except OverflowError:
        return math.inf
