import math
from typing import Type

import numpy as np
import pandas as pd

from src.models.material_model import MaterialModel


def goal_function(parameters: np.ndarray, data_frame: pd.DataFrame, material_model_class: Type[MaterialModel]):
    try:
        model = material_model_class(parameters)
        data_frame = data_frame.copy(deep=True)
        data_frame['comp_stress'] = data_frame.apply(
            lambda row: model(row['strain'], row['strain_rate'], row['temperature']),
            axis=1)
        data_frame['error'] = ((data_frame['comp_stress'] - data_frame['stress']) / data_frame['stress']) ** 2
        return data_frame['error'].mean()
    except OverflowError:
        return math.inf


def goal_function_derivatives(parameters: np.ndarray, data_frame: pd.DataFrame,
                              material_model_class: Type[MaterialModel]):
    model = material_model_class(parameters)

    labels = ["dF/d{}".format(key) for key in model.labels()]
    data_frame = data_frame.copy()

    def get_derivatives(row):
        args = (row['strain'], row['strain_rate'], row['temperature'])
        value = model(*args)
        expected = row['stress']
        derivatives = model.derivatives(*args)
        gf_derivatives = [
            2.0 * (value - expected) * derivative / (expected ** 2.0) for derivative in derivatives.values()
        ]
        return pd.Series(gf_derivatives)

    for label in labels:
        data_frame[label] = 0.0
    data_frame[labels] = data_frame.apply(get_derivatives, axis=1)
    return data_frame[labels].mean().to_dict()


