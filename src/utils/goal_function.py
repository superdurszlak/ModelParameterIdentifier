import numpy as np
import pandas as pd

from src.models.material_model import MaterialModel
from typing import Type


def goal_function(parameters: np.ndarray, data_fame: pd.DataFrame, material_model_class: Type[MaterialModel]):
    model = material_model_class(parameters)
    data_fame = data_fame.copy(deep=True)
    data_fame['comp_stress'] = data_fame.apply(
        lambda row: model(row['strain'], row['strain_rate'], row['temperature']),
        axis=1)
    data_fame['error'] = ((data_fame['comp_stress'] - data_fame['stress']) / data_fame['stress']) ** 2
    return data_fame['error'].mean()
