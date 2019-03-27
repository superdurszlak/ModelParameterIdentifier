import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as scopt

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from itertools import product

from src.models.johnson_cook_model import JohnsonCookModel
from src.models.modified_johnson_cook_model import ModifiedJohnsonCookModel
from src.models.zerilli_armstrong_fcc_model import ZerilliArmstrongFCCModel
from src.utils.goal_function import goal_function


def main():
    df = pd.read_csv('data_exp.csv', decimal=',')

    models = [JohnsonCookModel, ModifiedJohnsonCookModel, ZerilliArmstrongFCCModel]
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    results_map = dict((model_class, []) for model_class in models)
    executor = ProcessPoolExecutor()

    trial_limit = math.ceil(10 / (len(methods) * len(models)))

    for cls, method, _ in product(models, methods, range(trial_limit)):
        params = np.random.rand(cls.params_scaling().shape[0])
        future_result = executor.submit(
            fn=scopt.minimize,
            fun=goal_function,
            x0=params,
            args=(df.copy(), cls),
            method=method,
            tol=2.5e-3
        )
        results_map[cls].append(future_result)

    for cls in models:
        print("Waiting for optimizations for model {} to complete".format(cls.__name__))
        results = results_map[cls]
        wait(results, return_when=ALL_COMPLETED)
        results = [future_result.result() for future_result in results]
        result = sorted(results, key=lambda r: r.fun, reverse=True).pop()
        final = cls(result.x)
        dfc = df.copy()
        dfc['comp_stress'] = dfc.apply(lambda row: final(row['strain'], row['strain_rate'], row['temperature']), axis=1)
        msg = "Best result achieved for model={} is res.fun={} for params: res.x={}".format(
            cls.__name__,
            result.fun,
            final.params
        )

        grouped = dfc.groupby(['strain_rate', 'temperature'])

        ncols = 3
        nrows = int(np.ceil(grouped.ngroups / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 18), sharey=True)

        for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
            grouped.get_group(key).plot.scatter(x="strain", y="comp_stress", s=1.0, c='red', ax=ax)
            grouped.get_group(key).plot.scatter(x="strain", y="stress", s=1.0, ax=ax)

        filename = "{}.png".format(cls.__name__)
        plt.savefig(filename)

        print(msg)
        print(final.json)


if __name__ == "__main__":
    main()
