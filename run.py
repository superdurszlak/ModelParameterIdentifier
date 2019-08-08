import json
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from itertools import product

import numpy as np
import pandas as pd
import psopy
import scipy.optimize as scopt

from src import config, arguments
from src.plot import plot
from src.utils.goal_function import goal_function


def main():
    args = arguments.parser.parse_args()

    models_args = args.models
    methods = args.methods
    file_path = args.input[0]
    attempts = args.attempts[0]
    output = args.output[0]

    df = pd.read_csv(file_path, decimal=',')

    models = [v for k, v in config.ALLOWED_MODELS.items() if k in models_args]
    results_map = dict(((cls, method), []) for cls, method in product(models, methods))
    executor = ProcessPoolExecutor()

    result_dict = {}

    for cls, method in product(models, methods):
        if method == 'PSO':
            params = np.random.rand(attempts, cls.params_scaling().shape[0])
            future_result = executor.submit(
                fn=psopy.minimize,
                fun=goal_function,
                x0=params,
                args=(df.copy(), cls),
                tol=2.5e-3
            )
            results_map[(cls, method)].append(future_result)
        else:
            for _ in range(attempts):
                params = np.random.rand(cls.params_scaling().shape[0])
                future_result = executor.submit(
                    fn=scopt.minimize,
                    fun=goal_function,
                    x0=params,
                    args=(df.copy(), cls),
                    method=method,
                    tol=2.5e-3
                )
                results_map[(cls, method)].append(future_result)

    for cls in models:
        method_dict = {}
        for method in methods:
            print('Waiting for {} optimizations of model {} to complete'.format(method, cls.__name__))
            results = results_map[(cls, method)]
            wait(results, return_when=ALL_COMPLETED)
            results = [future_result.result() for future_result in results]
            results = sorted(results, key=lambda r: r.fun, reverse=False)[:min(config.MAX_RESULTS, len(results))]

            def result_mapper(result):
                model = cls(result.x)
                fitness = result.fun
                return {
                    'params': model.json,
                    'fitness': fitness,
                    'deviation_percentage': 100.0 * (fitness ** 0.5),
                    'method': method
                }

            best_result = results[0].x
            plot(df, cls(best_result), '{}_{}.png'.format(method, cls.__name__))
            method_dict[method] = [result_mapper(r) for r in results]
        result_dict[cls.__name__] = method_dict

    with open(output, 'w') as output:
        json.dump(result_dict, output)


if __name__ == "__main__":
    main()
