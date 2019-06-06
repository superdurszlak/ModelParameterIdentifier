import json
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from itertools import product

import numpy as np
import pandas as pd
import scipy.optimize as scopt

from src import config, arguments
from src.sensitivity.linear_sensitivity_analysis import LinearSensitivityAnalysis
from src.utils.goal_function import goal_function, goal_function_derivatives


def main():
    args = arguments.parser.parse_args()

    models_args = args.models
    methods = args.methods
    file_path = args.input[0]
    attempts = args.attempts[0]
    output = args.output[0]

    df = pd.read_csv(file_path, decimal=',')

    models = [v for k, v in config.ALLOWED_MODELS.items() if k in models_args]
    results_map = dict((model_class, []) for model_class in models)
    executor = ProcessPoolExecutor()

    result_dict = {}

    for cls, method, _ in product(models, methods, range(attempts)):
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
        results = sorted(results, key=lambda r: r.fun, reverse=False)[:min(config.MAX_RESULTS, len(results))]

        def result_mapper(result, index):
            model = cls(result.x)
            base_params = np.array(result.x)
            shape = base_params.shape
            deviations = np.ones(shape) * 0.1
            analysis = LinearSensitivityAnalysis(result.x, cls, deviations)
            analysis.run(goal_function, df.copy())
            filename = output.split(".")[0]
            filename = f"{filename}_{cls.__name__}_plot_{index}.png"
            analysis.plot(filename)
            return {
                'params': model.json,
                'vector': model.params.tolist(),
                'fitness': result.fun,
                'plot_file': filename,
                'sensitivity': analysis.maximum_sensitivity.tolist(),
                'threshold_sensitivity': analysis.threshold_sensitivity,
                'deviation_at_threshold_sensitivity': analysis.deviation_at_minimum_sensitivity.tolist(),
                'sensitivity_analysis_success': analysis.success,
                'goal_function_derivatives': goal_function_derivatives(result.x, df, cls)
            }

        result_dict[cls.__name__] = [result_mapper(r, i) for i, r in enumerate(results)]

    with open(output, 'w') as output:
        json.dump(result_dict, output)


if __name__ == "__main__":
    main()
