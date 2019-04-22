from argparse import ArgumentParser

from src.config import ALLOWED_METHODS, ALLOWED_MODELS

parser = ArgumentParser(
    description='Material model parameters identification tool'
)

parser.add_argument('--models', nargs='+', choices=ALLOWED_MODELS.keys(),
                    help='Material models for which the parameters will be determined')
parser.add_argument('--methods', nargs='+', choices=ALLOWED_METHODS,
                    help='Methods which will be used for parameters identification')
parser.add_argument('--attempts', nargs=1, type=int, choices=range(3, 101),
                    help='Maximum number of attempts per single model-method pair', default=10)
parser.add_argument('--input', nargs=1, help='Path to CSV file with input data')
parser.add_argument('--output', nargs=1, help='Path to JSON for output data')
