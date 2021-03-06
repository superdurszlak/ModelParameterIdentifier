from src.models.johnson_cook_model import JohnsonCookModel
from src.models.khan_huang_liang_model import KhanHuangLiangModel
from src.models.modified_johnson_cook_model import ModifiedJohnsonCookModel
from src.models.zerilli_armstrong_bcc_model import ZerilliArmstrongBCCModel
from src.models.zerilli_armstrong_fcc_model import ZerilliArmstrongFCCModel

ALLOWED_MODELS = {
    'JC': JohnsonCookModel,
    'MJC': ModifiedJohnsonCookModel,
    'ZA-FCC': ZerilliArmstrongFCCModel,
    'ZA-BCC': ZerilliArmstrongBCCModel,
    'KHL': KhanHuangLiangModel
}

ALLOWED_METHODS = [
    'Nelder-Mead',
    'Powell',
    'BFGS',
    'PSO'
]

MAX_RESULTS = 5
