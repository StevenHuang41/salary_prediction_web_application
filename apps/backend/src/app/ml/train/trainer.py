
from app.ml.models.linear import build_ridge
from app.ml.models.nn import build_MLP
from app.ml.models.tree import build_HGBR
from app.ml.train.compare import compare_model_family


models = {
    'linear': build_ridge,
    'tree': build_HGBR,
    'nn': build_MLP,
}

