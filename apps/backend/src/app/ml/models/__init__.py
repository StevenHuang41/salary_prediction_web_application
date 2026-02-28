from .linear import (
    build_linear,
    build_ridge,
    build_lasso,
    build_elasticNet,
)
from .tree import (
    build_HGBR,
    build_xgb,
    build_xgbrf,
)
from .nn import build_MLP

MODEL_REGISTRY = {
    "linear": build_linear,
    "ridge": build_ridge,
    "lasso": build_lasso,
    "elasticNet": build_elasticNet,
    "HGBR": build_HGBR,
    "xgb": build_xgb,
    "xgbrf": build_xgbrf,
    "nn": build_MLP,
}

__all__ = [
    "MODEL_REGISTRY",
]
