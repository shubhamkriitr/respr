from .mlp import LitResprVanillaMLP
from respr.util.common import BaseFactory

ML_MODELS = {
    "LitResprVanillaMLP": LitResprVanillaMLP
}

ML_FACTORY = BaseFactory({"resource_map": ML_MODELS})