from .mlp import LitResprVanillaMLP
from .cnn import LitResprResnet18
from respr.util.common import BaseFactory

ML_MODELS = {
    "LitResprVanillaMLP": LitResprVanillaMLP,
    "LitResprResnet18": LitResprResnet18
}

ML_FACTORY = BaseFactory({"resource_map": ML_MODELS})