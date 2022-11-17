from .mlp import LitResprVanillaMLP
from .cnn import LitResprResnet18, LitResprResnet18Small
from respr.util.common import BaseFactory

ML_MODELS = {
    "LitResprVanillaMLP": LitResprVanillaMLP,
    "LitResprResnet18": LitResprResnet18,
    "LitResprResnet18Small": LitResprResnet18Small
}

ML_FACTORY = BaseFactory({"resource_map": ML_MODELS})