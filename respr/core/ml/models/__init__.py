from .mlp import LitResprVanillaMLP
from .cnn import LitResprResnet18, LitResprResnet18Small
from .cnn_mc import LitResprMCDropoutCNN
from respr.util.common import BaseFactory

# these are pytorch lighting modules (internal model module has to configured
# separately. )
ML_MODELS = {
    "LitResprVanillaMLP": LitResprVanillaMLP,
    "LitResprResnet18": LitResprResnet18,
    "LitResprResnet18Small": LitResprResnet18Small,
    "LitResprMCDropoutCNN": LitResprMCDropoutCNN
}

ML_FACTORY = BaseFactory({"resource_map": ML_MODELS})