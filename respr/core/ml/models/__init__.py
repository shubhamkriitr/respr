from .mlp import LitResprVanillaMLP
from .cnn import (LitResprResnet18, LitResprResnet18Small,
                  LitResprResnet18LinearScaledMeanHead,
                  LitResprResnet18ReLUMeanHead, LitResprResnet18v2)
from .cnn_mc import LitResprMCDropoutCNN, LitResprMCDropoutCNNOnlyRegressionCostFunc
from .cnn_clr import LitResprMCDropoutCNNSimCLR
from respr.util.common import BaseFactory

# these are pytorch lighting modules (internal model module has to configured
# separately. )
ML_MODELS = {
    "LitResprVanillaMLP": LitResprVanillaMLP,
    "LitResprResnet18": LitResprResnet18,
    "LitResprResnet18Small": LitResprResnet18Small,
    "LitResprMCDropoutCNN": LitResprMCDropoutCNN,
    "LitResprResnet18LinearScaledMeanHead":\
        LitResprResnet18LinearScaledMeanHead,
    "LitResprResnet18ReLUMeanHead": LitResprResnet18ReLUMeanHead,
    "LitResprMCDropoutCNNSimCLR": LitResprMCDropoutCNNSimCLR,
    "LitResprResnet18v2": LitResprResnet18v2,
    "LitResprMCDropoutCNNOnlyRegressionCostFunc":\
        LitResprMCDropoutCNNOnlyRegressionCostFunc
}

ML_FACTORY = BaseFactory({"resource_map": ML_MODELS})