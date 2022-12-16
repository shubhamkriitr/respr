from respr.core.ml.models.cnn_mc import ResprMCDropoutCNNResnet18
from respr.util.common import fill_missing_values
import torch
from torch import nn, optim
import pytorch_lightning as pl
from respr.core.metrics import RMSELoss
from respr.util import logger
from respr.core.ml.models.util import ModelUtil
from respr.util.common import fill_missing_values
import pytorch_lightning as pl


class ResprMCDropoutCNNResnet18SimCLR(ResprMCDropoutCNNResnet18):
    
    def __init__(self, config={}) -> None:
        super().__init__(config)

        

    def _fill_default_config_items(self):
        defaults = {
            "projection_dim": 64
        }
        
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
    def _build(self):
        super()._build()
        self._fill_default_config_items()
        self._create_projection_head()
    
    def _create_projection_head(self):
        block_struct = super().get_block_structure()
        projection_dim = self._config["projection_dim"]
        
        n_features = block_struct[-1][1] # this is number of channels in the 
        # last layer before average pooling
        
        self.projection_head = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )
        
if __name__ == "__main__":
    model = ResprMCDropoutCNNResnet18SimCLR()
    print(model)