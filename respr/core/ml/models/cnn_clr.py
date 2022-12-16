from respr.core.ml.models.cnn_mc import (
    ResprMCDropoutCNNResnet18, LitResprMCDropoutCNN)
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
    
    def forward(self, x):
        z = self.get_embedding(x) # drop last dimension
        
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        
        return mu, log_var
    
    def project_embedding(self, z):
        p = self.projection_head(z)
        return p


class LitResprMCDropoutCNNSimCLR(LitResprMCDropoutCNN):
    MODE_CONTRASTIVE = "contrastive"
    MODE_REGRESSION = "regression"
    MODE_COMBINED = "combined"
    KEY_DEFAULT_MODE = "default_mode"
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._allowed_modes = [self.MODE_COMBINED, self.MODE_CONTRASTIVE,
                               self.MODE_REGRESSION]
        defaults = {
            self.KEY_DEFAULT_MODE: self.MODE_CONTRASTIVE,
            "mode_schedule": [
                
            ],
            "lambda_contrastive": 1.0,
            "lambda_regression": 1.0
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
        self._current_mode = None
    
    def sanity_check_mode_schedule(self):
        mode_schedule = self._config["mode_schedule"]
        for epoch_range, mode in mode_schedule:
            assert len(epoch_range) == 2
            assert all([isinstance(a, int) for a in epoch_range])
            assert mode in self._allowed_modes
        return True
    
    def switch_mode(self, next_epoch):
        mode_schedule = self._config["mode_schedule"]
        last_mode = self._current_mode
        for epoch_range, mode in mode_schedule:
            start_, end_ = epoch_range
            if start_ <= next_epoch <= end_:
                self._current_mode = mode
                break
        
        if last_mode != self._current_mode:
            self.handle_mode_change(current_mode=last_mode,
                                    next_mode=self._current_mode)
            
    def handle_mode_change(self, current_mode, next_mode):
        logger.info(f"Changing mode from `{current_mode}` to `{next_mode}`")
        
    def on_train_epoch_start(self) -> None:
        self.switch_mode(next_epoch=self.current_epoch)
        logger.info(f"Epoch: {self.current_epoch} started")
    
    def on_train_epoch_end(self) -> None:
        logger.info(f"Epoch: {self.current_epoch} ended")
    
    def adapt_batch_for_regression(self, batch):
        x, _, y = batch
        return x, y
    
    def training_step(self, batch, batch_idx):
        if self._current_mode == self.MODE_CONTRASTIVE:
            return self.cont_training_step(batch, batch_idx)
        
        batch = self.adapt_batch_for_regression(batch=batch)
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        if self._current_mode == self.MODE_CONTRASTIVE:
            return self.cont_validation_step(batch, batch_idx)
        
        batch = self.adapt_batch_for_regression(batch=batch)
        return super().validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        if self._current_mode == self.MODE_CONTRASTIVE:
            return self.cont_test_step(batch, batch_idx)
        
        batch = self.adapt_batch_for_regression(batch=batch)
        return super().test_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        return super().predict_step(batch, batch_idx, dataloader_idx)
    
        

        
if __name__ == "__main__":
    model = ResprMCDropoutCNNResnet18SimCLR()
    print(model)