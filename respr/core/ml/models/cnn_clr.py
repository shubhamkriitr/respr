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
from respr.contrib.simclr import NT_Xent

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


MODULE_CLASS_LOOKUP = {
    "ResprMCDropoutCNNResnet18SimCLR": ResprMCDropoutCNNResnet18SimCLR
}
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
            "lambda_regression": 1.0,
            "batch_size": 16,
            "temperature": 0.5,
            "world_size": 1,
            # "model_save_step": 10 # every 10 epochs
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
        self._current_mode = None
        self._init_contrastive_loss_module()
    
    def _init_contrastive_loss_module(self):
        keys = ["batch_size", "temperature", "world_size"]
        kwargs = {k: self._config[k] for k in keys}
        self.contrastive_cost_func = NT_Xent(**kwargs)
        return self.contrastive_cost_func
    
    def sanity_checks(self):
        self.sanity_check_mode_schedule()
        assert hasattr(self.model_module, "get_embedding")
        assert hasattr(self.model_module, "project_embedding")
    
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
        epoch_in_range = False
        new_mode = self._config["default_mode"]
        for epoch_range, mode in mode_schedule:
            start_, end_ = epoch_range
            if start_ <= next_epoch <= end_:
                new_mode = mode
                epoch_in_range = True
                break
        
        self._current_mode = new_mode
        if not epoch_in_range:
            logger.warning(f"Mode not specified for epoch: {next_epoch}"
                           f". Using default mode `{self._current_mode}`")
        
        if last_mode != self._current_mode:
            self.handle_mode_change(current_mode=last_mode,
                                    next_mode=self._current_mode)
            
    def handle_mode_change(self, current_mode, next_mode):
        logger.info(f"Changing mode from `{current_mode}` to `{next_mode}`")
        
    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            self.dummy_log("train")
            self.dummy_log("val") # this is workaround for suppressing error
            # from model checkpoint TODO: address this
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
        batch = self.adapt_batch_for_regression(batch=batch)
        return super().predict_step(batch, batch_idx, dataloader_idx)
    
    def cont_training_step(self, batch, batch_idx):
        return self.cont_shared_step(batch, batch_idx, step_name="train")
    
    def cont_validation_step(self, batch, batch_idx):
        return self.cont_shared_step(batch, batch_idx, step_name="val")
    
    def cont_test_step(self, batch, batch_idx):
        return self.cont_shared_step(batch, batch_idx, step_name="test")
    
    def cont_shared_step(self, batch, batch_idx, step_name):
        
        x1, x2, _ = batch
        
        if x1.shape[0] != self._config["batch_size"]:
            logger.warning(f"Skipping this batch as batch size(x1.shape[0])  "
                           f" is != {self._config['batch_size']}")
            return torch.tensor(0., device=x1.device, requires_grad=True)
        
        h1 = self.model_module.get_embedding(x1)
        z1 = self.model_module.project_embedding(h1)
        
        h2 = self.model_module.get_embedding(x2)
        z2 = self.model_module.project_embedding(h2)
        
        contr_loss = self.contrastive_cost_func(z1, z2)
        
        self.log(f"{step_name}_contrastive_loss", contr_loss)
        
        return contr_loss
    
    def dummy_log(self, step_name):
        """This adds dummy value to the logs"""
        # TODO: remove this workaround
        self.log(f"{step_name}{self.respr_loss_name}_loss", 1000)
        self.log(f"{step_name}_mae", 1000)
        self.log(f"{step_name}_rmse", 1000)
        self.log(f"{step_name}_contrastive_loss", 1000)
        
        
    
    def resolve_model_module_class(self):
        model_module_class_name = self._config["model_module_class"]
        model_module_class = MODULE_CLASS_LOOKUP[model_module_class_name]
        return model_module_class
        
    
        

        
if __name__ == "__main__":
    model = ResprMCDropoutCNNResnet18SimCLR()
    print(model)