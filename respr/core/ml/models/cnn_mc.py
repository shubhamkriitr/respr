import torch
from torch import nn, optim
import pytorch_lightning as pl
from respr.core.metrics import RMSELoss
from respr.util import logger
from respr.core.ml.models.util import ModelUtil
from respr.util.common import fill_missing_values
import pytorch_lightning as pl

class ResprMCDropoutCNN(nn.Module):
    def __init__(self, config={}) -> None:
        super().__init__()
        self.block_structure = self.get_block_structure()
        self._build()
    
    def _build(self):
        self.block_0 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=6,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=6),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_mu = nn.Linear(6, 1)

    def forward(self, x):
        z = self.block_0(x)
        z = torch.squeeze(z)
        mu = self.fc_mu(z)
        return mu
    
    def get_block_structure(self):
        return [
            (1, 64) , #conv2_x
            (1, 128) , #conv3_x
            (1, 256) , #conv4_x
            (1, 512)   #conv5_x
        ]

MODULE_CLASS_LOOKUP = {
    "ResprMCDropoutCNN": ResprMCDropoutCNN
}

class LitResprMCDropoutCNN(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        self._config = {}
        if "config" in kwargs:
            self._config = kwargs.pop("config")
        self._config = self._fill_missing_config_values()
        
        self.model_util = ModelUtil()
        super().__init__(*args, **kwargs)
        logger.info(f"Final config being used: {self._config}")
        model_module_class = self.resolve_model_module_class()
        self.model_module = model_module_class(
            self._config["module_config"])
        
        self.configure_y_normalization()
        self.metric_mae = nn.L1Loss(reduction="mean")
        self.metric_rmse = RMSELoss()
        self.respr_loss_name = ""
    
    def resolve_model_module_class(self):
        model_module_class_name = self._config["model_module_class"]
        model_module_class = MODULE_CLASS_LOOKUP[model_module_class_name]
        return model_module_class
    
    def configure_y_normalization(self):
        self.normalize_y = lambda x : x
        self.denormalize_y = lambda x : x
        self.denormalize_std = lambda x : x
        
        
    def _fill_missing_config_values(self):
        defaults = {
            # "cost_function": "mae"
            "optimization": {
                "lr": 1e-3,
                "weight_decay": 1e-4
            },
            "module_config": {}
        }
        for k, v in defaults.items():
            if k not in self._config:
                logger.warning(f"Key `{k}` not provided, using default"
                                f" value : {v}")
                self._config[k] = v
                
        return self._config
    
    def compute_loss(self, mu, y_true):
        delta = (y_true - mu)
        loss = (delta*delta)
        loss = torch.mean(loss)
        return loss
    
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model_module(x)
    
    def configure_optimizers(self):
        optim_config = self._config["optimization"]
        optimizer = optim.AdamW(self.parameters(), lr=optim_config["lr"],
                                weight_decay=optim_config["weight_decay"])
        return optimizer

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="train")

    def _shared_step(self, batch, step_name):
        x, labels = batch
        mu = self.model_module(x)
        mu = torch.squeeze(mu)
        loss = self.compute_loss(mu, self.normalize_y(labels))
        
        d_mu = self.denormalize_y(mu)
        mae = self.metric_mae(d_mu, labels)
        rmse = self.metric_rmse(d_mu, labels)
       
        
        self.log(f"{step_name}{self.respr_loss_name}_loss", loss)
        self.log(f"{step_name}_mae", mae)
        self.log(f"{step_name}_rmse", rmse)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="test")
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, _ = batch
        mu = self.model_module(x)
        y = self.denormalize_y(mu)
        
        std = 0 # TODO: do MC rollouts and compute std
        
        return y, std