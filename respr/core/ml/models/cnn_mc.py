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
        logger.warning(f"`activate_dropout_layers` method of this module"
            f" will set all sub modules whose class name starts with `Dropout`",
            f" to `train` mode during inference(prediction step). So"
            f" make sure to not name any other class starting with `Dropout`.")
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
        mu = self._config["y_normalization"]["y_mean"]
        std = self._config["y_normalization"]["y_std"]
        self.normalize_y = lambda x : (x - mu)/(std + 1e-7)
        self.denormalize_y = lambda x : mu + x*std
        self.denormalize_std = lambda x : x*std
        logger.debug("Y Normalization/Denormalization functions configured.")
        
        
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
        
        self._log_metrics(step_name, labels, d_mu, loss)
        return loss

    def _log_metrics(self, step_name, labels, d_mu, loss):
        # `d_mu` is denormalized mu
        mae = self.metric_mae(d_mu, labels)
        rmse = self.metric_rmse(d_mu, labels)
       
        
        self.log(f"{step_name}{self.respr_loss_name}_loss", loss)
        self.log(f"{step_name}_mae", mae)
        self.log(f"{step_name}_rmse", rmse)
    
    def _shared_val_and_test_step(self, batch, step_name):
        """Must not call this during training phase."""
        y_final, std = self._mc_rollout(batch)
        x, labels = batch
        
        # normalizing `y_final` as it was denormalized during _mc_rollout call
        mu = self.normalize_y(y_final)
        # computing loss on final estimate (average of MC rollouts) (NOTE: 
        # during train step loss is computed on estimates from just one 
        # rollout )
        loss = self.compute_loss(mu, self.normalize_y(labels))
        self._log_metrics(step_name=step_name, labels=labels,
                          d_mu=y_final, loss=loss)
        
        
        # also log uncertainty (during val and test stage)
        mean_std = torch.mean(std)
        self.log(f"{step_name}_uncertainty", mean_std)
        return loss
        

    def validation_step(self, batch, batch_idx):
        return self._shared_val_and_test_step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self._shared_val_and_test_step(batch, step_name="test")
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        y_final, std = self._mc_rollout(batch=batch)
        # y_final here is already denormalized (scaled properly)

        return y_final, std
    
    def _mc_rollout(self, batch):
        # Set just the dropout layers to train mode
        self.activate_dropout_layers(self.model_module)
        
        n_rollouts = self._config["num_monte_carlo_rollouts"]
        x, _ = batch
        
        y_buffer = []
        for _ in range(n_rollouts):
            mu = self.model_module(x)
            y = self.denormalize_y(mu)
            y_buffer.append(y)
        
        y_buffer = torch.concatenate(y_buffer, axis=1)
        y_final = torch.mean(y_buffer, axis=1, keepdims=True)
        std = torch.std(y_buffer, axis=1, keepdims=True)

        # set model (along with droputs) back to eval mode
        self.model_module.eval()
        
        return y_final, std
    
    def activate_dropout_layers(self, model_module):
        for m in model_module.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def deactivate_dropout_layers(self, model_module):
        for m in model_module.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.eval()