import torch
from torch import nn, optim
import pytorch_lightning as pl
from respr.core.metrics import RMSELoss
from respr.util import logger
from respr.core.ml.models.util import ModelUtil

CAPNOBASE_RR_MEAN = 18.8806
CAPNOBASE_RR_STD = 9.8441


def get_conv_bn_relu_block(num_channels, num_out_channels):
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                  kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm1d(num_features=num_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=num_channels, out_channels=num_out_channels,
                  kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm1d(num_features=num_out_channels),
        nn.ReLU(inplace=True)
    )
    
    return block

def get_one_conv_relu_block(num_channels, num_out_channels):
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_channels, out_channels=num_out_channels,
                  kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm1d(num_features=num_out_channels),
        nn.ReLU(inplace=True)
    )
    
    return block

def get_first_block(num_in_channels):
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_in_channels, out_channels=64,
                  kernel_size=7, stride=1, bias=False),
        nn.BatchNorm1d(num_features=64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )
    
    return block

def conv2_x_block(num_channels, num_sub_blocks, num_out_channels):
    class ResprResnetSubModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            
            self.blocks = nn.ModuleList()
            
            for i in range(num_sub_blocks):
                out_ch = num_channels
                if i == num_sub_blocks - 1:
                    out_ch = num_out_channels
                
                b = get_conv_bn_relu_block(num_channels, out_ch)
                self.blocks.append(b)
        
        def forward(self, x):
            z0 = x
            for block in self.blocks:
                z1 = block(z0)
                z0 = z1 + z0
            return z0

    return ResprResnetSubModule()

class ResprResnet18(nn.Module):
    
    
    def __init__(self, config={}) -> None:
        super().__init__()
        self.block_structure = self.get_block_structure()
        self._build()

    def get_block_structure(self):
        return [
            (2, 64) , #conv2_x
            (2, 128) , #conv3_x
            (2, 256) , #conv4_x
            (2, 512)   #conv5_x
        ]
        
        
    def _build(self):
        self.block_0 = get_first_block(1)
        self.blocks = [
                conv2_x_block(
                    num_channels=ch, num_sub_blocks=b, num_out_channels=ch)
                for b, ch in
                self.block_structure
            ]
        
        self.blocks = nn.ModuleList(self.blocks)
        
        bs = self.block_structure
        self.adjust_ch = [
            get_one_conv_relu_block(bs[i][1], bs[i+1][1]) 
            for i in range(len(bs)-1)
        ]
        
        self.adjust_ch = nn.ModuleList(self.adjust_ch)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(512, 1)
        self.fc_log_var = nn.Linear(512, 1)
    
    
    def forward(self, x):
        z = torch.unsqueeze(x, 1) # N x D -> N x 1 x D
        z = self.block_0(z)
        
        for i in range(len(self.blocks) - 1):
            b = self.blocks[i]
            z = b(z)
            z = self.adjust_ch[i](z)
        
        z = self.blocks[-1](z)
        
        z = self.avgpool(z)
        z = torch.squeeze(z) # drop last dimension
        
        mu = self.fc_mu(z)
        log_sig = self.fc_log_var(z)
        
        return mu, log_sig


class ResprResnet18Small(ResprResnet18):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    def get_block_structure(self):
        return [
            (1, 64) , #conv2_x
            (1, 128) , #conv3_x
            (1, 256) , #conv4_x
            (1, 512)   #conv5_x
        ]

def normalize_y(y):
    return (y - CAPNOBASE_RR_MEAN)/CAPNOBASE_RR_STD

def denormalize_y(y):
    return y*CAPNOBASE_RR_STD + CAPNOBASE_RR_MEAN

def denormalize_std(std):
    return std*CAPNOBASE_RR_STD

def lightning_wrapper(model_module_class):
    class _LitModule(pl.LightningModule):
        def __init__(self, *args, **kwargs) -> None:
            self._config = {}
            if "config" in kwargs:
                self._config = kwargs.pop("config")
            self._config = self._fill_missing_config_values()
            
            self.model_util = ModelUtil()
            super().__init__(*args, **kwargs)
            logger.info(f"Final config being used: {self._config}")
            self.model_module = model_module_class({})
            
            
            self.metric_mae = nn.L1Loss(reduction="mean")
            self.metric_rmse = RMSELoss()
            self.respr_loss_name = ""
            
        def _fill_missing_config_values(self):
            defaults = {
                # "cost_function": "mae"
                "optimization": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4
                }
            }
            for k, v in defaults.items():
                if k not in self._config:
                    logger.warning(f"Key `{k}` not provided, using default"
                                   f" value : {v}")
                    self._config[k] = v
                    
            return self._config
        
        def compute_loss(self, mu, log_var, y_true):
            delta = (y_true - mu)
            loss = (delta*delta)/torch.exp(log_var) + log_var
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
            mu, log_var = self.model_module(x)
            mu = torch.squeeze(mu)
            log_var = torch.squeeze(log_var)
            loss = self.compute_loss(mu, log_var, normalize_y(labels))
            
            d_mu = denormalize_y(mu)
            mae = self.metric_mae(d_mu, labels)
            rmse = self.metric_rmse(d_mu, labels)
            std = torch.sqrt(torch.exp(log_var))
            std = denormalize_std(std)
            mean_std = torch.mean(std)
            
            self.log(f"{step_name}{self.respr_loss_name}_loss", loss)
            self.log(f"{step_name}_mae", mae)
            self.log(f"{step_name}_rmse", rmse)
            self.log(f"{step_name}_uncertainty", mean_std)
            return loss

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, step_name="val")

        def test_step(self, batch, batch_idx):
            return self._shared_step(batch, step_name="test")
        
        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            x, _ = batch
            mu, log_var = self.model_module(x)
            y = denormalize_y(mu)
            
            std = torch.sqrt(torch.exp(log_var))
            std = denormalize_std(std)
            
            return y, std

    return _LitModule

LitResprResnet18 = lightning_wrapper(ResprResnet18)
LitResprResnet18Small = lightning_wrapper(ResprResnet18Small)

if __name__=="__main__":
    x = torch.zeros(size=(10, 9600))
    model = ResprResnet18()
    print(model)
    z = model(x)
    print(f"Done: {z}")
    
    