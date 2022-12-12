import torch
from torch import nn, optim
import pytorch_lightning as pl
from respr.core.metrics import RMSELoss
from respr.util import logger
from respr.core.ml.models.util import ModelUtil
from respr.util.common import fill_missing_values
import torch.nn.functional as F
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
        self._config = config
        defaults = {
            "input_channels": 1,
            "force_reshape_input": False # try to reshape input records to 
            # get the desired number of input channels
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
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
        self.block_0 = get_first_block(self._config["input_channels"])
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
        if self._config["force_reshape_input"]:
            z = torch.reshape(x, 
                              (x.shape[0], self._config["input_channels"], -1))
        elif len(x.shape) == 2:
            z = torch.unsqueeze(x, 1) # N x D -> N x 1 x D
        else:
            z = x
        z = self.block_0(z)
        
        for i in range(len(self.blocks) - 1):
            b = self.blocks[i]
            z = b(z)
            z = self.adjust_ch[i](z)
        
        z = self.blocks[-1](z)
        
        z = self.avgpool(z)
        z = torch.squeeze(z) # drop last dimension
        
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        
        return mu, log_var


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



class ResprResnet18ReLUMeanHead(ResprResnet18):
    def __init__(self, config={}) -> None:
        super().__init__(config)
        
    def forward(self, x):
        mu, log_var = super().forward(x)
        mu = F.relu(mu)
        return mu, log_var


class ResprResnet18LinearScaledMeanHead(ResprResnet18):
    def __init__(self, config={}) -> None:
        super().__init__(config)
        default_values = {
            "mean_output_scaling": {
                #>>> "beta": 0.5, # to be used in sigmoid (1/(1 + exp(-beta*mu)))
                "unscaled_mu_start": -5,
                "unscaled_mu_end": 5,
                # breath 
                "min": 0.1, # minimum breath rate 
                "max": 90 # max. breath rate 
            }
        }
        self._config = fill_missing_values(default_values=default_values,
                                           target_container=self._config)
        
        self._scale_mean_output = self._create_mean_output_scaling()
        
    
    def _create_mean_output_scaling(self):
        
        c = self._config["mean_output_scaling"]
        #>>> beta = c["beta"] # if using sigmoid # TODO: clean up comments
        min_breath_rate = c["min"] # absolute min --> 0
        max_breath_rate = c["max"] # absolute max --> inf
        m_start = c["unscaled_mu_start"]
        m_end = c["unscaled_mu_end"]
        
        # if using sigmoid
        #>>> s = lambda mu: min_breath_rate + \
        #     torch.sigmoid(beta*mu)*(max_breath_rate - min_breath_rate)
        
        
        def s(mu):
            mu = torch.clamp(mu, min=torch.tensor(m_start).to(mu.device),
                max=torch.tensor(m_end).to(mu.device))
            
            mu = min_breath_rate + ((mu - m_start) / (m_end - m_start))\
                                    *(max_breath_rate - min_breath_rate)
            
            return mu
        
        #do sanity checks
        samples = [ (mu, s(torch.tensor(mu)))
                     for mu in range(-10, 10, 1)]
        logger.info(f"Sample (mu logit, breath rate) pairs: {samples}")
        
        
        return s
        
        
    def forward(self, x):
        unscaled_mu, log_var = super().forward(x)
        mu = self._scale_mean_output(unscaled_mu)
        
        return mu, log_var


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
            self.model_module = model_module_class(
                self._config["module_config"])
            
            
            self.metric_mae = nn.L1Loss(reduction="mean")
            self.metric_rmse = RMSELoss()
            self.respr_loss_name = ""
            
            self.compute_loss = self._init_loss_computation()
            
        def _fill_missing_config_values(self):
            defaults = {
                # "cost_function": "mae"
                "optimization": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4
                },
                "module_config": {},
                "weighted_loss": {
                    "do_weighted_loss": False,
                    "bin_step": 2, # in breaths/min , it indicates
                    # starting from 0 breath/min, bins of size 2 bpm will
                    # be used. Samples with ground truth respiratory rate
                    # in the same bin will be treated as belonging to the same
                    # class
                    
                    "max_bpm": 120,
                    
                    # `loss_weight` will be used only if `do_weighted_loss` is set.
                    # `loss_weights` is a list of `list of tuple(bin) and 
                    # float(weight)`
                    "loss_weights": []
                }
                
                
            }
            for k, v in defaults.items():
                if k not in self._config:
                    logger.warning(f"Key `{k}` not provided, using default"
                                   f" value : {v}")
                    self._config[k] = v
                    
            return self._config
        
        def _compute_loss(self, mu, log_var, y_true):
            delta = (y_true - mu)
            loss = (delta*delta)/torch.exp(log_var) + log_var
            loss = torch.mean(loss)
            return loss
        
        def _init_loss_computation(self):
            c = self._config["weighted_loss"]
            do_weighted = c["do_weighted_loss"]
            
            if not do_weighted:
                return self._compute_loss
            
            bin_step  = c["bin_step"] # breaths/min
            max_bpm = c["max_bpm"]
            loss_weights = c["loss_weights"]
            
            assert len(loss_weights) == max_bpm / bin_step
            
            self._loss_weights = torch.tensor(loss_weights)
            self._bin_step = bin_step
            
            def _get_batch_class_weights(y_true):
                binned_y = (y_true / self._bin_step).type(torch.int64)
                self._loss_weights.to(y_true.device)
                weights = self._loss_weights[binned_y]
                return weights
            
            def _compute_loss(mu, log_var, y_true):
                weights = _get_batch_class_weights(y_true=y_true)
                delta = (y_true - mu)
                loss = (delta*delta)/torch.exp(log_var) + log_var
                loss = weights * loss
                loss = torch.mean(loss)
                return loss
            
            return _compute_loss
            
        
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
LitResprResnet18LinearScaledMeanHead \
    = lightning_wrapper(ResprResnet18LinearScaledMeanHead)
LitResprResnet18ReLUMeanHead = lightning_wrapper(ResprResnet18ReLUMeanHead)

if __name__=="__main__":
    x = torch.zeros(size=(10, 9600))
    model = ResprResnet18()
    print(model)
    z = model(x)
    print(f"Done: {z}")
    
    