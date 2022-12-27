import torch
from torch import nn, optim
import pytorch_lightning as pl
from respr.core.metrics import RMSELoss
from respr.util import logger
from respr.core.ml.models.util import ModelUtil
from respr.util.common import fill_missing_values
import pytorch_lightning as pl

def get_conv_bn_relu_block(num_channels, num_out_channels, dropout_p=0.4,
                           dilations=[1, 1], paddings=[1, 1]):
    assert len(dilations) == 2
    assert len(paddings) == 2
    d1, d2 = dilations
    p1, p2 = paddings
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                  kernel_size=3, stride=1, padding=p1, bias=False,
                  dilation=d1),
        nn.BatchNorm1d(num_features=num_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=num_channels, out_channels=num_out_channels,
                  kernel_size=3, stride=1, padding=p2, bias=False,
                  dilation=d2),
        nn.BatchNorm1d(num_features=num_out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p)
    )
    
    return block

def get_one_conv_relu_block(num_channels, num_out_channels, dropout_p=0.4,
                            dilations=[1], paddings=[1]):
    assert len(dilations) == 1
    assert len(paddings) == 1
    d1 = dilations[0]
    p1 = paddings[0]
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_channels, out_channels=num_out_channels,
                  kernel_size=3, stride=1, padding=p1, bias=False,
                  dilation=d1),
        nn.BatchNorm1d(num_features=num_out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p)
    )
    
    return block

def get_first_block(num_in_channels, dropout_p=0.5):
    block = nn.Sequential(
        nn.Conv1d(in_channels=num_in_channels, out_channels=64,
                  kernel_size=7, stride=1, bias=False),
        nn.BatchNorm1d(num_features=64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(p=dropout_p)
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

class ResprMCDropoutCNNResnet18(nn.Module):
    
    
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
        z = self.get_embedding(x) # drop last dimension
        
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        
        return mu, log_var

    def get_embedding(self, x):
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
        z = torch.squeeze(z)
        return z
        

class ResprMCDropoutCNNResnet18Small(ResprMCDropoutCNNResnet18):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    def get_block_structure(self):
        return [
            (1, 64) , #conv2_x
            (1, 128) , #conv3_x
            (1, 256) , #conv4_x
            (1, 512)   #conv5_x
        ]
class _DebugResprMCDropoutCNN(nn.Module):
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


class ResprMCDropoutCNNResnet18v2(ResprMCDropoutCNNResnet18):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    def _build(self):
        super()._build()
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

# This lookup is to support config based resolution of model module classes
MODULE_CLASS_LOOKUP = {
    "_DebugResprMCDropoutCNN": _DebugResprMCDropoutCNN,
    "ResprMCDropoutCNNResnet18": ResprMCDropoutCNNResnet18,
    "ResprMCDropoutCNNResnet18Small": ResprMCDropoutCNNResnet18Small,
    "ResprMCDropoutCNNResnet18v2": ResprMCDropoutCNNResnet18v2
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
        loss = self.compute_loss(mu=mu, log_var=log_var,
                                 y_true=self.normalize_y(labels))
        
        d_mu = self.denormalize_y(mu)
        std = torch.sqrt(torch.exp(log_var))
        std = self.denormalize_std(std)
        mean_var = torch.mean(std*std)
        
        self._log_metrics(step_name, labels, d_mu, loss, mean_var)
        return loss

    def _log_metrics(self, step_name, labels, d_mu, loss,
                     mean_var=None):
        # `d_mu` is denormalized mu
        mae = self.metric_mae(d_mu, labels)
        rmse = self.metric_rmse(d_mu, labels)
       
        
        self.log(f"{step_name}{self.respr_loss_name}_loss", loss)
        self.log(f"{step_name}_mae", mae)
        self.log(f"{step_name}_rmse", rmse)
        
        if mean_var is not None:
            self.log(f"{step_name}_uncertainty/mean_var", mean_var)
    
    def _shared_val_and_test_step(self, batch, step_name):
        """Must not call this during training phase."""
        _, labels = batch
        y_final, uncertainty, extras = self._mc_rollout(batch)
        aleatoric_unc = torch.mean(extras["aleatoric"])
        epistemic_unc = torch.mean(extras["epistemic"])
        
        # normalizing `y_final` as it was denormalized during _mc_rollout call
        mu = self.normalize_y(y_final)
        # computing loss on final estimate (average of MC rollouts) (NOTE: 
        # during train step loss is computed on estimates from just one 
        # rollout )
        log_var = torch.log(extras["aleatoric"])
        mu = torch.squeeze(mu)
        log_var = torch.squeeze(log_var)
        loss = self.compute_loss(mu=mu, log_var=log_var,
                                 y_true=self.normalize_y(labels))
        
        self._log_metrics(step_name=step_name, labels=labels,
                          d_mu=y_final, loss=loss)
        
        
        # also log uncertainty (during val and test stage)
        mean_uncertainty = torch.mean(uncertainty)
        self.log(f"{step_name}_uncertainty", mean_uncertainty)
        
        
        self.log(f"{step_name}_uncertainty/aleatoric", aleatoric_unc)
        self.log(f"{step_name}_uncertainty/epistemic", epistemic_unc)
        
        return loss
        

    def validation_step(self, batch, batch_idx):
        return self._shared_val_and_test_step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self._shared_val_and_test_step(batch, step_name="test")
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        y_final, uncertainty, _ = self._mc_rollout(batch=batch)
        # y_final here is already denormalized (scaled properly)

        return y_final, uncertainty
    
    def _mc_rollout(self, batch):
        # Set just the dropout layers to train mode
        self.activate_dropout_layers(self.model_module)
        
        n_rollouts = self._config["num_monte_carlo_rollouts"]
        x, _ = batch
        
        y_buffer = []
        pred_std_buffer = []
        for _ in range(n_rollouts):
            mu, log_var = self.model_module(x)
            y = self.denormalize_y(mu)
            y_buffer.append(y)
            
            std = torch.sqrt(torch.exp(log_var))
            std = self.denormalize_std(std)
            pred_std_buffer.append(std)
        try:
            y_buffer = torch.concatenate(y_buffer, axis=1)
            pred_std_buffer = torch.concatenate(pred_std_buffer, axis=1)
        except IndexError:
            #IndexError happens in case y has shape torch.Size([1]) (when
            # batch contains just one sample (
            # e.g. x.shape -> torch.Size([1, 9600])))
            y_buffer = [torch.unsqueeze(a, 1) for a 
                        in y_buffer if len(a.shape) == 1]
            y_buffer = torch.concatenate(y_buffer, axis=1)
            
            pred_std_buffer = [torch.unsqueeze(a, 1) for a 
                                in pred_std_buffer if len(a.shape) == 1]
            pred_std_buffer = torch.concatenate(pred_std_buffer, axis=1)
            
        y_final = torch.mean(y_buffer, axis=1, keepdims=True)
        epistemic_uncertainty = torch.var(y_buffer, axis=1, keepdims=True)
        aleatoric_uncertainty = torch.mean(pred_std_buffer*pred_std_buffer,
                                           axis=1, keepdim=True)
        
        uncertainty = torch.sqrt(epistemic_uncertainty + aleatoric_uncertainty)

        # set model (along with droputs) back to eval mode
        self.model_module.eval()
        
        return y_final, uncertainty, {
            "aleatoric": aleatoric_uncertainty,
            "epistemic": epistemic_uncertainty
        }
    
    def activate_dropout_layers(self, model_module):
        for m in model_module.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def deactivate_dropout_layers(self, model_module):
        for m in model_module.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.eval()
    
    def get_embedding(self, x):
        
        
        z = self.model_module.get_embedding(x)

        z = self._adjust_embedding_shape(z)
        
        return z
    
    def _adjust_embedding_shape(self, z):
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, axis=0)
        return z