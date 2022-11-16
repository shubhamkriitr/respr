import torch
from torch import nn, optim
import pytorch_lightning as pl

class ResprVanillaMLP(nn.Module):
    def __init__(self, config={}) -> None:
        super().__init__()
        self._config = config
        if "input_dim" not in config:
            self._config["input_dim"] = 32*300 # 32 sec window @ 300 Hz

        
        self.pre_embedding_dim = 125
        self.embedding_dim = self.pre_embedding_dim
        self._build_network()

            
    def _build_network(self):
        self._create_encoder()
        self._create_regression_head()

    def _create_encoder(self):
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(in_features=self._config["input_dim"], out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=600, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=600, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=600, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=600, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=600, out_features=250),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=250, out_features=250),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=125, out_features=125),
            nn.BatchNorm1d(num_features=125),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=125, out_features=self.pre_embedding_dim),
            nn.BatchNorm1d(num_features=self.pre_embedding_dim),
            nn.ReLU()  
        )

        
    def _create_regression_head(self):
        self.regression_head = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        
    def forward(self, x, return_latent_code=False):
       
        z = self.encoder(x)
       

        out_ = self.regression_head(z)
       
        return out_
    
    def extract_features(self, x):
        z = self.encoder(x)
        return z

# Lightning module

class LitResprVanillaMLP(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_module = ResprVanillaMLP({})
        
        self.loss_module = nn.MSELoss()
        self.metric_mae = nn.L1Loss(reduction="mean")
       
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model_module(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="train")

    def _shared_step(self, batch, step_name):
        x, labels = batch
        preds = self.model_module(x)
        loss = self.loss_module(preds, labels)
        mae = self.metric_mae(preds, labels)

        self.log(f"{step_name}_loss", loss)
        self.log(f"{step_name}_mae", mae)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, step_name="test")
    
    
    
    