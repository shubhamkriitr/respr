import torch
from torch import nn
from respr.util.common import fill_missing_values

class SummaryStatsModule(nn.Module):
    
    def __init__(self, config={}) -> None:
        super().__init__()
        self._config = config
        defaults = {
            "features": [
                "mean", # average pooling
                "max",
                "median",
                {
                    "quantiles" : [0.25, 0.75]
                },
                "std"
            ]
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
    def forward(self, x):
        """
        x should be of shape (batch, channels, feature_dim).
        
        
        """
        assert len(x.shape) == 3
        features = []
        
        # mean
        f_mean = torch.mean(x, dim=2, keepdim=True)
        features.append(f_mean)
        
        # median
        f_median = torch.quantile(input=x, q=0.5, dim=2, keepdim=True)
        features.append(f_median)
        
        # median abs deviation
        _madev = torch.abs(x - f_median).mean(dim=2, keepdim=True)
        features.append(_madev)
        
        # quantile 0.25
        q_1 = torch.quantile(input=x, q=0.25, dim=2, keepdim=True)
        features.append(q_1)
        
        # quantile 0.75
        q_3 = torch.quantile(input=x, q=0.75, dim=2, keepdim=True)
        features.append(q_3)
        
        # std
        stdev = torch.std(input=x, dim=2, keepdim=True)
        features.append(stdev)
        
        features = torch.concatenate(features, dim=1) # along channel dimension
        
        
        return features