from respr.core.metrics import RMSELoss
from torch import nn

class ModelUtil:
    
    def __init__(self) -> None:
        pass
    
    
    def get_metric(self, name):
        if name == "rmse":
            return RMSELoss()
        if name == "mae":
            return nn.L1Loss(reduction="mean")
        
        raise ValueError(f"Unknown cost function: `{name}`")