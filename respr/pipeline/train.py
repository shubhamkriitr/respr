from respr.core.ml.models.mlp import LitResprVanillaMLP
import pytorch_lightning as pl

from .base import TrainingPipeline

class TrainingPipelineSimCLR(TrainingPipeline):
    
    def __init__(self, config=...) -> None:
        super().__init__(config)
    
    
    