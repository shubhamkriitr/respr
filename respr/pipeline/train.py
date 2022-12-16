from respr.core.ml.models.mlp import LitResprVanillaMLP
import pytorch_lightning as pl

from .base import TrainingPipeline

class TrainingPipelineSimCLR(TrainingPipeline):
    
    def __init__(self, config=...) -> None:
        super().__init__(config)
        
    def extract_ground_truth_from_batch(self, batch):
        y_true = batch[2]
        return y_true
    
    
    