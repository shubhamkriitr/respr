from .base import (Pipeline, Pipeline2, DatasetBuilder, TrainingPipeline,
                   IndexedDatasetBuilder, DummyIndexedDatasetBuilder)
from .train import TrainingPipelineSimCLR
REGISTERED_PIPELINES = {
    "Pipeline": Pipeline,
    "Pipeline2": Pipeline2,
    "DatasetBuilder": DatasetBuilder,
    "IndexedDatasetBuilder": IndexedDatasetBuilder,
    "DummyIndexedDatasetBuilder": DummyIndexedDatasetBuilder,
    "TrainingPipeline":TrainingPipeline,
    "TrainingPipelineSimCLR": TrainingPipelineSimCLR
    
}  