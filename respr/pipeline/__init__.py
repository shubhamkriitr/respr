from .base import (Pipeline, Pipeline2, DatasetBuilder, TrainingPipeline,
                   IndexedDatasetBuilder, DummyIndexedDatasetBuilder)

REGISTERED_PIPELINES = {
    "Pipeline": Pipeline,
    "Pipeline2": Pipeline2,
    "DatasetBuilder": DatasetBuilder,
    "TrainingPipeline":TrainingPipeline,
    "IndexedDatasetBuilder": IndexedDatasetBuilder,
    "DummyIndexedDatasetBuilder": DummyIndexedDatasetBuilder
}  