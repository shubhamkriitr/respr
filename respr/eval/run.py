
from respr.pipeline.base import BasePipeline

from respr.eval.evaluation import (TYPE_PNN,
    TYPE_SIGNAL_PROCESSING, TYPE_SIGNAL_PROCESSING_2, 
    TYPE_SIGNAL_PROCESSING_2B)
from respr.eval.result_loader import (ResultLoaderPnn, ResultLoaderSignalProc,
        ResultLoaderSignalProcOld, ResultLoaderSignalProcType2B)

DEFAULT_LOADER_MAPPING = {TYPE_PNN: ResultLoaderPnn(),
                          TYPE_SIGNAL_PROCESSING_2: ResultLoaderSignalProc(),
                          TYPE_SIGNAL_PROCESSING: ResultLoaderSignalProcOld(),
                          TYPE_SIGNAL_PROCESSING_2B: ResultLoaderSignalProcType2B()}

class EvalPipeline(BasePipeline):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    
    
    