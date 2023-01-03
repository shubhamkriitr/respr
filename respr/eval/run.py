
from respr.pipeline.base import BasePipeline
from respr.util.common import logger
from respr.eval.evaluation import (TYPE_PNN,
    TYPE_SIGNAL_PROCESSING, TYPE_SIGNAL_PROCESSING_2, 
    TYPE_SIGNAL_PROCESSING_2B)
from respr.eval.result_loader import (ResultLoaderPnn, ResultLoaderSignalProc,
        ResultLoaderSignalProcOld, ResultLoaderSignalProcType2B)

DEFAULT_LOADER_MAPPING = {TYPE_PNN: ResultLoaderPnn(),
                          TYPE_SIGNAL_PROCESSING_2: ResultLoaderSignalProc(),
                          TYPE_SIGNAL_PROCESSING: ResultLoaderSignalProcOld(),
                          TYPE_SIGNAL_PROCESSING_2B: ResultLoaderSignalProcType2B()}

def gather_results_from_source(results_source, loaders=DEFAULT_LOADER_MAPPING):
    logger.info(f"Loader mapping: {loaders}")
    all_model_results = []
    loocv_fold_wise_metric = {}
    for source, type_code, tag in results_source:
        if type_code == TYPE_PNN:
            if tag in loocv_fold_wise_metric:
                logger.warning(f"Duplicate tag: {tag}")
            info_dict = {}
            data = loaders[type_code].load(source, result_container=info_dict)
            loocv_fold_wise_metric[tag] = info_dict
        else:
            data = loaders[type_code].load(source)
        all_model_results.append((data, type_code, tag))
    
    return all_model_results, loocv_fold_wise_metric
class EvalPipeline(BasePipeline):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    
    
    