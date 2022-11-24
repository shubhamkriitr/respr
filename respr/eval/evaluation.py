class BaseEvaluation:
    
    def __init__(self, config) -> None:
        self._config = config
        
    def run(self, *args, **kwargs):
        pass
    
class Evaluation(BaseEvaluation):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        
    
    def run(self, results_file_path):
        pass

class BaseResprEvaluator:
    def __init__(self, config={}):
        self._config = config
        self._prediction_prefix = "rr_est"
        self._std_prefix = "std_"
        
        possible_suffixes = ["riav", "rifv", "pnn", "fused", "riiv"]
        self._prediction_columns = [f"{self._prediction_prefix}{s}" for s in possible_suffixes]
        
    
    def vary_std_cutoff(self, predictions: pd.DataFrame, std_devs):
        # compute 1) percentage of windows retained
        # 2) MAE
        # 3) RMSE
        pass
    
    def compute_mae(self, df, gt_key, pred_keys):
        metrics = {}
        metric_name = "mae"
        for pred_k in pred_keys:
            k = f"[metric_{metric_name}]"+gt_key + ":" + pred_k
            metrics[k] = (df[gt_key] - df[pred_k]).abs().mean()
        return metrics

    def compute_rmse(self, df, gt_key, pred_keys):
        metrics = {}
        metric_name = "rmse"
        for pred_k in pred_keys:
            k = f"[metric_{metric_name}]"+gt_key + ":" + pred_k
            # metrics[k] = np.sqrt(((df[gt_key] - df[pred_k])**2).mean())
            metrics[k] = ((df[gt_key] - df[pred_k])**2).mean()**.5
        return metrics


    def add_record(self, cont, new_entry):
        for k in new_entry:
            try:
                cont[k].append(new_entry[k])
            except KeyError:
                cont[k] = [new_entry[k]]

        return cont

    def merge_records(self, cont, new_entry):
        for k in new_entry:
            try:
                cont[k] = cont[k] + new_entry[k]
            except KeyError:
                cont[k] = new_entry[k]

        return cont
    
    
    
    
        