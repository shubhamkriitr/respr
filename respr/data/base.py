import numpy as np
SIGNAL_DTYPE = np.float64

class BaseDataAdapter:
    
    def __init__(self, config):
        self.config = config
        self.data_root_dir = self.config["data_root_dir"]
        self.signal_dtype = SIGNAL_DTYPE
    
    def inspect(self):
        raise NotImplementedError()
    
    def extract_subject_ids_from_file_names(self, file_names:list):
        raise NotImplementedError()
    
    def get_file_paths(self, id_):
        raise NotImplementedError()
    
    def get(self, id_):
        raise NotImplementedError()
    

class StandardDataRecord:
    def __init__(self, data) -> None:
        self._data = data
        
    def value(self):
        return self._data
    
    def get_t(self, signal_name):
        t_loc = self.get(f"_metadata/signals/{signal_name}/t_loc")
        t = self.get(t_loc)
        return t
        
    
    def get(self, key = ""):
        value = self._data
        for k in key.split("/"):
            value = value[k]
        return value

