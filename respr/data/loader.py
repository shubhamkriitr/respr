from torch.utils.data import Dataset, DataLoader
from respr.data.base import BaseDataAdapter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

DTYPE_FLOAT = np.float32
# TODO: set seeds for random 

def create_train_val_test_split(sample_ids: list, val=0.2, test=0.2):
    # the _junk suffix means that we drop that variable completely
    train_samples, val_samples = train_test_split(sample_ids, test_size=val)

    adjusted_fraction = test/(1 - val)
    train_samples, test_samples = \
        train_test_split(train_samples, test_size=adjusted_fraction) 
    
    return train_samples, val_samples, test_samples


class ResprDataset(Dataset):
    def __init__(self, config, datasource: BaseDataAdapter, sample_ids: list) -> None:
        """_summary_
        #TODO: complete doc string
        Args:
            config (_type_): _description_
            datasource (BaseDataAdapter): _description_
            sample_ids (list): list of sample ids to use
        """
        super().__init__()
        self._config = config
        self.datasource = datasource
        self.num_samples = len(sample_ids)
        self.sample_ids = sample_ids
        self.window_duration = 32
        self.window_step_duration = 1
        self.expected_feature_size = None
        self.cache = {}
        self.use_cache = True
        
    def initialize_context(self):
        
        counters = []
        for id_ in sample_ids:
            counters.append(
                {   "id": id_,
                    "i": 0,
                    "valid_patch": None
                }
            )
        
        self.context = {
            "counters" : counters
        }
        
    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        pass
    
    def get_next_window_info(self, index: int):
        pass
        


class BaseResprCsvDataset(Dataset):
    def __init__(self, config={}) -> None:
        super().__init__()
        self._config = config
        self._dataset_path = self._config["dataset_path"]
        self.num_samples = None
        self.initialize_source()
    
    def initialize_source(self):
        """Read the whole dataset or do setup to read the data from disk on
        demand. Also do an initial scan to setup size (`num_samples`) and
        index for index base access."""
        df = pd.read_csv(self._dataset_path)
        cols = df.keys()
        self.x_cols = sorted([c for c in cols if c.startswith("x_")])
        x_cols_idx = [int(c.split("x_")[1]) for c in self.x_cols]
        x_idx_0 = x_cols_idx[0]
        assert x_idx_0 == 0
        x_idx_max = x_cols_idx[-1]
        assert (x_idx_max + 1) == len(x_cols_idx)
        
        self.num_samples = df["y"].shape[0]
        
        self.data = df
         

    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        x, y = self.data.loc[index, self.x_cols], self.data.loc[index, "y"]
        x = x.to_numpy()
        
        return x, y
    
    
    
if __name__ == "__main__":
    
    ds = BaseResprCsvDataset({"dataset_path": \
        "../artifacts/2022-11-16_161146/dataset.csv"})
    
    dl = DataLoader(dataset=ds, batch_size=3)
    
    for batch in dl:
        b = batch
        print(type(b))
    sample_ids = [i for i in range(10)]
    train_samples, val_samples, test_samples = \
        create_train_val_test_split(sample_ids, 0.2, 0.2)
    print("Done")
    
        