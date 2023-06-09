from torch.utils.data import DataLoader
from respr.util.common import fill_missing_values, BaseFactory
import random
import numpy as np
from respr.util import logger
# Transformations


# MAKE SURE to not do any in place transformations (make a copy first instead)

class DataTransforms:
    def __init__(self, config={}) -> None:
        defaults = {
            "seed": 0
        }
        self._config = config
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
        self.rng = random.Random(self._config["seed"])
        
class ZeroOutEnds(DataTransforms):
    def __init__(self, config={}) -> None:
        super().__init__(config=config)
        self._warn_limit = 0.15
        defaults = {
            "front": 0.1, #float or int
            "end": 0.1
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        self.front = self._config["front"]
        self.end = self._config["end"]
        self._sanity_check()
    
    def _sanity_check(self):
        if not ((0 < self.front <  self._warn_limit ) and \
                    (0 < self.end < self._warn_limit)):
            raise ValueError(f"Value out of limits")

    
    def __call__(self, x, y=None):
        assert len(x.shape) == 1# 1D
        n = x.shape[0]
        idx_front_end, idx_end_start = self.get_indices(x)
        x = np.copy(x)
        if idx_front_end > 0:
            x[0:idx_front_end] = 0.
        if idx_end_start < n:
            x[idx_end_start:] = 0.
        
        return x
    
    def get_indices(self, x, y=None):
        n = x.shape[0]
        n_front = self.rng.uniform(a=0, b=self.front)*n
        n_end = self.rng.uniform(a=0, b=self.end)*n
        
        idx_front_end = max(0, int(n_front - 1))
        idx_end_start = min(int(n - n_end), n)
        
        return idx_front_end, idx_end_start

class TranslateTransform(DataTransforms):
    def __init__(self, config={}) -> None:
        super().__init__(config=config)
        self._warn_limit = 0.15
        defaults = {
            "translation": 0.1, #float or int
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        self.translation = self._config["translation"]
        self._sanity_check()
    
    def _sanity_check(self):
        
        if not (0 <= self.translation <  self._warn_limit ):
            raise ValueError(f"Value out of limits")

    
    def __call__(self, x, y=None):
        assert len(x.shape) == 1# 1D
        n = x.shape[0]
        p = self.rng.random()
        t = int(self.rng.uniform(a=0, b=self.translation)*n)

        x_new = np.zeros_like(x)
        if p < 0.5:
            # shift left
            x_new[0:n-t] = x[t:n]
        else:
            # shift right
            x_new[t:n] = x[0:n-t]
        
        return x_new
            
class ScaleTransform(DataTransforms):
    def __init__(self, config={}) -> None:
        super().__init__(config=config)
        
        defaults = {
            "upper": 1.1, #float
            "lower": 0.9
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        self.upper = self._config["upper"]
        self.lower = self._config["lower"]
        self._sanity_check()
    
    def _sanity_check(self):
        if not ((0.8 < self.lower <  self.upper ) and \
                    (self.upper < 1.2)):
            raise ValueError(f"Value out of limits")

    
    def __call__(self, x, y=None):
        assert len(x.shape) == 1# 1D
        scale = self.rng.uniform(a=self.lower, b=self.upper)
        x_new = x*scale
        return  x_new
    
class AbsMeanShiftTransform(DataTransforms):
    def __init__(self, config={}) -> None:
        super().__init__(config=config)
        
        defaults = {
            "upper": 0.2, #float  
            "lower": 0.0
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        self.upper = self._config["upper"]
        self.lower = self._config["lower"]
        self._sanity_check()
    
    def _sanity_check(self):
        if not ((0 <= self.lower <=  self.upper ) and \
                    (self.upper <= 0.25)):
            raise ValueError(f"Value out of limits")

    
    def __call__(self, x, y=None):
        assert len(x.shape) == 1# 1D
        scale = self.rng.uniform(a=self.lower, b=self.upper)
        mean_val = np.nanmean(np.abs(x))
        shift_by = mean_val*scale
        x_new = x + shift_by
        return  x_new

class BaseResprDataAugmentationComposerSimCLR(DataTransforms):
    
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.init_transforms()

    def init_transforms(self):
        # TODO : Make config based #FIXME
        self.transforms = [
            AbsMeanShiftTransform(),
            ZeroOutEnds(),
            TranslateTransform(),
            ScaleTransform()
        ]
        logger.info(f"transformation order: {self.transforms}")
    
    
    def __call__(self, x, y=None):
        x_1 = self.forward(x, y) # y not being used currently though
        x_2 = self.forward(x, y)
        return x_1, x_2, y
        
    
    def forward(self, x, y):
        for t in self.transforms:
            x = t(x, y)
        return x

class DataAugmentationSingleOutput(BaseResprDataAugmentationComposerSimCLR):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def __call__(self, x, y=None):
        x_1 = self.forward(x, y)
        return x_1, y 

DATA_AUG_CLASSES = {
    "BaseResprDataAugmentationComposerSimCLR": \
        BaseResprDataAugmentationComposerSimCLR,
    "DataAugmentationSingleOutput": DataAugmentationSingleOutput
}
DATA_AUG_FACTORY = BaseFactory(config={
    "resource_map": DATA_AUG_CLASSES
})