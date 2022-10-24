class Const(object):
    
    def __init__(self, data) -> None:
        self._data = data
    
    def get(self, key):
        if key not in self._data:
            raise KeyError(f"Unknown attribute `{key}`")
        return self._data[key]

class DATASETS(Const):
    
    
    
    @property
    def BIDMC(self):
        return "BIDMC"

    
    