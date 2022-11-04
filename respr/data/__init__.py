
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

