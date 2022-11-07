from datetime import datetime

def get_timestamp_str(granularity=1000):
    if granularity != 1000:
        raise NotImplementedError()
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_")

class BaseFactory(object):
    def __init__(self, config=None) -> None:
        self.config = {} if config is None else config 
        self.resource_map = self.config["resource_map"] if "resource_map" in \
            self.config else {}
    
    def create(self, resource_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
 
        resource_class = self.get_uninitialized(resource_name)
        
        if config is not None:
            return resource_class(config=config)
        
        return resource_class(*args_to_pass, **kwargs_to_pass)

    def get_uninitialized(self, resource_name):
        try:
            return self.resource_map[resource_name]
        except KeyError:
            raise KeyError(f"{resource_name} is not allowed. Please use one of"
                           f" these names: {list(self.resource_map.keys())}")

    def get(self, resource_schema):
        name = resource_schema["name"]
        args = resource_schema["args"]
        kwargs = resource_schema["kwargs"]
        instance = self.create(name, config=None, args_to_pass=args,
                               kwargs_to_pass=kwargs)
        return instance
