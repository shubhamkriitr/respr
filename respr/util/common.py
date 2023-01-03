from datetime import datetime
from pathlib import Path
import os
import yaml
import pickle
from respr.util import logger
from pathlib import Path
import os

PROJECT_NAME = "respr"
PROJECT_ROOT = Path(os.path.abspath('')) / PROJECT_NAME

def save_pickle(output_path, data):
    with open(output_path, "wb") as f:
        pickle.dump(data, f, 
                    protocol=pickle.HIGHEST_PROTOCOL)

def get_timestamp_str(granularity=1000):
    if granularity != 1000:
        raise NotImplementedError()
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.dump(data, f)

def fill_missing_values(default_values: dict, target_container: dict,
                        warn=True):
    for k, v in default_values.items():
        if k not in target_container:
            if warn:
                logger.warning(f"Key {k} was not provided. Using default"
                               f" value : {v}")
            target_container[k] = v
    
    return target_container

    
            
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


class BaseVideoWriter:
    def __init__(self) -> None:
        self.ouput_path = None
        
    def open(self, file_path):
        self.ouput_path = Path(file_path)
        os.makedirs(file_path, exist_ok=False)
        self.frame_counter = 0
    
    def write(self, frame):
        frame_str = str(self.frame_counter).zfill(6)
        p = self.ouput_path / f"frame_{frame_str}.jpg"
        frame.savefig(p)
        self.frame_counter += 1
        
    def release(self):
        logger.info("OK")