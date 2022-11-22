from respr.util.common import BaseFactory
from .bidmc import BidmcDataAdapter
from .capnobase import CapnobaseDataAdapter, CapnobaseMatDataAdapter
from .loader import (BaseResprCsvDataset, ResprDataLoaderComposer,
                     ResprCsvDataLoaderComposer)

COMPONENTS_MAP = {
    "BidmcDataAdapter": BidmcDataAdapter,
    "CapnobaseDataAdapter": CapnobaseDataAdapter,
    "CapnobaseMatDataAdapter": CapnobaseMatDataAdapter,
    "BaseResprCsvDataset": BaseResprCsvDataset,
    "ResprDataLoaderComposer": ResprDataLoaderComposer,
    "ResprCsvDataLoaderComposer": ResprCsvDataLoaderComposer
}

# TODO: change name
DATA_ADAPTER_FACTORY = BaseFactory({"resource_map": COMPONENTS_MAP})