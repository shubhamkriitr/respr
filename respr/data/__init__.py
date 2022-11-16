from respr.util.common import BaseFactory
from .bidmc import BidmcDataAdapter
from .capnobase import CapnobaseDataAdapter, CapnobaseMatDataAdapter
from .loader import BaseResprCsvDataset, ResprDataLoaderComposer

COMPONENTS_MAP = {
    "BidmcDataAdapter": BidmcDataAdapter,
    "CapnobaseDataAdapter": CapnobaseDataAdapter,
    "CapnobaseMatDataAdapter": CapnobaseMatDataAdapter,
    "BaseResprCsvDataset": BaseResprCsvDataset,
    "ResprDataLoaderComposer": ResprDataLoaderComposer
}

# TODO: change name
DATA_ADAPTER_FACTORY = BaseFactory({"resource_map": COMPONENTS_MAP})