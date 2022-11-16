from respr.util.common import BaseFactory
from .bidmc import BidmcDataAdapter
from .capnobase import CapnobaseDataAdapter, CapnobaseMatDataAdapter
from .loader import BaseResprCsvDataset

COMPONENTS_MAP = {
    "BidmcDataAdapter": BidmcDataAdapter,
    "CapnobaseDataAdapter": CapnobaseDataAdapter,
    "CapnobaseMatDataAdapter": CapnobaseMatDataAdapter,
    "BaseResprCsvDataset": BaseResprCsvDataset
}
DATA_ADAPTER_FACTORY = BaseFactory({"resource_map": COMPONENTS_MAP})