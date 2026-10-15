from .lstm_ae import LSTMAEDetector      # torch-based models first (see src/__init__.py)
from .tranad import TranADDetector
from .iforest import IForestDetector

DETECTORS = {"lstm_ae": LSTMAEDetector, "tranad": TranADDetector, "iforest": IForestDetector}


def build(name: str, cfg: dict, seed: int, device: str):
    return DETECTORS[name](cfg[name], seed, device)
