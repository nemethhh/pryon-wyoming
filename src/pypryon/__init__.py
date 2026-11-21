"""pypryon: run Amazon Pryon wake-word models on stock onnxruntime.

``Spotter`` is the entry point (int16 16 kHz PCM in, ``PryonDetection`` out);
the lower layers are importable for advanced use.
"""
from .decoder import Detection, Orchestra
from .fe import lfbe_frames
from .modelset import ModelError, ModelSet, load_model_set
from .runner import GraphRunner, Posteriors
from .spotter import ACCEPT, ACCEPT_AUXILIARY, NEAR_MISS, PryonDetection, Spotter

__version__ = "0.2.0"

__all__ = [
    "ACCEPT",
    "ACCEPT_AUXILIARY",
    "NEAR_MISS",
    "Detection",
    "GraphRunner",
    "ModelError",
    "ModelSet",
    "Orchestra",
    "Posteriors",
    "PryonDetection",
    "Spotter",
    "__version__",
    "lfbe_frames",
    "load_model_set",
]
