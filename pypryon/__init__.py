"""pypryon: run Amazon Pryon wake-word models on stock onnxruntime.

Public API: `Spotter` (PCM in, detections out) and `PryonDetection`;
lower layers (`load_model_set`, `lfbe_frames`, `GraphRunner`, `Orchestra`)
are importable for advanced use.
"""
from .decoder import Detection, Orchestra
from .fe import lfbe_frames
from .modelset import ModelSet, load_model_set
from .runner import GraphRunner, Posteriors
from .spotter import PryonDetection, Spotter

__all__ = [
    "Detection",
    "GraphRunner",
    "ModelSet",
    "Orchestra",
    "Posteriors",
    "PryonDetection",
    "Spotter",
    "lfbe_frames",
    "load_model_set",
]
