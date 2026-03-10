"""Trust Calibration in AI Systems - Core Package."""

__version__ = "0.1.0"
__author__ = "XAI Research Team"
__email__ = "research@example.com"

from .models.calibration import CalibratedModel, PlattScaling, IsotonicCalibration
from .eval.metrics import CalibrationEvaluator, UncertaintyEvaluator, ModelComparison
from .data.datasets import (
    load_iris_dataset,
    load_wine_dataset, 
    load_breast_cancer_dataset,
    create_synthetic_dataset,
    DatasetMetadata
)
from .viz.plots import CalibrationVisualizer
from .utils.device import get_device, set_seed, get_device_info

__all__ = [
    "CalibratedModel",
    "PlattScaling", 
    "IsotonicCalibration",
    "CalibrationEvaluator",
    "UncertaintyEvaluator",
    "ModelComparison",
    "load_iris_dataset",
    "load_wine_dataset",
    "load_breast_cancer_dataset", 
    "create_synthetic_dataset",
    "DatasetMetadata",
    "CalibrationVisualizer",
    "get_device",
    "set_seed",
    "get_device_info",
]
