# Drift Detection v3.0 包初始化
from .drift_detector_v3 import (
    ThresholdAwareDriftDetectorV3,
    DriftLevel,
    CorrectionAction,
    ThresholdMode,
    DriftResult,
    create_detector,
    quick_detect
)

__version__ = "3.0.0"
__all__ = [
    "ThresholdAwareDriftDetectorV3",
    "DriftLevel",
    "CorrectionAction",
    "ThresholdMode",
    "DriftResult",
    "create_detector",
    "quick_detect"
]
