"""Video saliency prediction (attention maps) in PyTorch.

Public API
----------
- :class:`~saliency.models.baseline.SaliencyModel` (DeepLabV3 baseline)
- :class:`~saliency.models.improved.ImprovedSaliencyModel` (EfficientNet-B4)
- :class:`~saliency.data.dataset.SaliencyFrameDataset`
- losses in :mod:`saliency.losses`, metrics in :mod:`saliency.metrics`
- training in :mod:`saliency.engine`, inference in :mod:`saliency.evaluate`
"""

from .evaluate import SaliencyEvaluator
from .losses import CombinedSaliencyLoss, kld_loss
from .metrics import calculate_metrics, cc, nss, similarity
from .models import ImprovedSaliencyModel, SaliencyModel

__version__ = "0.1.0"

__all__ = [
    "SaliencyModel",
    "ImprovedSaliencyModel",
    "kld_loss",
    "CombinedSaliencyLoss",
    "similarity",
    "cc",
    "nss",
    "calculate_metrics",
    "SaliencyEvaluator",
]
