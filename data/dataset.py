"""Compatibility import for the new training datasets.

The former generic refinement dataset did not supply BR outputs and is retired.
Use TrainingDataset with an explicit model config and verified cache directories.
"""
from .training_dataset import TrainingDataset, CaseOrderSampler
__all__ = ["TrainingDataset", "CaseOrderSampler"]
