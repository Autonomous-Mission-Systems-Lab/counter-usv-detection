"""Dataset curation, harmonization, and loaders."""

from counterusv.data.eo_dataset import EODatasetConfig, EODetectionDataset, default_config
from counterusv.data.letterbox import LetterboxMeta, letterbox_image, letterbox_params, remap_boxes
from counterusv.data.overlay import (
    SEASHIPS_BANDS,
    SEASHIPS_BOTTOM_BAND,
    SEASHIPS_TOP_BAND,
    mask_seaships_overlay,
)

__all__ = [
    "EODatasetConfig",
    "EODetectionDataset",
    "default_config",
    "LetterboxMeta",
    "letterbox_image",
    "letterbox_params",
    "remap_boxes",
    "SEASHIPS_BANDS",
    "SEASHIPS_TOP_BAND",
    "SEASHIPS_BOTTOM_BAND",
    "mask_seaships_overlay",
]
