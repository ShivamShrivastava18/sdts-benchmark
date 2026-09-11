"""SDV TVAESynthesizer. Defaults are the SDV/TVAE paper defaults
(300 epochs, batch 500, 128x128 encoder and decoder)."""

from __future__ import annotations

from typing import Any

import torch
from sdv.single_table import TVAESynthesizer

from sdts.methods._sdv import SDVMethod
from sdts.methods.base import register


@register("tvae")
class TVAE(SDVMethod):
    synthesizer_cls = TVAESynthesizer

    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {
            "epochs": 300, "batch_size": 500,
            "compress_dims": (128, 128), "decompress_dims": (128, 128),
            "embedding_dim": 128, "l2scale": 1e-5, "loss_factor": 2,
            "enforce_min_max_values": True, "cuda": torch.cuda.is_available(),
            "verbose": False,
        }
