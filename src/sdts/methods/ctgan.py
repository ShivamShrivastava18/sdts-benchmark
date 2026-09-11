"""SDV CTGANSynthesizer. Defaults are the SDV/CTGAN paper defaults
(300 epochs, batch 500, generator and discriminator 256x256). Runs on CUDA
when available, otherwise CPU; MPS is not used by ctgan."""

from __future__ import annotations

from typing import Any

import torch
from sdv.single_table import CTGANSynthesizer

from sdts.methods._sdv import SDVMethod
from sdts.methods.base import register


@register("ctgan")
class CTGAN(SDVMethod):
    synthesizer_cls = CTGANSynthesizer

    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {
            "epochs": 300, "batch_size": 500,
            "generator_dim": (256, 256), "discriminator_dim": (256, 256),
            "generator_lr": 2e-4, "discriminator_lr": 2e-4,
            "embedding_dim": 128, "pac": 10,
            "enforce_min_max_values": True, "cuda": torch.cuda.is_available(),
            "verbose": False,
        }
