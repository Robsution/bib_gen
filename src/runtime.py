"""Shared configuration, device, and reproducibility helpers."""

import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("device")
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; using CPU instead", stacklevel=2)
        requested = "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, and a DataLoader sampler generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
