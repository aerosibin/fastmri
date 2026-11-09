"""
src/utils.py
General utility functions for seed setting, device configuration, and logging.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested_device: str = "auto") -> torch.device:
    """Selects best available device (CUDA GPU or CPU)."""
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return device


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = "checkpoint_latest.pt"
) -> Path:
    """Saves model checkpoint and updates best model."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        return best_path
    return filepath

