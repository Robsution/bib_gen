"""PyTorch dataset for processed beam-induced-background families."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BIBDataset(Dataset):
    """Load padded parent/daughter arrays produced by the preprocessing script.

    Each daughter row contains an activity mask in column zero, followed by
    continuous kinematics and a one-hot particle-species encoding.

    Args:
        data_dir: Directory containing ``<prefix>_parents.npy`` and
            ``<prefix>_daughters.npy``.
        prefix: Shared file prefix.
        min_daughters: Minimum accepted family size, inclusive.
        max_daughters: Maximum accepted family size, inclusive. ``None`` keeps
            all family sizes and normalizes counts by the stored padded length.
        filter_batch_size: Families inspected at once when filtering.
    """

    def __init__(
        self,
        data_dir: str | Path,
        prefix: str = "bib_data",
        min_daughters: int = 0,
        max_daughters: int | None = None,
        filter_batch_size: int = 20_000,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir)
        parents_path = data_dir / f"{prefix}_parents.npy"
        daughters_path = data_dir / f"{prefix}_daughters.npy"

        if not parents_path.exists() or not daughters_path.exists():
            raise FileNotFoundError(
                f"Expected processed arrays at {parents_path} and {daughters_path}"
            )
        if min_daughters < 0:
            raise ValueError("min_daughters must be non-negative")
        if max_daughters is not None and max_daughters < min_daughters:
            raise ValueError("max_daughters must be greater than or equal to min_daughters")

        parents = np.load(parents_path, mmap_mode="r")
        daughters = np.load(daughters_path, mmap_mode="r")
        self._validate_shapes(parents, daughters)

        self.padded_size = int(daughters.shape[1])
        self.count_scale = max_daughters if max_daughters is not None else self.padded_size

        needs_filter = min_daughters > 0 or (
            max_daughters is not None and max_daughters < self.padded_size
        )
        if needs_filter:
            parents, daughters = self._filter_families(
                parents,
                daughters,
                min_daughters=min_daughters,
                max_daughters=max_daughters,
                batch_size=filter_batch_size,
            )

        self.parents = parents
        self.daughters = daughters

    @staticmethod
    def _validate_shapes(parents: np.ndarray, daughters: np.ndarray) -> None:
        if parents.ndim != 2:
            raise ValueError(f"Parent array must be rank 2, got shape {parents.shape}")
        if daughters.ndim != 3:
            raise ValueError(f"Daughter array must be rank 3, got shape {daughters.shape}")
        if len(parents) != len(daughters):
            raise ValueError("Parent and daughter arrays contain different numbers of families")
        if daughters.shape[2] < 2:
            raise ValueError("Daughter rows must contain a mask and at least one feature")

    @staticmethod
    def _filter_families(
        parents: np.ndarray,
        daughters: np.ndarray,
        min_daughters: int,
        max_daughters: int | None,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        parent_chunks: list[np.ndarray] = []
        daughter_chunks: list[np.ndarray] = []

        for start in range(0, len(parents), batch_size):
            stop = min(start + batch_size, len(parents))
            daughter_chunk = np.asarray(daughters[start:stop])
            counts = (daughter_chunk[:, :, 0] > 0.5).sum(axis=1)
            keep = counts >= min_daughters
            if max_daughters is not None:
                keep &= counts <= max_daughters
            if np.any(keep):
                parent_chunks.append(np.asarray(parents[start:stop])[keep])
                daughter_chunks.append(daughter_chunk[keep])

        if not parent_chunks:
            raise ValueError(
                "No families satisfy the requested multiplicity interval "
                f"[{min_daughters}, {max_daughters}]"
            )

        return np.concatenate(parent_chunks), np.concatenate(daughter_chunks)

    def __len__(self) -> int:
        return len(self.parents)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent = torch.as_tensor(np.array(self.parents[index]), dtype=torch.float32)
        daughter_rows = np.array(self.daughters[index])
        mask = torch.as_tensor(daughter_rows[:, :1], dtype=torch.float32)
        features = torch.as_tensor(daughter_rows[:, 1:], dtype=torch.float32)
        normalized_n = mask.sum() / self.count_scale
        return parent, features, mask, normalized_n
