"""Validate structural invariants of a processed BIB dataset."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class ValidationReport:
    families: int
    parent_nan_families: int
    daughter_nan_families: int
    empty_families: int
    active_zero_feature_particles: int
    nonbinary_mask_particles: int

    @property
    def is_valid(self) -> bool:
        return all(
            value == 0
            for value in (
                self.parent_nan_families,
                self.daughter_nan_families,
                self.active_zero_feature_particles,
                self.nonbinary_mask_particles,
            )
        )


def validate_dataset(
    data_dir: str | Path,
    prefix: str = "bib_data",
    chunk_size: int = 10_000,
) -> ValidationReport:
    data_dir = Path(data_dir)
    parents = np.load(data_dir / f"{prefix}_parents.npy", mmap_mode="r")
    daughters = np.load(data_dir / f"{prefix}_daughters.npy", mmap_mode="r")

    if parents.ndim != 2 or daughters.ndim != 3:
        raise ValueError(f"Unexpected shapes: parents={parents.shape}, daughters={daughters.shape}")
    if len(parents) != len(daughters):
        raise ValueError("Parent and daughter arrays have different family counts")

    parent_nan_families = 0
    daughter_nan_families = 0
    empty_families = 0
    active_zero_feature_particles = 0
    nonbinary_mask_particles = 0

    for start in tqdm(range(0, len(parents), chunk_size), desc="Validating"):
        stop = min(start + chunk_size, len(parents))
        parent_chunk = np.asarray(parents[start:stop])
        daughter_chunk = np.asarray(daughters[start:stop])

        parent_nan_families += int(np.isnan(parent_chunk).any(axis=1).sum())
        daughter_nan_families += int(np.isnan(daughter_chunk).any(axis=(1, 2)).sum())

        raw_masks = daughter_chunk[:, :, 0]
        masks = raw_masks > 0.5
        empty_families += int((masks.sum(axis=1) == 0).sum())
        nonbinary_masks = ~np.isclose(raw_masks, 0.0) & ~np.isclose(raw_masks, 1.0)
        nonbinary_mask_particles += int(nonbinary_masks.sum())

        feature_magnitude = np.abs(daughter_chunk[:, :, 1:]).sum(axis=2)
        active_zero_feature_particles += int((masks & (feature_magnitude == 0)).sum())

    return ValidationReport(
        families=len(parents),
        parent_nan_families=parent_nan_families,
        daughter_nan_families=daughter_nan_families,
        empty_families=empty_families,
        active_zero_feature_particles=active_zero_feature_particles,
        nonbinary_mask_particles=nonbinary_mask_particles,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prefix", default="bib_data")
    parser.add_argument("--chunk-size", default=10_000, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = validate_dataset(arguments.data_dir, arguments.prefix, arguments.chunk_size)
    for field, value in report.__dict__.items():
        print(f"{field}: {value}")
    raise SystemExit(0 if report.is_valid else 1)
