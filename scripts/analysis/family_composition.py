"""Plot the most common particle-species compositions in real families."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.particles import particle_label


def count_compositions(
    data_dir: str | Path,
    prefix: str = "bib_data",
    family_size: int | None = None,
    chunk_size: int = 5_000,
) -> Counter[tuple[int, ...]]:
    data_dir = Path(data_dir)
    daughters = np.load(data_dir / f"{prefix}_daughters.npy", mmap_mode="r")
    with (data_dir / f"{prefix}_scalers.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    pdg_categories = np.asarray(metadata["pdg_categories"])
    species_start = 1 + len(metadata["kinematic_features"])

    compositions: Counter[tuple[int, ...]] = Counter()
    for start in tqdm(range(0, len(daughters), chunk_size), desc="Counting compositions"):
        stop = min(start + chunk_size, len(daughters))
        chunk = np.asarray(daughters[start:stop])
        masks = chunk[:, :, 0] > 0.5
        counts = masks.sum(axis=1)
        species_indices = np.argmax(chunk[:, :, species_start:], axis=2)

        for row, mask, count in zip(species_indices, masks, counts, strict=True):
            if count == 0 or (family_size is not None and count != family_size):
                continue
            composition = tuple(sorted(int(code) for code in pdg_categories[row[mask]]))
            compositions[composition] += 1

    return compositions


def plot_compositions(
    compositions: Counter[tuple[int, ...]],
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    most_common = compositions.most_common(top_n)
    if not most_common:
        raise ValueError("No families satisfy the requested selection")

    labels = [
        " + ".join(particle_label(code) for code in composition)
        for composition, _ in most_common
    ]
    counts = [count for _, count in most_common]
    labels.reverse()
    counts.reverse()

    figure, axis = plt.subplots(figsize=(14, 10))
    axis.barh(labels, counts)
    axis.set_xlabel("Families")
    axis.set_ylabel("Species composition")
    axis.set_title(f"Most common BIB family compositions (top {len(labels)})")
    axis.grid(axis="x", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prefix", default="bib_data")
    parser.add_argument("--family-size", type=int)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default="family_compositions.png")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    composition_counts = count_compositions(
        arguments.data_dir,
        prefix=arguments.prefix,
        family_size=arguments.family_size,
    )
    plot_compositions(composition_counts, arguments.output, top_n=arguments.top_n)
