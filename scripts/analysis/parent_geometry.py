"""Plot mean family size over the parent-position geometry."""

import argparse
import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_parent_geometry(
    data_dir: str | Path,
    output_path: str | Path,
    prefix: str = "bib_data",
    gridsize: int = 100,
) -> None:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parents = np.load(data_dir / f"{prefix}_parents.npy", mmap_mode="r")
    daughters = np.load(data_dir / f"{prefix}_daughters.npy", mmap_mode="r")
    family_sizes = (daughters[:, :, 0] > 0.5).sum(axis=1)

    with (data_dir / f"{prefix}_parent_scalers.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    data_min = np.asarray(metadata["data_min"])
    data_max = np.asarray(metadata["data_max"])
    physical_parents = ((parents + 1.0) / 2.0) * (data_max - data_min) + data_min
    x, y, z = physical_parents.T
    radius = np.hypot(x, y)

    figure, (transverse, longitudinal) = plt.subplots(1, 2, figsize=(18, 7))
    transverse_bins = transverse.hexbin(
        x,
        y,
        C=family_sizes,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        cmap="plasma",
        mincnt=1,
        norm=colors.LogNorm(),
    )
    transverse.set(xlabel="x [mm]", ylabel="y [mm]", title="Mean family size in (x, y)")
    transverse.set_aspect("equal")
    figure.colorbar(transverse_bins, ax=transverse, label="Mean daughter count")

    longitudinal_bins = longitudinal.hexbin(
        z,
        radius,
        C=family_sizes,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        cmap="plasma",
        mincnt=1,
        norm=colors.LogNorm(),
    )
    longitudinal.set(
        xlabel="z [mm]",
        ylabel="r [mm]",
        title="Mean family size in (z, r)",
    )
    figure.colorbar(longitudinal_bins, ax=longitudinal, label="Mean daughter count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prefix", default="bib_data")
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--output", default="parent_multiplicity_geometry.png")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot_parent_geometry(
        arguments.data_dir,
        arguments.output,
        prefix=arguments.prefix,
        gridsize=arguments.grid_size,
    )
