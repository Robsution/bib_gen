"""Convert variable-width FLUKA CSV rows into padded NumPy arrays."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PARENT_FEATURES = ("x", "y", "z")
KINEMATIC_FEATURES = ("energy", "x", "y", "z", "px", "py", "pz", "t")
PARENT_WIDTH = len(PARENT_FEATURES)
RAW_DAUGHTER_WIDTH = 11
PDG_OFFSET = 0
KINEMATIC_SLICE = slice(1, 9)


def line_chunks(path: str | Path, chunk_size: int) -> Iterator[list[str]]:
    """Yield nonempty lines without passing through the CSV field-size limit."""

    with Path(path).open(encoding="utf-8") as input_file:
        batch: list[str] = []
        for line in input_file:
            stripped = line.strip()
            if stripped:
                batch.append(stripped)
            if len(batch) == chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch


def parse_lines(lines: list[str], target_columns: int | None = None) -> np.ndarray:
    columns = pd.Series(lines).str.split(",", expand=True)
    if target_columns is not None and columns.shape[1] < target_columns:
        columns = columns.reindex(columns=range(target_columns))
    values = columns.apply(pd.to_numeric, errors="coerce").to_numpy()

    populated_columns = (~np.isnan(values)).sum(axis=1)
    malformed = (populated_columns < PARENT_WIDTH) | (
        (populated_columns - PARENT_WIDTH) % RAW_DAUGHTER_WIDTH != 0
    )
    if np.any(malformed):
        raise ValueError(
            "Each row must contain three parent values followed by complete "
            f"{RAW_DAUGHTER_WIDTH}-value daughter records"
        )
    if np.isnan(values[:, :PARENT_WIDTH]).any():
        raise ValueError("Parent coordinates must be numeric; the input must not contain a header")
    return values


def preprocess(
    csv_path: str | Path,
    output_dir: str | Path,
    prefix: str = "bib_data",
    chunk_size: int = 5_000,
    overwrite: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Scan, scale, one-hot encode, and pad the raw family records.

    The raw daughter schema has 11 values. This baseline retains the PDG code
    and the next eight values ``(energy, x, y, z, px, py, pz, t)``. The final
    two raw values are recorded as ignored offsets in the metadata.
    """

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parents_path = output_dir / f"{prefix}_parents.npy"
    daughters_path = output_dir / f"{prefix}_daughters.npy"
    if not overwrite and (parents_path.exists() or daughters_path.exists()):
        raise FileExistsError("Processed arrays already exist; pass --overwrite to replace them")

    total_families = 0
    maximum_columns = 0
    pdg_codes: set[int] = set()
    parent_min = np.full(PARENT_WIDTH, np.inf)
    parent_max = np.full(PARENT_WIDTH, -np.inf)
    kinematic_min = np.full(len(KINEMATIC_FEATURES), np.inf)
    kinematic_max = np.full(len(KINEMATIC_FEATURES), -np.inf)

    print("Scanning raw data")
    for lines in line_chunks(csv_path, chunk_size):
        values = parse_lines(lines)
        total_families += len(values)
        maximum_columns = max(maximum_columns, values.shape[1])

        parents = values[:, :PARENT_WIDTH]
        parent_min = np.minimum(parent_min, parents.min(axis=0))
        parent_max = np.maximum(parent_max, parents.max(axis=0))

        daughter_values = values[:, PARENT_WIDTH:]
        daughter_slots = daughter_values.shape[1] // RAW_DAUGHTER_WIDTH
        for slot in range(daughter_slots):
            start = slot * RAW_DAUGHTER_WIDTH
            particle = daughter_values[:, start : start + RAW_DAUGHTER_WIDTH]
            active = ~np.isnan(particle[:, PDG_OFFSET])
            if not np.any(active):
                continue

            active_particles = particle[active]
            kinematics = active_particles[:, KINEMATIC_SLICE]
            if np.isnan(kinematics).any():
                raise ValueError("An active daughter has missing kinematic values")
            pdg_codes.update(active_particles[:, PDG_OFFSET].astype(int))
            kinematic_min = np.minimum(kinematic_min, kinematics.min(axis=0))
            kinematic_max = np.maximum(kinematic_max, kinematics.max(axis=0))

    if total_families == 0 or not pdg_codes:
        raise ValueError("No families with active daughters were found")

    max_daughters = (maximum_columns - PARENT_WIDTH) // RAW_DAUGHTER_WIDTH
    sorted_pdgs = [int(code) for code in sorted(pdg_codes)]
    pdg_encoder = OneHotEncoder(
        categories=[sorted_pdgs],
        sparse_output=False,
        handle_unknown="error",
    ).fit(np.asarray(sorted_pdgs).reshape(-1, 1))
    kinematic_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
        np.vstack((kinematic_min, kinematic_max))
    )
    parent_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
        np.vstack((parent_min, parent_max))
    )

    parent_shape = (total_families, PARENT_WIDTH)
    daughter_feature_dim = 1 + len(KINEMATIC_FEATURES) + len(sorted_pdgs)
    daughter_shape = (total_families, max_daughters, daughter_feature_dim)
    parent_output = np.lib.format.open_memmap(
        parents_path, mode="w+", dtype="float32", shape=parent_shape
    )
    daughter_output = np.lib.format.open_memmap(
        daughters_path, mode="w+", dtype="float32", shape=daughter_shape
    )

    print("Writing processed arrays")
    output_index = 0
    for lines in line_chunks(csv_path, chunk_size):
        values = parse_lines(lines, target_columns=maximum_columns)
        batch_size = len(values)
        next_index = output_index + batch_size
        parent_output[output_index:next_index] = parent_scaler.transform(
            values[:, :PARENT_WIDTH]
        )

        batch = np.zeros((batch_size, max_daughters, daughter_feature_dim), dtype="float32")
        daughter_values = values[:, PARENT_WIDTH:]
        for slot in range(max_daughters):
            start = slot * RAW_DAUGHTER_WIDTH
            particle = daughter_values[:, start : start + RAW_DAUGHTER_WIDTH]
            active_indices = np.flatnonzero(~np.isnan(particle[:, PDG_OFFSET]))
            if len(active_indices) == 0:
                continue

            active_particles = particle[active_indices]
            one_hot_pdg = pdg_encoder.transform(
                active_particles[:, PDG_OFFSET].reshape(-1, 1)
            )
            scaled_kinematics = kinematic_scaler.transform(
                active_particles[:, KINEMATIC_SLICE]
            )
            batch[active_indices, slot] = np.column_stack(
                (np.ones(len(active_indices)), scaled_kinematics, one_hot_pdg)
            )

        daughter_output[output_index:next_index] = batch
        output_index = next_index

    parent_output.flush()
    daughter_output.flush()
    _write_metadata(
        output_dir,
        prefix,
        sorted_pdgs,
        kinematic_min,
        kinematic_max,
        parent_min,
        parent_max,
    )
    print(f"Wrote parents {parent_shape} and daughters {daughter_shape} to {output_dir}")
    return parent_shape, daughter_shape


def _write_metadata(
    output_dir: Path,
    prefix: str,
    pdg_codes: list[int],
    kinematic_min: np.ndarray,
    kinematic_max: np.ndarray,
    parent_min: np.ndarray,
    parent_max: np.ndarray,
) -> None:
    daughter_metadata = {
        "kinematic_features": list(KINEMATIC_FEATURES),
        "data_min": kinematic_min.tolist(),
        "data_max": kinematic_max.tolist(),
        "pdg_categories": pdg_codes,
        "raw_daughter_width": RAW_DAUGHTER_WIDTH,
        "ignored_raw_offsets": [9, 10],
    }
    parent_metadata = {
        "kinematic_features": list(PARENT_FEATURES),
        "data_min": parent_min.tolist(),
        "data_max": parent_max.tolist(),
    }
    with (output_dir / f"{prefix}_scalers.json").open("w", encoding="utf-8") as file:
        json.dump(daughter_metadata, file, indent=2)
    with (output_dir / f"{prefix}_parent_scalers.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(parent_metadata, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="bib_data")
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    preprocess(
        arguments.csv_path,
        arguments.output_dir,
        prefix=arguments.prefix,
        chunk_size=arguments.chunk_size,
        overwrite=arguments.overwrite,
    )
