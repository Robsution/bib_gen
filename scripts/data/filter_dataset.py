"""Create a naturally multiplicity-limited dataset without clipping large families."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def filter_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    max_daughters: int,
    min_daughters: int = 0,
    prefix: str = "bib_data",
    chunk_size: int = 5_000,
    overwrite: bool = False,
) -> int:
    """Keep complete families in ``[min_daughters, max_daughters]``.

    Families above the maximum are excluded rather than clipped. Output arrays
    retain only ``max_daughters`` padded slots and use the canonical filenames
    expected by :class:`src.dataset.BIBDataset`.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different")
    if min_daughters < 0 or max_daughters < min_daughters:
        raise ValueError("Expected 0 <= min_daughters <= max_daughters")

    parents_path = input_dir / f"{prefix}_parents.npy"
    daughters_path = input_dir / f"{prefix}_daughters.npy"
    parents = np.load(parents_path, mmap_mode="r")
    daughters = np.load(daughters_path, mmap_mode="r")
    if len(parents) != len(daughters):
        raise ValueError("Parent and daughter arrays have different family counts")
    if daughters.shape[1] < max_daughters:
        raise ValueError(
            f"Input has only {daughters.shape[1]} padded slots; cannot create max={max_daughters}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_parents_path = output_dir / f"{prefix}_parents.npy"
    output_daughters_path = output_dir / f"{prefix}_daughters.npy"
    existing = [path for path in (output_parents_path, output_daughters_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Pass --overwrite to replace them."
        )

    kept_families = 0
    for start in tqdm(range(0, len(daughters), chunk_size), desc="Counting families"):
        stop = min(start + chunk_size, len(daughters))
        counts = (np.asarray(daughters[start:stop, :, 0]) > 0.5).sum(axis=1)
        kept_families += int(((counts >= min_daughters) & (counts <= max_daughters)).sum())

    if kept_families == 0:
        raise ValueError("No families satisfy the requested multiplicity interval")

    output_parents = np.lib.format.open_memmap(
        output_parents_path,
        mode="w+",
        dtype=parents.dtype,
        shape=(kept_families, parents.shape[1]),
    )
    output_daughters = np.lib.format.open_memmap(
        output_daughters_path,
        mode="w+",
        dtype=daughters.dtype,
        shape=(kept_families, max_daughters, daughters.shape[2]),
    )

    output_index = 0
    for start in tqdm(range(0, len(daughters), chunk_size), desc="Writing families"):
        stop = min(start + chunk_size, len(daughters))
        daughter_chunk = np.asarray(daughters[start:stop])
        counts = (daughter_chunk[:, :, 0] > 0.5).sum(axis=1)
        keep = (counts >= min_daughters) & (counts <= max_daughters)
        chunk_count = int(keep.sum())
        if chunk_count == 0:
            continue

        next_index = output_index + chunk_count
        output_parents[output_index:next_index] = np.asarray(parents[start:stop])[keep]
        output_daughters[output_index:next_index] = daughter_chunk[keep, :max_daughters]
        output_index = next_index

    output_parents.flush()
    output_daughters.flush()

    for suffix in ("scalers.json", "parent_scalers.json"):
        source = input_dir / f"{prefix}_{suffix}"
        if source.exists():
            shutil.copy2(source, output_dir / source.name)

    print(f"Wrote {kept_families} complete families to {output_dir}")
    return kept_families


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-daughters", required=True, type=int)
    parser.add_argument("--min-daughters", default=0, type=int)
    parser.add_argument("--prefix", default="bib_data")
    parser.add_argument("--chunk-size", default=5_000, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    filter_dataset(
        arguments.input_dir,
        arguments.output_dir,
        arguments.max_daughters,
        min_daughters=arguments.min_daughters,
        prefix=arguments.prefix,
        chunk_size=arguments.chunk_size,
        overwrite=arguments.overwrite,
    )
