import json

import numpy as np

from scripts.data.filter_dataset import filter_dataset
from src.dataset import BIBDataset


def write_dataset(directory, counts, padded_size=4, feature_dim=11):
    directory.mkdir()
    parents = np.arange(len(counts) * 3, dtype=np.float32).reshape(len(counts), 3)
    daughters = np.zeros((len(counts), padded_size, feature_dim), dtype=np.float32)
    for family_index, count in enumerate(counts):
        daughters[family_index, :count, 0] = 1.0
        daughters[family_index, :count, 1] = family_index + 1.0
        daughters[family_index, :count, 9] = 1.0

    np.save(directory / "bib_data_parents.npy", parents)
    np.save(directory / "bib_data_daughters.npy", daughters)
    (directory / "bib_data_scalers.json").write_text(
        json.dumps({"pdg_categories": [22, 2112]}), encoding="utf-8"
    )
    (directory / "bib_data_parent_scalers.json").write_text("{}", encoding="utf-8")


def test_dataset_filters_families_and_normalizes_count(tmp_path):
    data_dir = tmp_path / "input"
    write_dataset(data_dir, counts=[1, 3, 4])

    dataset = BIBDataset(data_dir, min_daughters=1, max_daughters=3)

    assert len(dataset) == 2
    assert dataset.padded_size == 4
    assert dataset[0][3].item() == 1 / 3
    assert dataset[1][3].item() == 1.0


def test_filter_dataset_excludes_instead_of_clipping(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_dataset(input_dir, counts=[1, 3, 4])

    kept = filter_dataset(input_dir, output_dir, max_daughters=3)
    output_daughters = np.load(output_dir / "bib_data_daughters.npy")

    assert kept == 2
    assert output_daughters.shape == (2, 3, 11)
    assert (output_daughters[:, :, 0].sum(axis=1) == [1, 3]).all()
    assert (output_dir / "bib_data_scalers.json").exists()
