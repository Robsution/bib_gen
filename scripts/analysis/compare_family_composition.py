"""Compare real and generated compositions at fixed real family multiplicities."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BIBDataset
from src.models import BIBGenerator
from src.particles import particle_label
from src.runtime import load_yaml, resolve_device, seed_everything


def compare_compositions(
    config: dict,
    checkpoint_path: str | Path,
    target_families: int = 50_000,
    top_n: int = 15,
) -> Path:
    """Evaluate the kinematic generator using each real family's observed ``N``.

    This isolates species and family-generation behavior. It does not evaluate
    the separately trained multiplicity model.
    """

    seed_everything(int(config.get("seed", 12345)))
    device = resolve_device(config)
    max_daughters = int(config["max_daughters"])
    kinematic_dim = int(config.get("kinematic_dim", 8))
    dataset = BIBDataset(
        config["data_dir"],
        min_daughters=int(config.get("min_daughters", 0)),
        max_daughters=max_daughters,
    )
    if dataset.padded_size != max_daughters:
        raise ValueError(
            f"Dataset is padded to {dataset.padded_size}, but max_daughters={max_daughters}"
        )
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)

    parent, features, _, _ = dataset[0]
    generator = BIBGenerator(
        noise_dim=int(config["z_dim"]),
        parent_dim=parent.shape[0],
        max_daughters=max_daughters,
        kinematic_dim=kinematic_dim,
        species_dim=features.shape[1] - kinematic_dim,
        hidden_dim=int(config.get("generator_hidden_dim", 256)),
    ).to(device)
    generator.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    generator.eval()

    with (Path(config["data_dir"]) / "bib_data_scalers.json").open(encoding="utf-8") as file:
        pdg_categories = np.asarray(json.load(file)["pdg_categories"])

    real_counts: Counter[tuple[int, ...]] = Counter()
    generated_counts: Counter[tuple[int, ...]] = Counter()
    processed_families = 0

    with torch.no_grad():
        for real_parent, real_features, real_mask, normalized_n in dataloader:
            if processed_families >= target_families:
                break

            batch_size = real_parent.shape[0]
            real_parent = real_parent.to(device)
            normalized_n = normalized_n.to(device).float().unsqueeze(1)
            noise = torch.randn(batch_size, int(config["z_dim"]), device=device)
            generated_features, generated_mask = generator(noise, real_parent, normalized_n)

            _update_counts(
                real_counts,
                real_features.numpy(),
                real_mask.numpy(),
                pdg_categories,
                kinematic_dim,
            )
            _update_counts(
                generated_counts,
                generated_features.cpu().numpy(),
                generated_mask.cpu().numpy(),
                pdg_categories,
                kinematic_dim,
            )
            processed_families += batch_size

    output_dir = Path(config["plot_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "conditional_family_compositions.png"
    _plot_counts(real_counts, generated_counts, output_path, top_n)
    print(f"Saved {output_path}")
    return output_path


def _update_counts(
    output: Counter[tuple[int, ...]],
    features: np.ndarray,
    masks: np.ndarray,
    pdg_categories: np.ndarray,
    kinematic_dim: int,
) -> None:
    active = masks[:, :, 0] > 0.5
    indices = np.argmax(features[:, :, kinematic_dim:], axis=2)
    for row, mask in zip(indices, active, strict=True):
        if np.any(mask):
            output[tuple(sorted(int(code) for code in pdg_categories[row[mask]]))] += 1


def _plot_counts(
    real_counts: Counter[tuple[int, ...]],
    generated_counts: Counter[tuple[int, ...]],
    output_path: Path,
    top_n: int,
) -> None:
    topologies = [topology for topology, _ in real_counts.most_common(top_n)]
    labels = [" + ".join(particle_label(code) for code in topology) for topology in topologies]
    real_total = sum(real_counts.values())
    generated_total = sum(generated_counts.values())
    if real_total == 0 or generated_total == 0:
        raise ValueError("Cannot compare empty real or generated family samples")
    real_frequency = [real_counts[topology] / real_total for topology in topologies]
    generated_frequency = [
        generated_counts[topology] / generated_total for topology in topologies
    ]

    positions = np.arange(len(labels))
    width = 0.4
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.bar(positions - width / 2, real_frequency, width, label="FLUKA")
    axis.bar(positions + width / 2, generated_frequency, width, label="Generated")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Family-weighted frequency")
    axis.set_title("Family composition conditioned on observed multiplicity")
    axis.grid(axis="y", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_50.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-families", type=int, default=50_000)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    compare_compositions(
        load_yaml(arguments.config),
        arguments.checkpoint,
        target_families=arguments.target_families,
        top_n=arguments.top_n,
    )
