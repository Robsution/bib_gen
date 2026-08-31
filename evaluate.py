"""Plot particle-level distributions for the trained two-stage baseline."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BIBDataset
from src.models import BIBGenerator, CategoricalMultiplicity
from src.runtime import load_yaml, resolve_device, seed_everything


def latest_generator_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.glob("generator_epoch_*.pt"))
    candidates.extend(checkpoint_dir.glob("generator_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No generator checkpoints found in {checkpoint_dir}")

    def epoch(path: Path) -> int:
        match = re.search(r"generator_epoch_(\d+)\.(?:pt|pth)$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch)


def evaluate(
    config: dict,
    generator_checkpoint: str | Path | None,
    multiplicity_checkpoint: str | Path | None,
    target_particles: int,
) -> Path:
    if target_particles <= 0:
        raise ValueError("target_particles must be positive")
    seed_everything(int(config.get("seed", 12345)))
    device = resolve_device(config)
    max_daughters = int(config["max_daughters"])
    kinematic_dim = int(config.get("kinematic_dim", 8))
    checkpoint_dir = Path(config["save_dir"])

    generator_path = (
        Path(generator_checkpoint)
        if generator_checkpoint
        else latest_generator_checkpoint(checkpoint_dir)
    )
    if multiplicity_checkpoint:
        multiplicity_path = Path(multiplicity_checkpoint)
    elif (checkpoint_dir / "multiplicity_model.pt").exists():
        multiplicity_path = checkpoint_dir / "multiplicity_model.pt"
    else:
        multiplicity_path = checkpoint_dir / "oracle_weights.pth"
    if not generator_path.exists():
        raise FileNotFoundError(generator_path)
    if not multiplicity_path.exists():
        raise FileNotFoundError(multiplicity_path)

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

    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    species_dim = feature_dim - kinematic_dim

    multiplicity_model = CategoricalMultiplicity(
        parent_dim=parent_dim,
        max_n=max_daughters,
        hidden_dim=int(config.get("multiplicity_hidden_dim", 128)),
    ).to(device)
    multiplicity_model.load_state_dict(
        torch.load(multiplicity_path, map_location=device, weights_only=True)
    )
    multiplicity_model.eval()

    generator = BIBGenerator(
        noise_dim=int(config["z_dim"]),
        parent_dim=parent_dim,
        max_daughters=max_daughters,
        kinematic_dim=kinematic_dim,
        species_dim=species_dim,
        hidden_dim=int(config.get("generator_hidden_dim", 256)),
    ).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    generator.eval()

    metadata_path = Path(config["data_dir"]) / "bib_data_scalers.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    data_min = np.asarray(metadata["data_min"])
    data_max = np.asarray(metadata["data_max"])
    feature_names = metadata["kinematic_features"]
    pdg_categories = np.asarray(metadata["pdg_categories"])
    if len(feature_names) != kinematic_dim:
        raise ValueError("Scaler metadata does not match the configured kinematic dimension")
    if len(pdg_categories) != species_dim:
        raise ValueError("PDG metadata does not match the processed feature dimension")

    real_batches: list[np.ndarray] = []
    generated_batches: list[np.ndarray] = []
    real_count = 0
    generated_count = 0

    with torch.no_grad():
        for parent, real_features, real_mask, _ in dataloader:
            if real_count >= target_particles and generated_count >= target_particles:
                break

            parent = parent.to(device)
            batch_size = parent.shape[0]
            count_probabilities = torch.softmax(multiplicity_model(parent), dim=-1)
            target_n = torch.multinomial(count_probabilities, num_samples=1).float()
            normalized_n = target_n / max_daughters
            noise = torch.randn(batch_size, int(config["z_dim"]), device=device)
            generated_features, generated_mask = generator(noise, parent, normalized_n)

            real_valid = real_mask.numpy().squeeze(-1) > 0.5
            generated_valid = generated_mask.cpu().numpy().squeeze(-1) > 0.5
            real_particles = real_features.numpy()[real_valid]
            generated_particles = generated_features.cpu().numpy()[generated_valid]

            if real_count < target_particles and len(real_particles):
                real_batches.append(real_particles)
                real_count += len(real_particles)
            if generated_count < target_particles and len(generated_particles):
                generated_batches.append(generated_particles)
                generated_count += len(generated_particles)

    if not real_batches or not generated_batches:
        raise RuntimeError("Evaluation produced an empty real or generated particle sample")

    real_scaled = np.vstack(real_batches)[:target_particles]
    generated_scaled = np.vstack(generated_batches)[:target_particles]
    real_kinematics = inverse_minmax(real_scaled[:, :kinematic_dim], data_min, data_max)
    generated_kinematics = inverse_minmax(
        generated_scaled[:, :kinematic_dim], data_min, data_max
    )
    real_pdg = pdg_categories[np.argmax(real_scaled[:, kinematic_dim:], axis=1)]
    generated_pdg = pdg_categories[
        np.argmax(generated_scaled[:, kinematic_dim:], axis=1)
    ]

    output_dir = Path(config["plot_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "particle_distributions.png"
    plot_particle_distributions(
        real_kinematics,
        generated_kinematics,
        real_pdg,
        generated_pdg,
        feature_names,
        pdg_categories,
        output_path,
    )
    print(
        f"Compared {len(real_scaled)} real and {len(generated_scaled)} generated particles; "
        f"saved {output_path}"
    )
    return output_path


def inverse_minmax(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return ((values + 1.0) / 2.0) * (data_max - data_min) + data_min


def plot_particle_distributions(
    real_kinematics: np.ndarray,
    generated_kinematics: np.ndarray,
    real_pdg: np.ndarray,
    generated_pdg: np.ndarray,
    feature_names: list[str],
    pdg_categories: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(16, 16))

    for index, name in enumerate(feature_names):
        axis = axes.flat[index]
        lower = min(real_kinematics[:, index].min(), generated_kinematics[:, index].min())
        upper = max(real_kinematics[:, index].max(), generated_kinematics[:, index].max())
        if lower == upper:
            upper = lower + 1.0
        bins = np.linspace(lower, upper, 80)
        axis.hist(
            real_kinematics[:, index],
            bins=bins,
            density=True,
            histtype="step",
            label="FLUKA",
        )
        axis.hist(
            generated_kinematics[:, index],
            bins=bins,
            density=True,
            histtype="step",
            label="Generated",
        )
        axis.set_title(name)
        axis.grid(alpha=0.3)
        axis.legend()

    species_axis = axes.flat[len(feature_names)]
    positions = np.arange(len(pdg_categories))
    real_frequencies = np.asarray([(real_pdg == pdg).mean() for pdg in pdg_categories])
    generated_frequencies = np.asarray(
        [(generated_pdg == pdg).mean() for pdg in pdg_categories]
    )
    width = 0.4
    species_axis.bar(positions - width / 2, real_frequencies, width, label="FLUKA")
    species_axis.bar(
        positions + width / 2, generated_frequencies, width, label="Generated"
    )
    species_axis.set_xticks(positions)
    species_axis.set_xticklabels([str(int(pdg)) for pdg in pdg_categories], rotation=60)
    species_axis.set_title("Particle species")
    species_axis.set_ylabel("Particle-weighted frequency")
    species_axis.legend()

    for axis in axes.flat[len(feature_names) + 1 :]:
        axis.set_visible(False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_50.yaml")
    parser.add_argument("--generator-checkpoint")
    parser.add_argument("--multiplicity-checkpoint")
    parser.add_argument("--target-particles", type=int, default=100_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(
        load_yaml(arguments.config),
        arguments.generator_checkpoint,
        arguments.multiplicity_checkpoint,
        arguments.target_particles,
    )
