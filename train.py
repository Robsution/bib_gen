"""Train the categorical-multiplicity plus conditioned SN-GAN baseline."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import BIBDataset
from src.models import BIBGenerator, CategoricalMultiplicity, DeepSetsDiscriminator
from src.monitoring import TrainingMonitor
from src.runtime import load_yaml, resolve_device, seed_everything


def train_multiplicity_model(
    config: dict,
    dataset: BIBDataset,
    parent_dim: int,
    max_daughters: int,
    device: torch.device,
    loader_generator: torch.Generator,
) -> CategoricalMultiplicity:
    """Fit the conditional categorical model with early stopping on training loss."""

    model = CategoricalMultiplicity(
        parent_dim=parent_dim,
        max_n=max_daughters,
        hidden_dim=config.get("multiplicity_hidden_dim", 128),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get("multiplicity_learning_rate", 1e-3)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("multiplicity_batch_size", 256),
        shuffle=True,
        generator=loader_generator,
    )

    max_epochs = config.get("multiplicity_epochs", 500)
    patience = config.get("multiplicity_patience", 10)
    min_delta = config.get("multiplicity_min_delta", 1e-4)
    best_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    print("Training the categorical multiplicity model")
    for epoch in range(max_epochs):
        batch_losses: list[float] = []
        for parent, _, _, normalized_n in dataloader:
            parent = parent.to(device)
            true_n = (normalized_n.to(device) * max_daughters).round().long().view(-1)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(parent), true_n)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = float(np.mean(batch_losses))
        print(f"Multiplicity epoch {epoch + 1}/{max_epochs}: loss={mean_loss:.5f}")

        if mean_loss < best_loss - min_delta:
            best_loss = mean_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_weights is None:
        raise RuntimeError("Multiplicity training completed without producing model weights")

    model.load_state_dict(best_weights)
    checkpoint = Path(config["save_dir"]) / "multiplicity_model.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"Saved multiplicity model to {checkpoint}")
    return model


def load_or_train_multiplicity(
    config: dict,
    dataset: BIBDataset,
    parent_dim: int,
    max_daughters: int,
    device: torch.device,
    loader_generator: torch.Generator,
    retrain: bool,
) -> CategoricalMultiplicity:
    model = CategoricalMultiplicity(
        parent_dim=parent_dim,
        max_n=max_daughters,
        hidden_dim=config.get("multiplicity_hidden_dim", 128),
    ).to(device)

    checkpoint_dir = Path(config["save_dir"])
    checkpoint = checkpoint_dir / "multiplicity_model.pt"
    legacy_checkpoint = checkpoint_dir / "oracle_weights.pth"
    existing_checkpoint = checkpoint if checkpoint.exists() else legacy_checkpoint

    if existing_checkpoint.exists() and not retrain:
        model.load_state_dict(
            torch.load(existing_checkpoint, map_location=device, weights_only=True)
        )
        print(f"Loaded multiplicity model from {existing_checkpoint}")
        return model

    return train_multiplicity_model(
        config,
        dataset,
        parent_dim,
        max_daughters,
        device,
        loader_generator,
    )


def train(config: dict, resume_epoch: int = 0, retrain_multiplicity: bool = False) -> None:
    seed = int(config.get("seed", 12345))
    loader_generator = seed_everything(seed)
    device = resolve_device(config)

    save_dir = Path(config["save_dir"])
    plot_dir = Path(config["plot_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    max_daughters = int(config["max_daughters"])
    dataset = BIBDataset(
        config["data_dir"],
        min_daughters=int(config.get("min_daughters", 0)),
        max_daughters=max_daughters,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.get("num_workers", 2)),
        generator=loader_generator,
    )

    sample_parent, sample_features, _, _ = dataset[0]
    if dataset.padded_size != max_daughters:
        raise ValueError(
            "The processed daughter-axis length must match max_daughters: "
            f"data has {dataset.padded_size}, config requests {max_daughters}. "
            "Build a filtered dataset with scripts.data.filter_dataset first."
        )

    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    kinematic_dim = int(config.get("kinematic_dim", 8))
    species_dim = feature_dim - kinematic_dim
    if species_dim <= 0:
        raise ValueError("feature_dim must exceed kinematic_dim")

    multiplicity_model = load_or_train_multiplicity(
        config,
        dataset,
        parent_dim,
        max_daughters,
        device,
        loader_generator,
        retrain_multiplicity,
    )
    multiplicity_model.eval().requires_grad_(False)

    generator = BIBGenerator(
        noise_dim=int(config["z_dim"]),
        parent_dim=parent_dim,
        max_daughters=max_daughters,
        kinematic_dim=kinematic_dim,
        species_dim=species_dim,
        hidden_dim=int(config.get("generator_hidden_dim", 256)),
    ).to(device)
    discriminator = DeepSetsDiscriminator(
        feature_dim=feature_dim,
        parent_dim=parent_dim,
        hidden_dim=int(config.get("discriminator_hidden_dim", 128)),
    ).to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(config["lr_g"]), betas=(0.0, 0.9)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=float(config["lr_d"]), betas=(0.0, 0.9)
    )
    monitor = TrainingMonitor(plot_dir)

    if resume_epoch > 0:
        generator_checkpoint = resolve_epoch_checkpoint(save_dir, "generator", resume_epoch)
        discriminator_checkpoint = resolve_epoch_checkpoint(
            save_dir, "discriminator", resume_epoch
        )
        generator.load_state_dict(
            torch.load(
                generator_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )
        discriminator.load_state_dict(
            torch.load(
                discriminator_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

    global_step = resume_epoch * len(dataloader)
    epochs = int(config["epochs"])
    noise_dim = int(config["z_dim"])
    generator_interval = int(config.get("n_critic", 1))
    checkpoint_interval = int(config.get("checkpoint_interval", 50))

    print(f"Training conditioned SN-GAN on {device}")
    for epoch in range(resume_epoch, epochs):
        d_losses: list[float] = []
        g_losses: list[float] = []
        score_gaps: list[float] = []
        real_scores: list[float] = []
        fake_scores: list[float] = []

        for real_parent, real_features, real_mask, real_n in dataloader:
            real_parent = real_parent.to(device)
            real_features = real_features.to(device)
            real_mask = real_mask.to(device)
            real_n = real_n.to(device).float().unsqueeze(1)
            batch_size = real_parent.shape[0]

            with torch.no_grad():
                count_probabilities = torch.softmax(multiplicity_model(real_parent), dim=-1)
                target_n_integer = torch.multinomial(count_probabilities, num_samples=1).float()
                target_n = target_n_integer / max_daughters

                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_features, fake_mask = generator(noise, real_parent, target_n)

            real_score = discriminator(real_features, real_parent, real_mask, real_n)
            fake_score = discriminator(fake_features, real_parent, fake_mask, target_n)
            discriminator_loss = F.relu(1.0 - real_score).mean() + F.relu(
                1.0 + fake_score
            ).mean()

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            d_losses.append(discriminator_loss.item())
            score_gaps.append((real_score.mean() - fake_score.mean()).item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())

            if global_step % generator_interval == 0:
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_features, fake_mask = generator(noise, real_parent, target_n)
                generator_loss = -discriminator(
                    fake_features, real_parent, fake_mask, target_n
                ).mean()

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
                g_losses.append(generator_loss.item())

            global_step += 1

        mean_d_loss = float(np.mean(d_losses))
        mean_g_loss = float(np.mean(g_losses)) if g_losses else float("nan")
        mean_score_gap = float(np.mean(score_gaps))
        mean_real_score = float(np.mean(real_scores))
        mean_fake_score = float(np.mean(fake_scores))

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"d_loss={mean_d_loss:.5f}, g_loss={mean_g_loss:.5f}, "
            f"score_gap={mean_score_gap:.5f}"
        )
        monitor.update(
            mean_d_loss,
            mean_g_loss,
            mean_score_gap,
            mean_real_score,
            mean_fake_score,
        )
        monitor.plot(epoch + 1)

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(generator.state_dict(), save_dir / f"generator_epoch_{epoch + 1}.pt")
            torch.save(
                discriminator.state_dict(), save_dir / f"discriminator_epoch_{epoch + 1}.pt"
            )


def resolve_epoch_checkpoint(directory: Path, model_name: str, epoch: int) -> Path:
    """Return a current ``.pt`` checkpoint or its earlier ``.pth`` equivalent."""

    candidates = (
        directory / f"{model_name}_epoch_{epoch}.pt",
        directory / f"{model_name}_epoch_{epoch}.pth",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {model_name} checkpoint found for epoch {epoch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_50.yaml")
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument(
        "--retrain-multiplicity",
        action="store_true",
        help="Ignore an existing multiplicity checkpoint and fit it again.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        load_yaml(arguments.config),
        resume_epoch=arguments.resume_epoch,
        retrain_multiplicity=arguments.retrain_multiplicity,
    )
