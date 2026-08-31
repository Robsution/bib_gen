"""Compare simple conditional models for daughter multiplicity."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import BIBDataset
from src.models.multiplicity import (
    CategoricalMultiplicity,
    DequantizedGaussianMultiplicity,
    GaussianMixtureMultiplicity,
    NegativeBinomialMultiplicity,
)
from src.runtime import load_yaml, resolve_device, seed_everything


def wasserstein_1_from_histograms(
    predicted_probabilities: torch.Tensor,
    true_n: torch.Tensor,
    max_daughters: int,
) -> float:
    """Compute discrete one-dimensional Wasserstein distance for one batch."""

    probabilities = predicted_probabilities / (
        predicted_probabilities.sum(dim=-1, keepdim=True) + 1e-8
    )
    predicted_histogram = probabilities.mean(dim=0)
    true_histogram = torch.bincount(
        true_n.long(), minlength=max_daughters + 1
    ).float() / len(true_n)
    predicted_cdf = torch.cumsum(predicted_histogram, dim=0)
    true_cdf = torch.cumsum(true_histogram, dim=0)
    return torch.abs(predicted_cdf - true_cdf).sum().item()


def plot_history(history: dict, output_dir: Path) -> None:
    epochs = range(1, len(next(iter(history.values()))["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, metrics in history.items():
        axes[0].plot(epochs, metrics["wasserstein_1"], label=name)
        axes[1].plot(epochs, metrics["loss"], label=name)

    axes[0].set_title("Multiplicity distribution error")
    axes[0].set_ylabel("Wasserstein-1 distance [particles]")
    axes[1].set_title("Training objective")
    axes[1].set_ylabel("Loss or negative log-likelihood")
    axes[1].set_yscale("symlog")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.4)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "multiplicity_comparison.png", dpi=200)
    plt.close(figure)


def train_models(config: dict) -> None:
    seed = int(config.get("seed", 12345))
    loader_generator = seed_everything(seed)
    device = resolve_device(config)
    max_daughters = int(config["max_daughters"])
    output_dir = Path(config["plot_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BIBDataset(
        config["data_dir"],
        min_daughters=int(config.get("min_daughters", 0)),
        max_daughters=max_daughters,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    parent_dim = dataset[0][0].shape[0]
    hidden_dim = int(config.get("multiplicity_hidden_dim", 128))

    models = {
        "Categorical": CategoricalMultiplicity(parent_dim, max_daughters, hidden_dim),
        "Negative binomial": NegativeBinomialMultiplicity(parent_dim, hidden_dim),
        "Gaussian mixture": GaussianMixtureMultiplicity(
            parent_dim,
            components=int(config.get("mixture_components", 5)),
            hidden_dim=hidden_dim,
        ),
        "Dequantized Gaussian": DequantizedGaussianMultiplicity(parent_dim, hidden_dim),
    }
    models = {name: model.to(device) for name, model in models.items()}
    learning_rate = float(config.get("multiplicity_learning_rate", 1e-3))
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=learning_rate)
        for name, model in models.items()
    }
    history = {
        name: {"loss": [], "wasserstein_1": []}
        for name in models
    }

    count_grid = torch.arange(max_daughters + 1, device=device).float().unsqueeze(0)
    mixture_grid = torch.arange(max_daughters + 1, device=device).float().view(-1, 1, 1)

    for epoch in range(int(config.get("multiplicity_epochs", 50))):
        epoch_metrics = {
            name: {"loss": [], "wasserstein_1": []}
            for name in models
        }

        for parent, _, _, normalized_n in dataloader:
            parent = parent.to(device)
            true_n = (normalized_n.to(device) * max_daughters).round()

            categorical = models["Categorical"]
            optimizers["Categorical"].zero_grad()
            logits = categorical(parent)
            loss = nn.functional.cross_entropy(logits, true_n.long())
            loss.backward()
            optimizers["Categorical"].step()
            probabilities = torch.softmax(logits.detach(), dim=-1)
            _record(epoch_metrics["Categorical"], loss, probabilities, true_n, max_daughters)

            negative_binomial = models["Negative binomial"]
            optimizers["Negative binomial"].zero_grad()
            total_count, nb_logits = negative_binomial(parent)
            distribution = torch.distributions.NegativeBinomial(total_count, logits=nb_logits)
            loss = -distribution.log_prob(true_n.unsqueeze(1)).mean()
            loss.backward()
            optimizers["Negative binomial"].step()
            probabilities = torch.exp(distribution.log_prob(count_grid)).detach()
            _record(
                epoch_metrics["Negative binomial"],
                loss,
                probabilities,
                true_n,
                max_daughters,
            )

            mixture = models["Gaussian mixture"]
            optimizers["Gaussian mixture"].zero_grad()
            weights, means, scales = mixture(parent)
            distribution = torch.distributions.Normal(means, scales)
            component_log_probabilities = distribution.log_prob(true_n.unsqueeze(1))
            loss = -torch.logsumexp(
                torch.log(weights + 1e-8) + component_log_probabilities, dim=-1
            ).mean()
            loss.backward()
            optimizers["Gaussian mixture"].step()
            grid_log_probabilities = distribution.log_prob(mixture_grid).permute(1, 0, 2)
            probabilities = (
                weights.unsqueeze(1) * torch.exp(grid_log_probabilities)
            ).sum(dim=-1).detach()
            _record(
                epoch_metrics["Gaussian mixture"],
                loss,
                probabilities,
                true_n,
                max_daughters,
            )

            gaussian = models["Dequantized Gaussian"]
            optimizers["Dequantized Gaussian"].zero_grad()
            mean, scale = gaussian(parent)
            dequantized_n = true_n.unsqueeze(1) + torch.rand_like(mean) - 0.5
            distribution = torch.distributions.Normal(mean, scale)
            loss = -distribution.log_prob(dequantized_n).mean()
            loss.backward()
            optimizers["Dequantized Gaussian"].step()
            probabilities = torch.exp(distribution.log_prob(count_grid)).detach()
            _record(
                epoch_metrics["Dequantized Gaussian"],
                loss,
                probabilities,
                true_n,
                max_daughters,
            )

        summary = []
        for name in models:
            mean_loss = float(np.mean(epoch_metrics[name]["loss"]))
            mean_w1 = float(np.mean(epoch_metrics[name]["wasserstein_1"]))
            history[name]["loss"].append(mean_loss)
            history[name]["wasserstein_1"].append(mean_w1)
            summary.append(f"{name}: loss={mean_loss:.4f}, W1={mean_w1:.4f}")

        print(f"Epoch {epoch + 1}: " + " | ".join(summary))
        plot_history(history, output_dir)


def _record(
    metrics: dict[str, list[float]],
    loss: torch.Tensor,
    probabilities: torch.Tensor,
    true_n: torch.Tensor,
    max_daughters: int,
) -> None:
    metrics["loss"].append(loss.item())
    metrics["wasserstein_1"].append(
        wasserstein_1_from_histograms(probabilities, true_n, max_daughters)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiplicity_50.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_models(load_yaml(arguments.config))
