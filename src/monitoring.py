"""Training diagnostics for the SN-GAN baseline."""

from pathlib import Path

import matplotlib.pyplot as plt


class TrainingMonitor:
    """Accumulate epoch summaries and write a compact training dashboard."""

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, list[float]] = {
            "d_loss": [],
            "g_loss": [],
            "score_gap": [],
            "real_score": [],
            "fake_score": [],
        }

    def update(
        self,
        d_loss: float,
        g_loss: float,
        score_gap: float,
        real_score: float,
        fake_score: float,
    ) -> None:
        self.history["d_loss"].append(d_loss)
        self.history["g_loss"].append(g_loss)
        self.history["score_gap"].append(score_gap)
        self.history["real_score"].append(real_score)
        self.history["fake_score"].append(fake_score)

    def plot(self, epoch: int) -> None:
        """Overwrite ``training_dashboard.png`` with all recorded epochs."""

        epochs = range(1, len(self.history["d_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"SN-GAN training diagnostics through epoch {epoch}")

        axes[0, 0].plot(epochs, self.history["d_loss"], label="Discriminator")
        axes[0, 0].plot(epochs, self.history["g_loss"], label="Generator")
        axes[0, 0].set_title("Hinge losses")
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.history["score_gap"])
        axes[0, 1].set_title("Mean discriminator score gap")

        axes[1, 0].plot(epochs, self.history["real_score"], label="Real")
        axes[1, 0].plot(epochs, self.history["fake_score"], label="Generated")
        axes[1, 0].axhline(1.0, linestyle="--", alpha=0.5)
        axes[1, 0].axhline(-1.0, linestyle="--", alpha=0.5)
        axes[1, 0].set_title("Discriminator scores")
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.history["g_loss"])
        axes[1, 1].set_title("Generator loss")

        for axis in axes.flat:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.save_dir / "training_dashboard.png", dpi=150)
        plt.close(fig)
