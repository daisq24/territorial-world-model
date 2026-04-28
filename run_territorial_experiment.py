from __future__ import annotations

import os
from pathlib import Path

here = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(here / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from territorial_nav_experiment import run_experiment


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    labels = [
        "Next-state acc",
        "Aliased-state acc",
        "Cross-boundary acc",
        "Planning success",
    ]
    flat_values = [
        metrics["flat_next_state_accuracy"],
        metrics["flat_aliased_state_accuracy"],
        metrics["flat_cross_boundary_accuracy"],
        metrics["flat_planning_success"],
    ]
    territorial_values = [
        metrics["territorial_next_state_accuracy"],
        metrics["territorial_aliased_state_accuracy"],
        metrics["territorial_cross_boundary_accuracy"],
        metrics["territorial_planning_success"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], flat_values, width=width, label="Flat model", color="#9aa5b1")
    ax.bar([i + width / 2 for i in x], territorial_values, width=width, label="Territorial model", color="#2b6cb0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Territorial vs Flat World Model")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = here / "outputs"
    metrics = run_experiment(output_dir)
    plot_metrics(metrics, output_dir)

    print("Experiment finished.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
