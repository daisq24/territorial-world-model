from __future__ import annotations

from pathlib import Path
import json
import os

here = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(here / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rplan_navigation_experiment import run_rplan_navigation_experiment


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    labels = ["Navigation success"]
    flat_values = [metrics["flat_navigation_success"]]
    territorial_values = [metrics["territorial_navigation_success"]]
    neural_values = [metrics["neural_circuit_navigation_success"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = range(len(labels))
    width = 0.24
    ax.bar([i - width for i in x], flat_values, width=width, color="#a0aec0", label="Flat world model")
    ax.bar(list(x), territorial_values, width=width, color="#2f855a", label="Territorial world model")
    ax.bar([i + width for i in x], neural_values, width=width, color="#b83280", label="Neural-circuit model")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("RPlan Indoor Navigation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rplan_navigation_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = here / "outputs"
    metrics = run_rplan_navigation_experiment(output_dir)
    plot_metrics(metrics, output_dir)
    print("RPlan navigation experiment finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
