from __future__ import annotations

from pathlib import Path
import json
import os

import matplotlib

here = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(here / "outputs" / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rplan_room_graph_experiment import run_rplan_experiment


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    labels = ["Next-room acc", "Aliased acc"]
    flat_values = [
        metrics["flat_next_room_accuracy"],
        metrics["flat_aliased_transition_accuracy"],
    ]
    territorial_values = [
        metrics["territorial_next_room_accuracy"],
        metrics["territorial_aliased_transition_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], flat_values, width=width, color="#a0aec0", label="Flat room model")
    ax.bar([i + width / 2 for i in x], territorial_values, width=width, color="#dd6b20", label="Territorial room model")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("RPlan Room-Graph Prediction")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rplan_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = here / "outputs"
    metrics = run_rplan_experiment(output_dir)
    plot_metrics(metrics, output_dir)
    print("RPlan experiment finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
