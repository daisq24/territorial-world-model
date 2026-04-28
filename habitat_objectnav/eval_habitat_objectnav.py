from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


DEFAULT_DATA_PATH = "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "habitat_objectnav_metrics.json"


def _load_habitat_config(config_path: str):
    from habitat.config import read_write
    from habitat.config.default import get_config

    return get_config, read_write


def _episode_summary(env, metrics: dict[str, Any], episode_index: int) -> dict[str, Any]:
    episode = getattr(env, "current_episode", None)
    return {
        "episode_index": episode_index,
        "episode_id": getattr(episode, "episode_id", None),
        "scene_id": getattr(episode, "scene_id", None),
        "object_category": getattr(episode, "object_category", None),
        "geodesic_distance": getattr(episode, "info", {}).get("geodesic_distance")
        if episode is not None
        else None,
        "metrics": metrics,
    }


def _aggregate_episode_metrics(episode_metrics: list[dict[str, Any]]) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    if not episode_metrics:
        return aggregates

    metric_names = sorted(
        {
            metric_name
            for item in episode_metrics
            for metric_name, metric_value in item["metrics"].items()
            if isinstance(metric_value, (int, float))
        }
    )
    for metric_name in metric_names:
        values = [
            float(item["metrics"][metric_name])
            for item in episode_metrics
            if isinstance(item["metrics"].get(metric_name), (int, float))
        ]
        if values:
            aggregates[metric_name] = mean(values)
    return aggregates


def evaluate(
    config_path: str,
    num_episodes: int,
    split: str,
    data_path: str,
    scene_dataset: str | None = None,
    max_steps_per_episode: int | None = None,
    log_every: int = 1,
    use_openai_verifier: bool = False,
    openai_model: str = "gpt-4.1-mini",
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    try:
        import habitat
    except Exception as exc:
        raise RuntimeError(
            "Habitat is not installed yet. Install habitat-lab / habitat-sim first."
        ) from exc

    get_config, read_write = _load_habitat_config(config_path)
    from policy import NeuralCircuitObjectNavPolicy

    cfg = get_config(config_path)
    with read_write(cfg):
        cfg.habitat.dataset.split = split
        cfg.habitat.dataset.data_path = data_path
        if scene_dataset is not None:
            cfg.habitat.simulator.scene_dataset = scene_dataset

    env = habitat.Env(config=cfg)
    policy = NeuralCircuitObjectNavPolicy(
        use_openai_verifier=use_openai_verifier,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
    )
    episode_metrics: list[dict[str, Any]] = []

    try:
        episodes_run = 0
        while episodes_run < num_episodes:
            observations = env.reset()
            policy.reset()
            current_episode = getattr(env, "current_episode", None)
            policy.set_goal(getattr(current_episode, "object_category", None))
            done = False
            steps = 0
            while not done:
                action = policy.act(observations)
                observations = env.step({"action": action})
                done = env.episode_over
                steps += 1
                if max_steps_per_episode is not None and steps >= max_steps_per_episode and not done:
                    observations = env.step({"action": 0})
                    done = env.episode_over
            metrics = env.get_metrics()
            summary = _episode_summary(env, metrics, episodes_run)
            summary["steps"] = steps
            episode_metrics.append(summary)
            episodes_run += 1
            if log_every > 0 and (episodes_run % log_every == 0 or episodes_run == num_episodes):
                print(
                    json.dumps(
                        {
                            "progress": f"{episodes_run}/{num_episodes}",
                            "episode_id": summary["episode_id"],
                            "object_category": summary["object_category"],
                            "steps": steps,
                            "success": metrics.get("success", None),
                            "spl": metrics.get("spl", None),
                            "distance_to_goal": metrics.get("distance_to_goal", None),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        env.close()

    aggregates = _aggregate_episode_metrics(episode_metrics)
    return {
        "config_path": config_path,
        "split": split,
        "data_path": data_path,
        "scene_dataset": scene_dataset,
        "use_openai_verifier": use_openai_verifier,
        "openai_model": openai_model if use_openai_verifier else None,
        "episodes": len(episode_metrics),
        "aggregates": aggregates,
        "success_rate": aggregates.get("success", 0.0),
        "episodes_detail": episode_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--scene-dataset", default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--use-openai-verifier", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    metrics = evaluate(
        config_path=args.config,
        num_episodes=args.episodes,
        split=args.split,
        data_path=args.data_path,
        scene_dataset=args.scene_dataset,
        max_steps_per_episode=args.max_steps_per_episode,
        log_every=args.log_every,
        use_openai_verifier=args.use_openai_verifier,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
