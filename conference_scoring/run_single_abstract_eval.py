"""
Run evaluation sets for the faithfulness study.

Usage:
    python run_eval.py --model claude --task noaff_nudge --epochs 10
    python run_eval.py --model deepseek --task mit_nudge --epochs 10
"""

import argparse
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model

from dotenv import load_dotenv
load_dotenv()

from task import (
    eval_single_noaff_nudge,
    eval_single_noaff_nonudge,
    eval_single_mit_nudge,
    eval_single_mit_nonudge,
)

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs" / "eval_single"

# Model configurations
MODELS = {
    "claude": {
        "model_id": "anthropic/claude-sonnet-4-20250514",
        "config": GenerateConfig(reasoning_tokens=4096),
        "borderline": "generated_abstracts/borderline_claude_combined.json",
    },
    "deepseek": {
        "model_id": "openrouter/deepseek/deepseek-r1-0528",
        "config": GenerateConfig(
            reasoning={"enabled": True},
            provider={"order": ["parasail"]},
            max_tokens=4096,
        ),
        "borderline": "generated_abstracts/borderline_deepseek_combined.json",
    },
    "olmo": {
        "model_id": "openrouter/allenai/olmo-3-7b-think",
        "config": GenerateConfig(
            reasoning={"enabled": True},
            max_tokens=32768,
        ),
        "borderline": "generated_abstracts/borderline_olmo_combined.json",
    },
}

# Task mapping
TASKS = {
    "noaff_nudge": eval_single_noaff_nudge,
    "noaff_nonudge": eval_single_noaff_nonudge,
    "mit_nudge": eval_single_mit_nudge,
    "mit_nonudge": eval_single_mit_nonudge,
}


def run_eval(
    model_key: str,
    task_key: str,
    epochs: int = 10,
):
    """Run evaluation for a specific model and task combination."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
    if task_key not in TASKS:
        raise ValueError(f"Unknown task: {task_key}. Choose from: {list(TASKS.keys())}")

    model_cfg = MODELS[model_key]
    task_fn = TASKS[task_key]

    # Set up paths
    borderline_path = str(BASE_DIR / model_cfg["borderline"])
    log_dir = LOGS_DIR / task_key / model_key
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get model with config
    model = get_model(model_cfg["model_id"], config=model_cfg["config"])

    # Create task
    task = task_fn(borderline_path=borderline_path)

    print(f"\n{'='*60}")
    print(f"EVAL: {task_key} | {model_key}")
    print(f"{'='*60}")
    print(f"Model: {model_cfg['model_id']}")
    print(f"Borderline abstracts: {borderline_path}")
    print(f"Epochs: {epochs}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Run evaluation
    logs = eval(
        tasks=task,
        model=model,
        epochs=epochs,
        log_dir=str(log_dir),
    )

    # Print results
    for log in logs:
        status = "✓" if log.status == "success" else "✗"
        print(f"\n  {status} {log.location}")

        if log.results and log.results.scores:
            for score in log.results.scores:
                acc = score.metrics.get("accuracy")
                if acc:
                    print(f"      {score.name}: accuracy={acc.value:.1%}")

    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE: {task_key} | {model_key}")
    print(f"{'='*60}\n")

    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation sets")
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(MODELS.keys()),
        help="Model to evaluate"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        choices=list(TASKS.keys()),
        help="Task to run"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of epochs (default: 10)"
    )

    args = parser.parse_args()

    run_eval(
        model_key=args.model,
        task_key=args.task,
        epochs=args.epochs,
    )
