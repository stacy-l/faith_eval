"""
Run the SimpleQA Verified task across selected OpenRouter reasoning models.
"""

from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig, get_model

from simpleqa.simpleqa import simpleqa_verified

# Load environment (expects OPENROUTER_API_KEY, etc. in project root .env)
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

EVAL_MODELS = [
    get_model(
        "openrouter/deepseek/deepseek-r1",
        config=GenerateConfig(
            max_tokens=2048,
            reasoning_tokens=4096,
            attempt_timeout=600,
        ),
        reasoning_enabled=True,
    ),
    get_model(
        "openrouter/x-ai/grok-3-mini",
        config=GenerateConfig(
            max_tokens=2048,
            reasoning_effort="medium",
            attempt_timeout=600,
        ),
        reasoning_enabled=True,
    ),
    get_model(
        "openrouter/anthropic/claude-haiku-4-5-20251001",
        config=GenerateConfig(
            max_tokens=2048,
            reasoning_tokens=4096,
            attempt_timeout=600,
        ),
        reasoning_enabled=True,
    ),
]

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

EPOCHS_PER_QUESTION = 3
TIME_LIMIT = 600
MAX_RETRIES = 6
MAX_CONNECTIONS = 12
LOG_DIR = "./logs/simpleqa_verified/baseline_set"


def run_eval():
    task = simpleqa_verified()
    success, logs = eval_set(
        tasks=[task],
        model=EVAL_MODELS,
        epochs=EPOCHS_PER_QUESTION,
        max_connections=MAX_CONNECTIONS,
        max_retries=MAX_RETRIES,
        time_limit=TIME_LIMIT,
        fail_on_error=3,
        log_dir=LOG_DIR,
    )

    if success:
        print(f"\n✅ Complete! {len(logs)} total logs generated.")
    else:
        print("\n⚠️  Some evaluations failed.")

    return logs


if __name__ == "__main__":
    run_eval()
