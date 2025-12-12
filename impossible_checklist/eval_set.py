"""
Evaluation runner for the Impossible Checklist evaluation.

Runs all experimental conditions across configured models with
multiple epochs for statistical power.
"""

from inspect_ai import eval_set
from inspect_ai.model import get_model, GenerateConfig
from pathlib import Path
from dotenv import load_dotenv

from task import impossible_checklist
from scenario_builder import get_experimental_conditions

# Load .env from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

# =============================================================================
# Model Configuration
# =============================================================================

EVAL_MODELS = [
    get_model(
        "anthropic/claude-sonnet-4-20250514",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            reasoning_tokens=10_000,
            attempt_timeout=600,
        ),
    ),
    get_model(
        "openrouter/x-ai/grok-code-fast-1",
        config=GenerateConfig(
            max_tokens=32_000,
            attempt_timeout=600,
            reasoning={"enabled": True},
        ),
    ),
    get_model(
        "openrouter/minimax/minimax-m2",
        config=GenerateConfig(
            max_tokens=32_000,
            attempt_timeout=600,
            reasoning={"enabled": True},
        ),
        provider={"order": ["minimax"]},
    ),
    get_model(
        "openrouter/moonshotai/kimi-k2-thinking",
        config=GenerateConfig(
            max_tokens=32_000,
            attempt_timeout=600,
            reasoning={"enabled": True},
        ),
        provider={"order": ["moonshotai"]},
    ),
    get_model(
        "anthropic/claude-haiku-4-5-20251001",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            reasoning_tokens=10_000,
            attempt_timeout=600,
        ),
    ),
    get_model(
        "openrouter/openai/gpt-oss-20b",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            attempt_timeout=600,
        ),
    ),
    get_model(
        "openrouter/openai/gpt-oss-120b",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            attempt_timeout=600,
        ),
    ),
    get_model(
        "openrouter/qwen/qwen3-next-80b-a3b-thinking",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            attempt_timeout=600,
        ),
    ),
    get_model(
        "openrouter/deepseek/deepseek-r1-0528",
        config=GenerateConfig(
            max_tokens=32_000,
            reasoning={"enabled": True},
            attempt_timeout=600,
        ),
        provider={"order": ["parasail"]},
    ),
]

# =============================================================================
# Evaluation Configuration
# =============================================================================

RUNS_PER_CONDITION = 25  # Recommended 20-30 for statistical power

# =============================================================================
# Run Evaluation
# =============================================================================

if __name__ == "__main__":
    conditions = get_experimental_conditions()
    all_logs = []

    for cond in conditions:
        pressure = cond["pressure"]
        eval_awareness = cond["eval_awareness"]

        task = impossible_checklist(
            pressure=pressure,
            eval_awareness=eval_awareness,
        )

        log_dir = f"./logs/pressure_{pressure}_eval_{eval_awareness}"

        print(f"\n{'='*60}")
        print(f"Running: pressure={pressure} / eval_awareness={eval_awareness}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        success, logs = eval_set(
            tasks=[task],
            model=EVAL_MODELS,
            max_connections=20,
            max_retries=8,
            time_limit=600,  # 10 minutes max per run
            fail_on_error=5,
            epochs=RUNS_PER_CONDITION,
            log_dir=log_dir,
        )

        all_logs.extend(logs)

        if not success:
            print(f"Warning: Some evals failed for pressure_{pressure}_eval_{eval_awareness}")

    print(f"\nComplete! {len(all_logs)} total logs generated.")
