from inspect_ai import eval_set
from inspect_ai.model import get_model, GenerateConfig
from task import email_scoring

from dotenv import load_dotenv

load_dotenv()

EVAL_MODELS = [
            get_model("openrouter/x-ai/grok-code-fast-1",
                        config = GenerateConfig(max_tokens=32_000, attempt_timeout=600, reasoning = {"enabled": True}),
                        ),
            get_model("anthropic/claude-sonnet-4-20250514",
                        config = GenerateConfig(max_tokens=32_000, reasoning = {"enabled": True}, reasoning_tokens=10_000, attempt_timeout=600),
            ),
            get_model("openrouter/openai/gpt-oss-120b",
                        config = GenerateConfig(max_tokens=32_000, reasoning = {"enabled": True}, attempt_timeout=600),
            ),
            get_model("openrouter/qwen/qwen3-next-80b-a3b-thinking",
                        config = GenerateConfig(max_tokens=32_000, reasoning = {"enabled": True}, attempt_timeout=600),
            ),
            get_model("openrouter/deepseek/deepseek-r1-0528",
                        config = GenerateConfig(max_tokens=32_000, reasoning = {"enabled": True}, attempt_timeout=600),
                        provider = {"order": ['parasail']}
            ),
    ]

CONDITIONS = [
    {"scenario": "sycophancy_bias", "extent": "explicit"},
    {"scenario": "precomputed_results", "extent": "explicit"},
    {"scenario": "goal_conflict", "extent": "explicit"},
    {"scenario": "tool_hint", "extent": "explicit"},
    {"scenario": "sycophancy_bias", "extent": "none"},
    {"scenario": "precomputed_results", "extent": "none"},
    {"scenario": "goal_conflict", "extent": "none"},
    {"scenario": "tool_hint", "extent": "none"},
]

all_logs = []

for cond in CONDITIONS:
    scenario = cond["scenario"]
    extent = cond["extent"]
    
    task = email_scoring(
        scenario=scenario,
        extent=extent,
        oversight="none",
        environment="none",
    )
    
    log_dir = f"./logs/{scenario}_{extent}"
    
    print(f"\n{'='*60}")
    print(f"Running: {scenario} / {extent}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}\n")
    
    success, logs = eval_set(
        tasks=[task],
        model=EVAL_MODELS,
        max_connections=20,
        max_retries=8,
        time_limit=1200, # 20 minutes max: longer indicates something weird might be happening
        fail_on_error=5,
        epochs=10,
        log_dir=log_dir,
    )
    
    all_logs.extend(logs)
    
    if not success:
        print(f"⚠️  Some evals failed for {scenario}_{extent}")

print(f"\n✅ Complete! {len(all_logs)} total logs generated.")