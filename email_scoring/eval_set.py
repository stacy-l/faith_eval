from inspect_ai import eval_set
from inspect_ai.model import get_model, GenerateConfig
from task import email_scoring

from scenario_definitions import list_scenarios

from dotenv import load_dotenv
load_dotenv()

EVAL_MODELS = [
            get_model("anthropic/claude-sonnet-4-20250514",
                        config = GenerateConfig(max_tokens=32_000, reasoning = {"enabled": True}, reasoning_tokens=10_000, attempt_timeout=600),
            ),
            get_model("openrouter/x-ai/grok-3-mini",
                        config = GenerateConfig(max_tokens=32_000, attempt_timeout=600, reasoning = {"enabled": True}),
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

# =============================================================================
# Evaluation Configuration
# =============================================================================

RUNS_PER_CONDITION = 10  # Full run for analysis
TIME_LIMIT = 900  # 15 minutes

# Specify which scenarios to run (or use list_scenarios() for all)
# Paired comparisons: email/sysprompt vs agent_instructions
SCENARIOS_TO_RUN = [
    # Sycophancy: email vs agent
    "cloud_computing_sycophancy_email",
    "cloud_computing_sycophancy_agent",
    # Social proof: email vs agent
    "cloud_computing_social_proof_email",
    "cloud_computing_social_proof_agent",
    # Eco: email vs agent
    "cloud_computing_eco_email",
    "cloud_computing_eco_agent",
    # Negative: email vs agent
    "cloud_computing_negative_email",
    "cloud_computing_negative_agent",
]

# Uncomment to run all scenarios:
# SCENARIOS_TO_RUN = list_scenarios()

all_logs = []

for scenario_id in SCENARIOS_TO_RUN:
    task = email_scoring(scenario_id=scenario_id)

    log_dir = f"./logs/{scenario_id}"

    print(f"\n{'='*60}")
    print(f"Running: {scenario_id}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}\n")

    success, logs = eval_set(
        tasks=[task],
        model=EVAL_MODELS,
        max_connections=20,
        max_retries=8,
        time_limit=TIME_LIMIT,
        fail_on_error=5,
        epochs=RUNS_PER_CONDITION,
        log_dir=log_dir,
    )

    all_logs.extend(logs)

    if not success:
        print(f"Warning: Some evals failed for {scenario_id}")

print(f"\nComplete! {len(all_logs)} total logs generated.")
