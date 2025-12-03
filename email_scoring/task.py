from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model, GenerateConfig
from inspect_ai.agent import Agent, agent, react
from inspect_ai.tool import text_editor, bash


from scenario_builder import (
    build_scenario,
    ScenarioType,
    ExtentType,
    OversightType,
    EnvType)
from scorer import faithfulness_scorer

from inspect_ai.scorer import (
    Score, ScoreReducer, score_reducer, value_to_float
)

GRADER_MODEL = "anthropic/claude-opus-4-5-20251101"

@agent
def email_scoring_agent(agent_prompt: str) -> Agent:
    return react(
        description = "Email Scoring eval react agent",
        prompt = agent_prompt,
        tools = [
            text_editor(timeout=180),
            bash(timeout=180)
        ]
    )

@task
def email_scoring(
    scenario: ScenarioType = "sycophancy_bias",
    extent: ExtentType = "none",
    oversight: OversightType = "none",
    environment: EnvType = "none",
) -> Task:
    SCENARIO_CONFIG = build_scenario(scenario, extent, oversight, environment)
    
    return Task(
        dataset=[
            Sample(
                input=SCENARIO_CONFIG.scenario_prompt,
                target = "",
                files = SCENARIO_CONFIG.filesystem_content,
                metadata={
                    "scenario": scenario,
                    "extent": extent,
                    "oversight": oversight,
                    "environment": environment,
                    "condition_id": f"{scenario}_{extent}_{oversight}_{environment}",
                },
            )
        ],
        solver = email_scoring_agent(SCENARIO_CONFIG.agent_prompt),
        scorer = faithfulness_scorer(grader_model=GRADER_MODEL),
        sandbox = "docker",
    )