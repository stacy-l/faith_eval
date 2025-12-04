from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.agent import Agent, agent, react
from inspect_ai.tool import text_editor, bash

from scenario_builder import build_scenario, ScenarioType, ExtentType
from scorer import faithfulness_scorer

GRADER_MODEL = "anthropic/claude-opus-4-5-20251101"

@agent
def email_scoring_agent(agent_prompt: str) -> Agent:
    return react(
        description = "Email Scoring eval react agent",
        tools = [
            text_editor(timeout=180),
            bash(timeout=180)
        ]
    )

@task
def email_scoring(
    scenario: ScenarioType = "metadata_eco",
    extent: ExtentType = "none",
) -> Task:
    SCENARIO_CONFIG = build_scenario(scenario, extent)
    
    return Task(
        dataset=[
            Sample(
                input=SCENARIO_CONFIG.scenario_prompt,
                files = SCENARIO_CONFIG.filesystem_content,
                metadata={
                    "scenario": scenario,
                    "extent": extent,
                    "condition_id": f"{scenario}_{extent}",
                },
            )
        ],
        solver = email_scoring_agent(SCENARIO_CONFIG.agent_prompt),
        scorer = faithfulness_scorer(grader_model=GRADER_MODEL),
        sandbox=("docker", "compose.yaml"),
    )
