"""Email Scoring eval task definition."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.agent import Agent, AgentPrompt, agent, react
from inspect_ai.tool import text_editor, bash

from scenario_builder import build_scenario, get_available_scenarios
from scorer import faithfulness_scorer

GRADER_MODEL = "anthropic/claude-opus-4-5-20251101"


@agent
def email_scoring_agent(agent_instructions: str) -> Agent:
    return react(
        description="Email scoring agent",
        prompt=AgentPrompt(instructions=agent_instructions),
        tools=[
            text_editor(timeout=180),
            bash(timeout=180)
        ]
    )


@task
def email_scoring(
    scenario_id: str = "cloud_computing_baseline",
) -> Task:
    """
    Email scoring faithfulness evaluation task.

    Args:
        scenario_id: The scenario to run. Use get_available_scenarios() to see options.
                    Examples:
                    - "cloud_computing_baseline" (no cue, reliability-focused task)
                    - "cloud_computing_sycophancy_cirrus" (CEO mentions Cirrus Tech)
                    - "cloud_computing_social_proof_cirrus" (industry adoption cue)

    Returns:
        Task configured for the specified scenario
    """
    scenario_config = build_scenario(scenario_id)

    return Task(
        dataset=[
            Sample(
                input=scenario_config.scenario_prompt,
                files=scenario_config.filesystem_content,
                metadata=scenario_config.metadata,
            )
        ],
        solver=email_scoring_agent(scenario_config.agent_instructions),
        scorer=faithfulness_scorer(grader_model=GRADER_MODEL),
        sandbox=("docker", "compose.yaml"),
    )


# Convenience function to list available scenarios
def list_scenarios() -> list[str]:
    """List all available scenario IDs for this eval."""
    return get_available_scenarios()
