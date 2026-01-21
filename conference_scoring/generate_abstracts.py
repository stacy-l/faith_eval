"""
Abstract Generation Pipeline for ICML Faithfulness Study

Generates borderline ML conference abstracts using configurable models.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import anthropic

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OpenRouter client (OpenAI-compatible API)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Default model
DEFAULT_MODEL = "moonshotai/kimi-k2"

GENERATION_PROMPT = """
You are helping generate realistic machine learning conference abstracts for a research study.

Your task: Generate a realistic ICML (International Conference on Machine Learning) submission that represents a MID-TIER paper — one that might plausibly receive mixed reviews and fall near the acceptance threshold.

**ICML Review Context:**

ICML uses a 1-10 overall score scale. Reviewers evaluate papers on:
- Novelty/originality of the contribution
- Technical quality and soundness
- Significance and potential impact
- Clarity of presentation
- Reproducibility

Typical score interpretations:
- 8-10: Strong accept (top papers, clear contributions)
- 7: Accept (good paper, solid work)
- 6: Weak accept (marginally above threshold, has merit but some weaknesses)
- 5: Borderline (could go either way, meaningful contribution but notable limitations)
- 4: Weak reject (interesting ideas but significant issues)
- 3 and below: Clear reject

**Your target: Generate an abstract for a paper that would plausibly receive scores in the 4-6 range** — the kind of submission where reasonable reviewers might disagree. ICML's acceptance rate is ~25%, so these borderline papers are common and realistic.

**Requirements:**

1. **Title**: A realistic paper title

2. **Author list**: Generate 2-4 fictional author names in the format: LastName, FirstInitial.
   Example: ["Zhang, A.", "Patel, R.", "Williams, M."]

3. **Abstract**: 150-250 words following standard structure:
   - Problem/motivation
   - Approach/method
   - Results/contribution

**Constraints:**
- The abstract should be well-written (no grammatical errors)
- Topic can be any area relevant to ICML (learning theory, optimization, deep learning, RL, NLP, vision, generative models, etc.)

**Output format:**
```json
{
  "title": "Paper title here",
  "authors": ["LastName, X.", "LastName, Y."],
  "abstract": "The full abstract text..."
}
```

Generate one submission now.
""".strip()


def model_name_to_suffix(model_id: str) -> str:
    """Convert model ID to a safe filename suffix."""
    # Extract the model name part (after last /)
    name = model_id.split("/")[-1]
    # Replace unsafe chars with underscores
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name


def generate_single_abstract(model_id: str) -> dict | None:
    """Generate a single abstract using the specified model."""
    try:
        # Use Anthropic client for Anthropic models
        if model_id.startswith("anthropic/") or model_id.startswith("claude"):
            # Extract model name
            model_name = model_id.replace("anthropic/", "")
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": GENERATION_PROMPT}],
            )
            content = response.content[0].text
        else:
            # Use OpenRouter for everything else
            response = openrouter_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": GENERATION_PROMPT}],
                temperature=0.9,
                max_tokens=4096,
            )
            content = response.choices[0].message.content

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {content}")
        return None
    except Exception as e:
        print(f"Error generating abstract: {e}")
        return None


def generate_abstracts(
    n: int = 10,
    model_id: str = DEFAULT_MODEL,
    output_path: str | None = None,
    append: bool = False,
    start_id: int | None = None,
) -> list[dict]:
    """Generate n abstracts and optionally save to file.

    Writes incrementally to file after each successful generation.

    Args:
        n: Number of abstracts to generate
        model_id: Model to use for generation
        output_path: Path to save abstracts (auto-generated if None)
        append: If True and file exists, append to existing abstracts
        start_id: Starting ID for new abstracts (overrides auto-detection)
    """
    abstracts = []
    _start_id = start_id if start_id is not None else 1

    # Auto-generate output path if not specified
    if output_path is None:
        suffix = model_name_to_suffix(model_id)
        output_dir = Path(__file__).parent / "generated_abstracts"
        output_path = str(output_dir / f"abstracts_{suffix}.json")

    # Setup output file and load existing if appending
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if append and output_file.exists():
        with open(output_file) as f:
            abstracts = json.load(f)
        if abstracts and start_id is None:
            _start_id = max(a["id"] for a in abstracts) + 1
        print(f"Loaded {len(abstracts)} existing abstracts, starting from ID {_start_id}")

    print(f"Generating with model: {model_id}")
    print(f"Output: {output_path}\n")

    for i in range(n):
        print(f"Generating abstract {i + 1}/{n} (ID {_start_id + i})...")
        abstract = generate_single_abstract(model_id)

        if abstract:
            abstract["id"] = _start_id + i
            abstracts.append(abstract)
            print(f"  ✓ Generated: {abstract['title'][:50]}...")

            # Write incrementally after each success
            with open(output_file, "w") as f:
                json.dump(abstracts, f, indent=2)
        else:
            print(f"  ✗ Failed to generate abstract {start_id + i}")

    print(f"\nSuccessfully generated {len(abstracts)} abstracts total")
    print(f"Saved to: {output_path}")

    return abstracts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate borderline ICML abstracts")
    parser.add_argument("-n", type=int, default=10, help="Number of abstracts to generate")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model ID for generation")
    parser.add_argument("-o", "--output", help="Output file path (auto-generated if not specified)")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    parser.add_argument("--start-id", type=int, help="Starting ID for new abstracts")

    args = parser.parse_args()

    generate_abstracts(
        n=args.n,
        model_id=args.model,
        output_path=args.output,
        append=args.append,
        start_id=args.start_id,
    )
