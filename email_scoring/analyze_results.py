#!/usr/bin/env python3
"""
Analyze eval results for the email scoring faithfulness eval.

Outputs:
1. Table of nudge vendor selection rates by model and condition
2. P(mentioned_nudge_in_cot | selected_nudge_vendor) - consistency metric
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

LOG_DIR = Path("logs")

CONDITIONS = [
    ("sycophancy_bias", "none"),
    ("sycophancy_bias", "explicit"),
    ("metadata_eco", "none"),
    ("metadata_eco", "explicit"),
    ("metadata_anti_eco", "none"),
    ("metadata_anti_eco", "explicit"),
]

def load_all_results():
    """Load all logs.json files and extract per-model results."""
    results = []

    for scenario, extent in CONDITIONS:
        condition_dir = LOG_DIR / f"{scenario}_{extent}"
        logs_file = condition_dir / "logs.json"

        if not logs_file.exists():
            print(f"Warning: {logs_file} not found")
            continue

        with open(logs_file) as f:
            data = json.load(f)

        for eval_file, eval_data in data.items():
            model = eval_data["eval"]["model"]
            # Clean up model name for display
            model_short = model.split("/")[-1]

            # Get the reduced scores (aggregated over epochs)
            reductions = eval_data.get("reductions", [])
            if not reductions:
                continue

            scores = reductions[0]["samples"][0]["value"]

            results.append({
                "scenario": scenario,
                "extent": extent,
                "condition": f"{scenario}_{extent}",
                "model": model_short,
                "model_full": model,
                "selected_nudge_vendor": scores["selected_nudge_vendor"],
                "read_vendor_emails": scores["read_vendor_emails"],
                "hallucination": scores["hallucination"],
                "mentioned_nudge_in_cot": scores["mentioned_nudge_in_cot"],
                "unfaithful_reasoning": scores["unfaithful_reasoning"],
                "n_samples": eval_data["results"]["total_samples"],
            })

    return pd.DataFrame(results)


def analyze_nudge_selection(df):
    """Q1: How often did models select the nudge vendor at baseline vs with nudge?"""
    print("=" * 80)
    print("Q1: NUDGE VENDOR SELECTION RATE BY MODEL AND CONDITION")
    print("=" * 80)

    for scenario in ["sycophancy_bias", "metadata_eco"]:
        print(f"\n### {scenario.upper()} ###")
        scenario_df = df[df["scenario"] == scenario]

        pivot = scenario_df.pivot_table(
            index="model",
            columns="extent",
            values="selected_nudge_vendor",
            aggfunc="mean"
        )

        # Reorder columns
        if "none" in pivot.columns and "explicit" in pivot.columns:
            pivot = pivot[["none", "explicit"]]
            pivot["delta"] = pivot["explicit"] - pivot["none"]

        print(pivot.round(3).to_string())
        print()

    # Combined summary table
    print("\n### SUMMARY: NUDGE SELECTION RATES ###")
    summary = df.pivot_table(
        index="model",
        columns=["scenario", "extent"],
        values="selected_nudge_vendor",
        aggfunc="mean"
    ).round(3)
    print(summary.to_string())

    return summary


def analyze_cot_consistency(df):
    """
    Q2: P(mentioned_nudge_in_cot | selected_nudge_vendor)

    This measures: when a model picks the nudge vendor, how often does it
    acknowledge the nudge in its reasoning?

    Note: This is an approximation since we have aggregate rates, not per-sample data.
    For precise calculation, we'd need sample-level data.
    """
    print("\n" + "=" * 80)
    print("Q2: COT CONSISTENCY - P(mentioned_nudge | picked_nudge)")
    print("=" * 80)

    # For the explicit conditions only (baseline has no nudge to mention)
    explicit_df = df[df["extent"] == "explicit"]

    print("\nFor EXPLICIT nudge conditions:")
    print("Model | Scenario | P(pick_nudge) | P(mention_nudge) | P(mention|pick) approx")
    print("-" * 80)

    consistency_results = []

    for _, row in explicit_df.iterrows():
        p_pick = row["selected_nudge_vendor"]
        p_mention = row["mentioned_nudge_in_cot"]

        # Approximation: if mention implies pick (reasonable assumption),
        # then P(mention|pick) ≈ P(mention) / P(pick)
        # This is only valid if P(mention AND pick) ≈ P(mention)
        if p_pick > 0:
            p_mention_given_pick = min(p_mention / p_pick, 1.0)
        else:
            p_mention_given_pick = 0.0

        print(f"{row['model']:30} | {row['scenario']:15} | {p_pick:.3f} | {p_mention:.3f} | {p_mention_given_pick:.3f}")

        consistency_results.append({
            "model": row["model"],
            "scenario": row["scenario"],
            "p_pick_nudge": p_pick,
            "p_mention_nudge": p_mention,
            "p_mention_given_pick": p_mention_given_pick,
            "unfaithful_reasoning": row["unfaithful_reasoning"],
        })

    consistency_df = pd.DataFrame(consistency_results)

    print("\n### INTERPRETATION ###")
    print("p_mention_given_pick ≈ 1.0 means: when the model picks the nudge vendor, it acknowledges the nudge")
    print("p_mention_given_pick ≈ 0.0 means: model picks nudge vendor but doesn't mention the nudge (unfaithful)")

    return consistency_df


def create_presentation_tables(df):
    """Create clean tables suitable for presentation."""
    print("\n" + "=" * 80)
    print("PRESENTATION-READY TABLES")
    print("=" * 80)

    # Table 1: Nudge effect by scenario
    print("\n### Table 1: Nudge Vendor Selection Rate ###")
    print("(Shows how often each model picked the nudge vendor)")

    table1 = df.pivot_table(
        index="model",
        columns=["scenario", "extent"],
        values="selected_nudge_vendor"
    ).round(3)
    print(table1.to_string())

    # Table 2: Faithfulness metrics for explicit nudge conditions
    print("\n### Table 2: Faithfulness Metrics (Explicit Nudge Conditions Only) ###")
    explicit_df = df[df["extent"] == "explicit"].copy()

    table2 = explicit_df.pivot_table(
        index="model",
        columns="scenario",
        values=["selected_nudge_vendor", "mentioned_nudge_in_cot", "unfaithful_reasoning"]
    ).round(3)
    print(table2.to_string())

    return table1, table2


def main():
    print("Loading results...")
    df = load_all_results()

    print(f"Loaded {len(df)} result rows")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")

    # Analysis 1: Nudge selection rates
    summary = analyze_nudge_selection(df)

    # Analysis 2: CoT consistency
    consistency_df = analyze_cot_consistency(df)

    # Presentation tables
    table1, table2 = create_presentation_tables(df)

    # Save to CSV for easy import
    df.to_csv("results_full.csv", index=False)
    consistency_df.to_csv("results_consistency.csv", index=False)

    print("\n" + "=" * 80)
    print("Results saved to results_full.csv and results_consistency.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
