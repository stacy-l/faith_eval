#!/usr/bin/env python3
"""
Extract vendor selection answers from .eval files and compute baseline frequencies.

Uses Inspect AI's log reading API to parse the .eval files without manual unzipping.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

from inspect_ai.log import read_eval_log

LOG_DIR = Path("logs")

# Only analyze baseline conditions (extent = "none")
BASELINE_CONDITIONS = [
    ("sycophancy_bias", "none"),
    ("metadata_eco", "none"),
    ("metadata_anti_eco", "none"),
]

ALL_VENDORS = ["EcoRun", "MassCompute", "SuperCloud"]


def normalize_vendor(answer):
    """Normalize vendor name to canonical form, or return 'Other' if not recognized."""
    if answer is None:
        return "Other"

    answer_lower = answer.lower().strip()

    if "ecorun" in answer_lower:
        return "EcoRun"
    elif "masscompute" in answer_lower:
        return "MassCompute"
    elif "supercloud" in answer_lower:
        return "SuperCloud"
    else:
        return "Other"


def extract_answers_from_eval_files():
    """Read all .eval files and extract the submitted vendor answers."""
    results = []

    for scenario, extent in BASELINE_CONDITIONS:
        condition_dir = LOG_DIR / f"{scenario}_{extent}"

        if not condition_dir.exists():
            print(f"Warning: {condition_dir} not found")
            continue

        eval_files = list(condition_dir.glob("*.eval"))
        print(f"Processing {len(eval_files)} eval files for {scenario}_{extent}...")

        for eval_file in eval_files:
            try:
                log = read_eval_log(str(eval_file))
            except Exception as e:
                print(f"  Error reading {eval_file.name}: {e}")
                continue

            model = log.eval.model
            model_short = model.split("/")[-1] if "/" in model else model

            # Iterate over all samples (epochs)
            for sample in log.samples:
                # The answer is stored in the score
                if sample.scores:
                    # Get the first scorer's result
                    score_key = list(sample.scores.keys())[0]
                    score = sample.scores[score_key]
                    answer = score.answer if hasattr(score, 'answer') else None
                else:
                    answer = None

                # Normalize to canonical vendor names
                normalized = normalize_vendor(answer)

                results.append({
                    "scenario": scenario,
                    "extent": extent,
                    "model": model_short,
                    "model_full": model,
                    "sample_id": sample.id,
                    "epoch": sample.epoch,
                    "answer_raw": answer,
                    "answer": normalized,
                })

    return pd.DataFrame(results)


def compute_vendor_frequencies(df):
    """Compute vendor selection frequencies by model."""
    print("\n" + "=" * 80)
    print("BASELINE VENDOR SELECTION FREQUENCIES (extent = 'none')")
    print("=" * 80)

    # Overall frequencies across all models and scenarios
    print("\n### Overall Frequencies (All Models, All Baseline Scenarios) ###")
    overall = df["answer"].value_counts(normalize=True).round(3)
    print(overall.to_string())

    # By model (aggregated across all baseline scenarios)
    print("\n### By Model (Aggregated Across All Baseline Scenarios) ###")
    model_vendor_counts = df.groupby(["model", "answer"]).size().unstack(fill_value=0)
    # Reorder columns to have vendors first, then Other
    cols = [c for c in ALL_VENDORS if c in model_vendor_counts.columns]
    if "Other" in model_vendor_counts.columns:
        cols.append("Other")
    model_vendor_counts = model_vendor_counts[cols]
    model_vendor_freq = model_vendor_counts.div(model_vendor_counts.sum(axis=1), axis=0).round(3)
    print(model_vendor_freq.to_string())

    # By scenario (aggregated across all models)
    print("\n### By Scenario (Aggregated Across All Models) ###")
    scenario_vendor_counts = df.groupby(["scenario", "answer"]).size().unstack(fill_value=0)
    cols = [c for c in ALL_VENDORS if c in scenario_vendor_counts.columns]
    if "Other" in scenario_vendor_counts.columns:
        cols.append("Other")
    scenario_vendor_counts = scenario_vendor_counts[cols]
    scenario_vendor_freq = scenario_vendor_counts.div(scenario_vendor_counts.sum(axis=1), axis=0).round(3)
    print(scenario_vendor_freq.to_string())

    # Detailed: by model and scenario
    print("\n### Detailed: By Model and Scenario ###")
    for scenario in df["scenario"].unique():
        print(f"\n--- {scenario} ---")
        scenario_df = df[df["scenario"] == scenario]
        detailed = scenario_df.groupby(["model", "answer"]).size().unstack(fill_value=0)
        cols = [c for c in ALL_VENDORS if c in detailed.columns]
        if "Other" in detailed.columns:
            cols.append("Other")
        detailed = detailed[cols]
        detailed_freq = detailed.div(detailed.sum(axis=1), axis=0).round(3)
        print(detailed_freq.to_string())

    return model_vendor_freq, scenario_vendor_freq


def main():
    print("Extracting answers from .eval files...")
    df = extract_answers_from_eval_files()

    if df.empty:
        print("No data extracted!")
        return

    print(f"\nExtracted {len(df)} samples")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Scenarios: {df['scenario'].unique().tolist()}")
    print(f"Normalized answers: {df['answer'].unique().tolist()}")

    # Show "Other" examples
    other_df = df[df["answer"] == "Other"]
    if not other_df.empty:
        print(f"\n{len(other_df)} samples classified as 'Other':")
        for _, row in other_df.head(5).iterrows():
            print(f"  - {row['model']}: {row['answer_raw'][:50] if row['answer_raw'] else 'None'}...")

    # Compute frequencies
    model_freq, scenario_freq = compute_vendor_frequencies(df)

    # Save raw data
    df.to_csv("baseline_vendor_selections.csv", index=False)
    print("\n" + "=" * 80)
    print("Raw data saved to baseline_vendor_selections.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
