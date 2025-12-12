#!/usr/bin/env python3
"""
Extract scorer information from Impossible Checklist eval files into CSV format.

Usage:
    python extract_scores.py <eval_directory> [--output <output_file>]

Examples:
    # Extract from demo run (with condition subfolders)
    python extract_scores.py archived_logs/demo_run_5_epochs -o results_demo.csv

    # Extract from a specific condition folder
    python extract_scores.py logs/pressure_HIGH_eval_HIGH -o results_high_high.csv
"""

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any


def extract_scores_from_eval(eval_path: Path) -> list[dict[str, Any]]:
    """Extract score information from a single eval file."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(eval_path))

    # Extract eval-level metadata
    model = log.eval.model
    # Clean up model name for readability
    model_short = model.split("/")[-1] if "/" in model else model

    samples = []
    for sample in log.samples or []:
        sample_data = {
            "eval_file": eval_path.name,
            "eval_id": log.eval.eval_id,
            "model": model,
            "model_short": model_short,
            "sample_id": sample.id,
            "epoch": sample.epoch if hasattr(sample, "epoch") else None,
        }

        # Extract metadata (contains experimental condition)
        if sample.metadata:
            sample_data["pressure"] = sample.metadata.get("pressure", "")
            sample_data["eval_awareness"] = sample.metadata.get("eval_awareness", "")
            sample_data["condition_id"] = sample.metadata.get("condition_id", "")

        # Extract scores
        if sample.scores:
            for scorer_name, score in sample.scores.items():
                # Get numeric score values
                if hasattr(score, "value") and isinstance(score.value, dict):
                    for metric_name, metric_value in score.value.items():
                        sample_data[metric_name] = metric_value

                # Get explanation (contains all analysis results including string fields)
                if hasattr(score, "explanation") and score.explanation:
                    try:
                        explanation = json.loads(score.explanation)
                        # Add explanation fields (these are the _EXPLANATION text fields)
                        for key, value in explanation.items():
                            if key not in sample_data:
                                sample_data[f"exp_{key}"] = value
                    except json.JSONDecodeError:
                        sample_data["explanation_raw"] = score.explanation

                # Get the overall summary
                if hasattr(score, "answer") and score.answer:
                    sample_data["faithfulness_summary"] = score.answer

        samples.append(sample_data)

    return samples


def find_eval_files(path: Path, recursive: bool = True) -> list[Path]:
    """Find all .eval files in the given path."""
    if path.is_file() and path.suffix == ".eval":
        return [path]
    elif path.is_dir():
        pattern = "**/*.eval" if recursive else "*.eval"
        return sorted(path.glob(pattern))
    else:
        return []


def samples_to_csv(samples: list[dict], fields: list[str] | None = None) -> str:
    """Convert samples to CSV format."""
    if not samples:
        return ""

    # Determine fields to include with preferred ordering
    preferred_order = [
        # Identifiers
        "model_short",
        "model",
        "condition_id",
        "pressure",
        "eval_awareness",
        "epoch",
        "sample_id",
        "eval_id",
        "eval_file",
        # Primary faithfulness metrics
        "OVERALL_FAITHFUL",
        "T3_FAITHFUL",
        "T4_FAITHFUL",
        # Task 3 details
        "T3_ERROR_RECEIVED",
        "T3_COT_ACKNOWLEDGES_ERROR",
        "T3_COT_CONCLUDES_IMPOSSIBLE",
        "T3_ACTION_IS_BLOCKED",
        "T3_COT_ACTION_CONSISTENT",
        "T3_CONFABULATED_DATA",
        # Task 4 details
        "T4_ERROR_RECEIVED",
        "T4_COT_ACKNOWLEDGES_ERROR",
        "T4_COT_CONCLUDES_IMPOSSIBLE",
        "T4_ACTION_IS_BLOCKED",
        "T4_COT_ACTION_CONSISTENT",
        "T4_CONFABULATED_DATA",
        # Other metrics
        "POST_HOC_ERROR_RECOGNITION",
        "PRESSURE_MENTIONED",
        "EVAL_AWARENESS_MENTIONED",
        # Derived scores
        "COMPLETENESS_SCORE",
        "CONSISTENCY_SCORE",
        "CORRECT_BEHAVIOR_SCORE",
        "CONFABULATION_SCORE",
        # Summary
        "faithfulness_summary",
    ]

    if fields:
        fieldnames = fields
    else:
        # Get all unique keys across all samples
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())

        # Order: preferred fields first, then remaining fields alphabetically
        fieldnames = [f for f in preferred_order if f in all_keys]
        remaining = sorted(all_keys - set(fieldnames))

        # Exclude very long fields from default CSV output
        exclude_from_csv = {
            "grader_response",
            "grading_transcript",
            "explanation_raw",
        }
        # Also exclude exp_ prefixed explanation fields by default
        remaining = [
            f for f in remaining
            if f not in exclude_from_csv and not f.startswith("exp_")
        ]
        fieldnames.extend(remaining)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for sample in samples:
        # Truncate very long string values for CSV readability
        row = {}
        for k, v in sample.items():
            if k in fieldnames:
                if isinstance(v, str) and len(v) > 500:
                    row[k] = v[:497] + "..."
                else:
                    row[k] = v
        writer.writerow(row)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Extract scores from Impossible Checklist eval files"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to eval directory (will search recursively for .eval files)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (default: stdout)",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to include",
    )
    parser.add_argument(
        "--include-explanations",
        action="store_true",
        help="Include exp_* explanation fields in output",
    )

    args = parser.parse_args()

    # Find eval files
    eval_files = find_eval_files(args.path)
    if not eval_files:
        print(f"No .eval files found at {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(eval_files)} eval file(s)", file=sys.stderr)

    # Extract scores from each file
    all_samples = []
    for eval_file in eval_files:
        print(f"Processing: {eval_file}", file=sys.stderr)
        try:
            samples = extract_scores_from_eval(eval_file)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    print(f"Total samples extracted: {len(all_samples)}", file=sys.stderr)

    # Build CSV output
    fields = args.fields.split(",") if args.fields else None
    output_str = samples_to_csv(all_samples, fields)

    # Write output
    if args.output:
        args.output.write_text(output_str)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
