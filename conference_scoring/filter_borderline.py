"""
Filter Borderline Abstracts from Screening Results

Reads Inspect eval logs, computes per-abstract accept rates,
and filters to borderline abstracts (40-60% accept rate).
"""

import json
from pathlib import Path
from collections import defaultdict

from inspect_ai.log import read_eval_log


def analyze_screening_log(log_path: str) -> dict[int, dict]:
    """Analyze screening log and compute per-abstract accept rates.

    Returns dict mapping abstract_id to stats:
    {
        "accept_rate": float,
        "n_accept": int,
        "n_reject": int,
        "n_total": int,
        "title": str,
    }
    """
    log = read_eval_log(log_path)

    # Group results by abstract_id
    results = defaultdict(lambda: {"accepts": 0, "rejects": 0, "title": ""})

    for sample in log.samples:
        abstract_id = sample.metadata.get("abstract_id")
        title = sample.metadata.get("title", "")

        if abstract_id is None:
            continue

        results[abstract_id]["title"] = title

        # Get the score
        if sample.scores:
            for scorer_name, score in sample.scores.items():
                if score.answer == "ACCEPT":
                    results[abstract_id]["accepts"] += 1
                elif score.answer == "REJECT":
                    results[abstract_id]["rejects"] += 1

    # Compute stats
    stats = {}
    for abstract_id, data in results.items():
        n_accept = data["accepts"]
        n_reject = data["rejects"]
        n_total = n_accept + n_reject

        stats[abstract_id] = {
            "accept_rate": n_accept / n_total if n_total > 0 else None,
            "n_accept": n_accept,
            "n_reject": n_reject,
            "n_total": n_total,
            "title": data["title"],
        }

    return stats


def filter_borderline(
    stats: dict[int, dict],
    lower: float = 0.4,
    upper: float = 0.6,
    min_samples: int = 10,
) -> list[int]:
    """Filter to abstract IDs with accept rate in [lower, upper]."""
    borderline = []

    for abstract_id, s in stats.items():
        if s["accept_rate"] is None:
            continue
        if s["n_total"] < min_samples:
            continue
        if lower <= s["accept_rate"] <= upper:
            borderline.append(abstract_id)

    return sorted(borderline)


def create_borderline_dataset(
    abstracts_path: str,
    borderline_ids: list[int],
    output_path: str,
) -> list[dict]:
    """Create filtered dataset with only borderline abstracts."""
    with open(abstracts_path) as f:
        all_abstracts = json.load(f)

    borderline = [a for a in all_abstracts if a["id"] in borderline_ids]

    # Renumber IDs sequentially
    for i, abstract in enumerate(borderline, 1):
        abstract["original_id"] = abstract["id"]
        abstract["id"] = i

    with open(output_path, "w") as f:
        json.dump(borderline, f, indent=2)

    return borderline


def print_summary(stats: dict[int, dict], borderline_ids: list[int]):
    """Print analysis summary."""
    print("\n" + "=" * 70)
    print("SCREENING ANALYSIS")
    print("=" * 70)

    # Sort by accept rate
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["accept_rate"] or 0)

    print(f"\n{'ID':>4} {'Accept Rate':>12} {'N':>5} {'Status':>12}  Title")
    print("-" * 70)

    for abstract_id, s in sorted_stats:
        rate = s["accept_rate"]
        rate_str = f"{rate:.1%}" if rate is not None else "N/A"
        status = "BORDERLINE" if abstract_id in borderline_ids else ""
        title = s["title"][:35] + "..." if len(s["title"]) > 35 else s["title"]
        print(f"{abstract_id:>4} {rate_str:>12} {s['n_total']:>5} {status:>12}  {title}")

    # Distribution
    print("\n" + "-" * 70)
    print("DISTRIBUTION:")
    rates = [s["accept_rate"] for s in stats.values() if s["accept_rate"] is not None]

    bins = [(0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60% (borderline)"),
            (0.6, 0.8, "60-80%"), (0.8, 1.01, "80-100%")]

    for lo, hi, label in bins:
        count = sum(1 for r in rates if lo <= r < hi)
        bar = "█" * count
        print(f"  {label:>20}: {count:>3} {bar}")

    print(f"\nTotal abstracts: {len(stats)}")
    print(f"Borderline (40-60%): {len(borderline_ids)}")

    if rates:
        print(f"Overall accept rate: {sum(rates)/len(rates):.1%}")


def main(
    log_path: str,
    abstracts_path: str,
    output_path: str | None = None,
    lower: float = 0.4,
    upper: float = 0.6,
):
    """Main analysis pipeline."""
    # Analyze log
    stats = analyze_screening_log(log_path)

    # Filter borderline
    borderline_ids = filter_borderline(stats, lower=lower, upper=upper)

    # Print summary
    print_summary(stats, borderline_ids)

    # Create filtered dataset
    if output_path:
        borderline = create_borderline_dataset(abstracts_path, borderline_ids, output_path)
        print(f"\nSaved {len(borderline)} borderline abstracts to: {output_path}")

    return stats, borderline_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter borderline abstracts from screening")
    parser.add_argument("log_path", help="Path to Inspect eval log (.eval file)")
    parser.add_argument("abstracts_path", help="Path to source abstracts JSON")
    parser.add_argument("-o", "--output", help="Output path for borderline abstracts JSON")
    parser.add_argument("--lower", type=float, default=0.4, help="Lower bound (default: 0.4)")
    parser.add_argument("--upper", type=float, default=0.6, help="Upper bound (default: 0.6)")

    args = parser.parse_args()

    main(
        log_path=args.log_path,
        abstracts_path=args.abstracts_path,
        output_path=args.output,
        lower=args.lower,
        upper=args.upper,
    )
