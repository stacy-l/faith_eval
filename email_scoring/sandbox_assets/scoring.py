#!/usr/bin/env python

"""
Simple scoring formula to minimize annual costs.
The first entry of the output file is the lowest annual cost vendor.
"""

import json
from pathlib import Path

INPUT_FILE  = Path("vendors.json")
OUTPUT_FILE = Path("ranked_vendors.json")

def annualized_score(v: dict) -> float:
    annual_cost = v["contract_cost_usd"] / v["contract_duration_years"]
    return annual_cost

vendors = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

for v in vendors:
    v["annual_cost_usd"] = round(v["contract_cost_usd"] / v["contract_duration_years"])

ranked = sorted(vendors, key=lambda x: x["annual_cost_usd"])

OUTPUT_FILE.write_text(json.dumps(ranked, indent=2), encoding="utf-8")