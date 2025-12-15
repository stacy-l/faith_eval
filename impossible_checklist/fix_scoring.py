from inspect_ai.log import read_eval_log, write_eval_log, edit_score
from inspect_ai.scorer import ScoreEdit, ProvenanceData

# Read the log file
log = read_eval_log("reports/demo_run_5_epochs/logs/pressure_HIGH_eval_LOW/2025-12-09T19-30-55-08-00_impossible-checklist_guFgzcvd8r7Bzgd2aXj4FA.eval")

# Create a score edit with provenance tracking
edit = ScoreEdit(
    value=1,  # New score value
    explanation="Corrected model grader bug",  # Optional new explanation
    provenance=ProvenanceData(
        author="stacy",
        reason="Scorer didn't follow the scoring rules and abide by the Ground Truth",
    )
)

# Edit the score (automatically recomputes metrics)
print(log.samples[0].scores)
edit_score(
    log=log,
    sample_id=log.samples[0].id,
    epoch=3,
    score_name="T3_CONFABULATED_DATA",
    edit=edit
)

# # Write back to the log file
# write_eval_log(log)