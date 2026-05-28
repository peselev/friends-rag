#!/usr/bin/env bash
# Run both evals + the comparison, log everything, save the output.
# Usage: ./run_eval_overnight.sh

set -e  # Exit on first error - no point continuing if eval 1 crashes

# Activate venv
source .venv/bin/activate

# Log file with timestamp
LOG="eval_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
echo "Started at $(date)" | tee "$LOG"

# Run all three, append both stdout and stderr to the log
{
    echo ""
    echo "=== Step 1: Emory eval ==="
    python -m src.eval.runner

    echo ""
    echo "=== Step 2: Lexical paraphrases eval ==="
    python -m src.eval.runner \
        --questions data/processed/eval_lexical_paraphrases.jsonl \
        --results data/processed/eval_results_lexical.jsonl

    echo ""
    echo "=== Step 3: Compare results ==="
    python -m scripts.compare_eval_sets

    echo ""
    echo "Finished at $(date)"
} 2>&1 | tee -a "$LOG"

echo ""
echo "Done. Full output saved to $LOG"