#!/usr/bin/env bash
# Run the full 2x3 grid for one or more conversations sequentially.
# Designed for overnight unattended use: logs everything to logs/,
# never aborts on a per-cell failure, copies Hindsight result files
# under stable per-cell names before subsequent runs overwrite them.
#
# Usage:
#   bash run_conv_full_grid.sh conv-30 conv-49
#
# Requirements before running:
#   - HF_TOKEN exported in shell
#   - ~/.local/bin on PATH (for uv)
#   - hindsight/.env has the literal HF token populated

set -u  # error on unset vars, but NOT -e: we want cell failures to be visible without aborting the run

BENCH_DIR="/Users/anna/Dropbox/phd-stuff/CODE/2026/agent-memory-benchmarks/2026-04-27-locomo"
HINDSIGHT_DIR="/Users/anna/Dropbox/phd-stuff/CODE/2026/hindsight"
HINDSIGHT_RESULTS="$HINDSIGHT_DIR/hindsight-dev/benchmarks/locomo/results"
LOG_DIR="$BENCH_DIR/logs/grid_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cd "$BENCH_DIR"

echo "=== full-grid run started $(date) ===" | tee "$LOG_DIR/_meta.log"
echo "samples: $@" | tee -a "$LOG_DIR/_meta.log"
echo "log dir: $LOG_DIR" | tee -a "$LOG_DIR/_meta.log"

# Helper: judge the most recent JSONL written
judge_latest () {
    local label="$1"
    local latest
    latest=$(ls -t "$BENCH_DIR/runs"/baseline_qa_*.jsonl | head -1)
    echo ">>> judging $latest as $label" | tee -a "$LOG_DIR/_meta.log"
    python3 "$BENCH_DIR/summarize_run.py" "$latest" --judge \
        2>&1 | tee "$LOG_DIR/${label}_judge.log"
}

for SAMPLE in "$@"; do
    echo "" | tee -a "$LOG_DIR/_meta.log"
    echo "============================================================" | tee -a "$LOG_DIR/_meta.log"
    echo "SAMPLE: $SAMPLE   start $(date)" | tee -a "$LOG_DIR/_meta.log"
    echo "============================================================" | tee -a "$LOG_DIR/_meta.log"

    # --- Cell E: graph backend, no MCQ. Also builds + caches the graph. ---
    echo ">>> cell E (graph, no MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    python3 baseline_qa.py --backend graph --sample-id "$SAMPLE" --filter cat5 \
        2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_E.log"
    judge_latest "${SAMPLE}_cell_E"

    # --- Cell F: graph backend, with MCQ. Loads cached graph. ---
    echo ">>> cell F (graph, MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    python3 baseline_qa.py --backend graph --sample-id "$SAMPLE" --filter cat5 --cat5-mcq \
        2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_F.log"
    judge_latest "${SAMPLE}_cell_F"

    # --- Cell A: full-context, no MCQ ---
    echo ">>> cell A (full-context, no MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    python3 baseline_qa.py --sample-id "$SAMPLE" --filter cat5 \
        2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_A.log"
    judge_latest "${SAMPLE}_cell_A"

    # --- Cell B: full-context, with MCQ ---
    echo ">>> cell B (full-context, MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    python3 baseline_qa.py --sample-id "$SAMPLE" --filter cat5 --cat5-mcq \
        2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_B.log"
    judge_latest "${SAMPLE}_cell_B"

    # --- Cell C: Hindsight, no MCQ (first hindsight call for this sample
    #     does ingestion; we use HINDSIGHT_BENCHMARK_SAMPLE_IDS to target
    #     this conversation). ---
    echo ">>> cell C (Hindsight, no MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    (
        cd "$HINDSIGHT_DIR"
        HINDSIGHT_BENCHMARK_ONLY_CAT5=1 \
        HINDSIGHT_BENCHMARK_SAMPLE_IDS="$SAMPLE" \
            bash scripts/benchmarks/run-locomo.sh --max-conversations 1
    ) 2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_C.log"
    cp "$HINDSIGHT_RESULTS/benchmark_results.json" \
       "$HINDSIGHT_RESULTS/${SAMPLE}_cat5_no_mcq.json" 2>/dev/null \
       && echo ">>> saved Hindsight cell C result: ${SAMPLE}_cat5_no_mcq.json" | tee -a "$LOG_DIR/_meta.log"

    # --- Cell D: Hindsight, with MCQ (reuses ingested memory bank via --skip-ingestion) ---
    echo ">>> cell D (Hindsight, MCQ) for $SAMPLE — start $(date)" | tee -a "$LOG_DIR/_meta.log"
    (
        cd "$HINDSIGHT_DIR"
        HINDSIGHT_BENCHMARK_ONLY_CAT5=1 \
        HINDSIGHT_BENCHMARK_CAT5_MCQ=1 \
        HINDSIGHT_BENCHMARK_SAMPLE_IDS="$SAMPLE" \
            bash scripts/benchmarks/run-locomo.sh --max-conversations 1 --skip-ingestion
    ) 2>&1 | tee "$LOG_DIR/${SAMPLE}_cell_D.log"
    cp "$HINDSIGHT_RESULTS/benchmark_results.json" \
       "$HINDSIGHT_RESULTS/${SAMPLE}_cat5_with_mcq.json" 2>/dev/null \
       && echo ">>> saved Hindsight cell D result: ${SAMPLE}_cat5_with_mcq.json" | tee -a "$LOG_DIR/_meta.log"

    echo "============================================================" | tee -a "$LOG_DIR/_meta.log"
    echo "SAMPLE $SAMPLE COMPLETE — finish $(date)" | tee -a "$LOG_DIR/_meta.log"
    echo "============================================================" | tee -a "$LOG_DIR/_meta.log"
done

echo "" | tee -a "$LOG_DIR/_meta.log"
echo "=== full-grid run finished $(date) ===" | tee -a "$LOG_DIR/_meta.log"
echo "logs at $LOG_DIR" | tee -a "$LOG_DIR/_meta.log"
