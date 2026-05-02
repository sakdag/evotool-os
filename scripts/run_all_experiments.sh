#!/usr/bin/env bash
# ============================================================
# Run all EVOTOOL experiments end-to-end.
# Execute this AFTER starting the vLLM server in a separate terminal:
#   bash scripts/start_vllm.sh
#
# Resumable: each step is skipped if its output JSON already exists.
# To force a re-run of a specific step, delete its output file, or
# pass --force to the individual script.
#
# Usage:
#   conda activate evotool
#   bash scripts/run_all_experiments.sh 2>&1 | tee experiments/run_all.log
# ============================================================

PYTHON="python"

# ---- Fast mode -------------------------------------------------------
# Set FAST=1 to use tiny splits and 2 generations for rapid iteration.
# Example: FAST=1 bash scripts/run_all_experiments.sh
# In fast mode results go to experiments/results_fast/ so they don't
# collide with full-run outputs.
# ----------------------------------------------------------------------
FAST="${FAST:-0}"
if [ "$FAST" = "1" ]; then
    RESULTS_DIR="experiments/results_fast"
    FAST_FLAG="--fast"
    MODE_LABEL="FAST (tiny splits, 2 gens)"
else
    RESULTS_DIR="experiments/results"
    FAST_FLAG=""
    MODE_LABEL="FULL"
fi

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Colours for step status
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

_run() {
    local label="$1"; shift
    local output="$1"; shift

    if [ -f "$output" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $label  (output exists: $output)"
        return 0
    fi

    echo -e "\n${GREEN}[RUN ]${NC} $label"
    if "$@"; then
        echo -e "${GREEN}[DONE]${NC} $label"
    else
        echo -e "${RED}[FAIL]${NC} $label  (exit code $?)"
        echo "       Output would have been: $output"
        echo "       Fix the error then re-run this script — completed steps will be skipped."
        # Continue to next step instead of aborting everything
    fi
}

echo "================================================================"
echo "EVOTOOL Experiment Suite  [$MODE_LABEL]"
echo "================================================================"
echo "Completed steps are automatically skipped."
echo ""

# ----------------------------------------------------------------
# 1. Static baseline (INITIAL_POLICY, no evolution)
# ----------------------------------------------------------------
_run "Static baseline: τ-Bench Retail" \
    "$RESULTS_DIR/static_tau_bench_retail.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark tau_bench_retail \
        --use-initial \
        --output "$RESULTS_DIR/static_tau_bench_retail.json" \
        --max-steps 30 $FAST_FLAG

_run "Static baseline: τ-Bench Airline" \
    "$RESULTS_DIR/static_tau_bench_airline.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark tau_bench_airline \
        --use-initial \
        --output "$RESULTS_DIR/static_tau_bench_airline.json" \
        --max-steps 30 $FAST_FLAG

_run "Static baseline: BFCL Single" \
    "$RESULTS_DIR/static_bfcl_single.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark bfcl_single \
        --use-initial \
        --output "$RESULTS_DIR/static_bfcl_single.json" \
        --max-steps 15 $FAST_FLAG

_run "Static baseline: BFCL Multi" \
    "$RESULTS_DIR/static_bfcl_multi.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark bfcl_multi \
        --use-initial \
        --output "$RESULTS_DIR/static_bfcl_multi.json" \
        --max-steps 15 $FAST_FLAG

# ----------------------------------------------------------------
# 2. ReAct baseline
# ----------------------------------------------------------------
_run "ReAct baseline: τ-Bench Retail" \
    "$RESULTS_DIR/baseline_react_tau_bench_retail.json" \
    $PYTHON scripts/run_baseline.py \
        --method react \
        --benchmark tau_bench_retail \
        --output "$RESULTS_DIR/baseline_react_tau_bench_retail.json" \
        $FAST_FLAG

_run "ReAct baseline: τ-Bench Airline" \
    "$RESULTS_DIR/baseline_react_tau_bench_airline.json" \
    $PYTHON scripts/run_baseline.py \
        --method react \
        --benchmark tau_bench_airline \
        --output "$RESULTS_DIR/baseline_react_tau_bench_airline.json" \
        $FAST_FLAG

_run "ReAct baseline: BFCL Single" \
    "$RESULTS_DIR/baseline_react_bfcl_single.json" \
    $PYTHON scripts/run_baseline.py \
        --method react \
        --benchmark bfcl_single \
        --output "$RESULTS_DIR/baseline_react_bfcl_single.json" \
        $FAST_FLAG

_run "ReAct baseline: BFCL Multi" \
    "$RESULTS_DIR/baseline_react_bfcl_multi.json" \
    $PYTHON scripts/run_baseline.py \
        --method react \
        --benchmark bfcl_multi \
        --output "$RESULTS_DIR/baseline_react_bfcl_multi.json" \
        $FAST_FLAG

# ----------------------------------------------------------------
# 3. EVOTOOL full  (fast mode: G=2, B=2, tiny splits)
#    Checkpoints saved to *.ckpt.json — run resumes automatically.
# ----------------------------------------------------------------
_run "EVOTOOL: τ-Bench Retail" \
    "$RESULTS_DIR/evotool_tau_bench_retail_full.json" \
    $PYTHON scripts/run_evotool.py \
        --benchmark tau_bench_retail \
        --generations 8 \
        --batch-size 3 \
        --ablation full \
        --output "$RESULTS_DIR/evotool_tau_bench_retail_full.json" \
        $FAST_FLAG

_run "EVOTOOL eval: τ-Bench Retail" \
    "$RESULTS_DIR/eval_tau_bench_retail_full.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark tau_bench_retail \
        --policy "$RESULTS_DIR/evotool_tau_bench_retail_full.json" \
        --output "$RESULTS_DIR/eval_tau_bench_retail_full.json" \
        $FAST_FLAG

_run "EVOTOOL: τ-Bench Airline" \
    "$RESULTS_DIR/evotool_tau_bench_airline_full.json" \
    $PYTHON scripts/run_evotool.py \
        --benchmark tau_bench_airline \
        --generations 8 \
        --batch-size 3 \
        --ablation full \
        --output "$RESULTS_DIR/evotool_tau_bench_airline_full.json" \
        $FAST_FLAG

_run "EVOTOOL eval: τ-Bench Airline" \
    "$RESULTS_DIR/eval_tau_bench_airline_full.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark tau_bench_airline \
        --policy "$RESULTS_DIR/evotool_tau_bench_airline_full.json" \
        --output "$RESULTS_DIR/eval_tau_bench_airline_full.json" \
        $FAST_FLAG

_run "EVOTOOL: BFCL Single" \
    "$RESULTS_DIR/evotool_bfcl_single_full.json" \
    $PYTHON scripts/run_evotool.py \
        --benchmark bfcl_single \
        --generations 8 \
        --batch-size 3 \
        --ablation full \
        --output "$RESULTS_DIR/evotool_bfcl_single_full.json" \
        $FAST_FLAG

_run "EVOTOOL eval: BFCL Single" \
    "$RESULTS_DIR/eval_bfcl_single_full.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark bfcl_single \
        --policy "$RESULTS_DIR/evotool_bfcl_single_full.json" \
        --output "$RESULTS_DIR/eval_bfcl_single_full.json" \
        $FAST_FLAG

_run "EVOTOOL: BFCL Multi" \
    "$RESULTS_DIR/evotool_bfcl_multi_full.json" \
    $PYTHON scripts/run_evotool.py \
        --benchmark bfcl_multi \
        --generations 8 \
        --batch-size 3 \
        --ablation full \
        --output "$RESULTS_DIR/evotool_bfcl_multi_full.json" \
        $FAST_FLAG

_run "EVOTOOL eval: BFCL Multi" \
    "$RESULTS_DIR/eval_bfcl_multi_full.json" \
    $PYTHON scripts/run_eval.py \
        --benchmark bfcl_multi \
        --policy "$RESULTS_DIR/evotool_bfcl_multi_full.json" \
        --output "$RESULTS_DIR/eval_bfcl_multi_full.json" \
        $FAST_FLAG

# ----------------------------------------------------------------
# 4. Ablations (Table 2)
# ----------------------------------------------------------------
for ABLATION in random plan_only sel_only call_only syn_only monolithic; do
    _run "Ablation [$ABLATION]: τ-Bench Retail" \
        "$RESULTS_DIR/ablation_${ABLATION}_tau_bench_retail.json" \
        $PYTHON scripts/run_evotool.py \
            --benchmark tau_bench_retail \
            --ablation "$ABLATION" \
            --output "$RESULTS_DIR/ablation_${ABLATION}_tau_bench_retail.json" \
            $FAST_FLAG

    _run "Ablation [$ABLATION] eval: τ-Bench Retail" \
        "$RESULTS_DIR/eval_ablation_${ABLATION}_tau_bench_retail.json" \
        $PYTHON scripts/run_eval.py \
            --benchmark tau_bench_retail \
            --policy "$RESULTS_DIR/ablation_${ABLATION}_tau_bench_retail.json" \
            --output "$RESULTS_DIR/eval_ablation_${ABLATION}_tau_bench_retail.json" \
            $FAST_FLAG

    _run "Ablation [$ABLATION]: BFCL Single" \
        "$RESULTS_DIR/ablation_${ABLATION}_bfcl_single.json" \
        $PYTHON scripts/run_evotool.py \
            --benchmark bfcl_single \
            --ablation "$ABLATION" \
            --output "$RESULTS_DIR/ablation_${ABLATION}_bfcl_single.json" \
            $FAST_FLAG

    _run "Ablation [$ABLATION] eval: BFCL Single" \
        "$RESULTS_DIR/eval_ablation_${ABLATION}_bfcl_single.json" \
        $PYTHON scripts/run_eval.py \
            --benchmark bfcl_single \
            --policy "$RESULTS_DIR/ablation_${ABLATION}_bfcl_single.json" \
            --output "$RESULTS_DIR/eval_ablation_${ABLATION}_bfcl_single.json" \
            $FAST_FLAG
done

# ----------------------------------------------------------------
# 5. Analysis
# ----------------------------------------------------------------
echo ""
echo "--- Analysis ---"
$PYTHON scripts/analyze_results.py \
    --results-dir "$RESULTS_DIR" \
    --logs-dir "$LOG_DIR"

echo ""
echo "================================================================"
echo "All experiments complete. Results in $RESULTS_DIR/"
echo "================================================================"
