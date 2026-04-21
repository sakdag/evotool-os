#!/usr/bin/env bash
# ============================================================
# Run all EVOTOOL experiments end-to-end.
# Execute this AFTER starting the vLLM server in a separate terminal:
#   bash scripts/start_vllm.sh
#
# Usage:
#   conda activate evotool
#   bash scripts/run_all_experiments.sh 2>&1 | tee experiments/run_all.log
# ============================================================
set -e

PYTHON="python"
LOG_DIR="experiments/logs"
RESULTS_DIR="experiments/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "================================================================"
echo "EVOTOOL Full Experiment Suite"
echo "================================================================"
echo ""

# ----------------------------------------------------------------
# 1. SANITY CHECK: Static baseline (INITIAL_POLICY, no evolution)
#    Expected (Table 2 Static row, Qwen3-8B):
#      τ-Bench Avg ≈ 21.0,  BFCL Avg ≈ 52.0
# ----------------------------------------------------------------
echo "--- [1/6] Static baseline: τ-Bench Retail ---"
$PYTHON scripts/run_eval.py \
  --benchmark tau_bench_retail \
  --use-initial \
  --output "$RESULTS_DIR/static_tau_bench_retail.json" \
  --max-steps 30

echo "--- [2/6] Static baseline: τ-Bench Airline ---"
$PYTHON scripts/run_eval.py \
  --benchmark tau_bench_airline \
  --use-initial \
  --output "$RESULTS_DIR/static_tau_bench_airline.json" \
  --max-steps 30

echo "--- [3/6] Static baseline: BFCL Single ---"
$PYTHON scripts/run_eval.py \
  --benchmark bfcl_single \
  --use-initial \
  --output "$RESULTS_DIR/static_bfcl_single.json" \
  --max-steps 15

echo "--- [4/6] Static baseline: BFCL Multi ---"
$PYTHON scripts/run_eval.py \
  --benchmark bfcl_multi \
  --use-initial \
  --output "$RESULTS_DIR/static_bfcl_multi.json" \
  --max-steps 15

# ----------------------------------------------------------------
# 2. ReAct baseline
#    Expected (Table 1, Qwen3-8B):
#      τ-Bench Retail=33.1, Airline=14.4
#      BFCL Single=70.6, Multi=33.3
# ----------------------------------------------------------------
echo "--- ReAct baseline: τ-Bench Retail ---"
$PYTHON scripts/run_baseline.py \
  --method react \
  --benchmark tau_bench_retail \
  --output "$RESULTS_DIR/baseline_react_tau_bench_retail.json"

echo "--- ReAct baseline: τ-Bench Airline ---"
$PYTHON scripts/run_baseline.py \
  --method react \
  --benchmark tau_bench_airline \
  --output "$RESULTS_DIR/baseline_react_tau_bench_airline.json"

echo "--- ReAct baseline: BFCL Single ---"
$PYTHON scripts/run_baseline.py \
  --method react \
  --benchmark bfcl_single \
  --output "$RESULTS_DIR/baseline_react_bfcl_single.json"

echo "--- ReAct baseline: BFCL Multi ---"
$PYTHON scripts/run_baseline.py \
  --method react \
  --benchmark bfcl_multi \
  --output "$RESULTS_DIR/baseline_react_bfcl_multi.json"

# ----------------------------------------------------------------
# 3. EVOTOOL full (G=8, B=3)
#    Expected (Table 1, Qwen3-8B):
#      τ-Bench Retail=35.9, Airline=15.7
#      BFCL Single=76.9, Multi=36.4
# ----------------------------------------------------------------
echo "--- EVOTOOL: τ-Bench Retail ---"
$PYTHON scripts/run_evotool.py \
  --benchmark tau_bench_retail \
  --generations 8 \
  --batch-size 3 \
  --ablation full \
  --output "$RESULTS_DIR/evotool_tau_bench_retail_full.json"

echo "--- EVOTOOL eval: τ-Bench Retail ---"
$PYTHON scripts/run_eval.py \
  --benchmark tau_bench_retail \
  --policy "$RESULTS_DIR/evotool_tau_bench_retail_full.json" \
  --output "$RESULTS_DIR/eval_tau_bench_retail_full.json"

echo "--- EVOTOOL: τ-Bench Airline ---"
$PYTHON scripts/run_evotool.py \
  --benchmark tau_bench_airline \
  --generations 8 \
  --batch-size 3 \
  --ablation full \
  --output "$RESULTS_DIR/evotool_tau_bench_airline_full.json"

echo "--- EVOTOOL eval: τ-Bench Airline ---"
$PYTHON scripts/run_eval.py \
  --benchmark tau_bench_airline \
  --policy "$RESULTS_DIR/evotool_tau_bench_airline_full.json" \
  --output "$RESULTS_DIR/eval_tau_bench_airline_full.json"

echo "--- EVOTOOL: BFCL Single ---"
$PYTHON scripts/run_evotool.py \
  --benchmark bfcl_single \
  --generations 8 \
  --batch-size 3 \
  --ablation full \
  --output "$RESULTS_DIR/evotool_bfcl_single_full.json"

echo "--- EVOTOOL eval: BFCL Single ---"
$PYTHON scripts/run_eval.py \
  --benchmark bfcl_single \
  --policy "$RESULTS_DIR/evotool_bfcl_single_full.json" \
  --output "$RESULTS_DIR/eval_bfcl_single_full.json"

echo "--- EVOTOOL: BFCL Multi ---"
$PYTHON scripts/run_evotool.py \
  --benchmark bfcl_multi \
  --generations 8 \
  --batch-size 3 \
  --ablation full \
  --output "$RESULTS_DIR/evotool_bfcl_multi_full.json"

echo "--- EVOTOOL eval: BFCL Multi ---"
$PYTHON scripts/run_eval.py \
  --benchmark bfcl_multi \
  --policy "$RESULTS_DIR/evotool_bfcl_multi_full.json" \
  --output "$RESULTS_DIR/eval_bfcl_multi_full.json"

# ----------------------------------------------------------------
# 4. Ablations (Table 2)
# ----------------------------------------------------------------
for ABLATION in random plan_only sel_only call_only syn_only monolithic; do
  echo "--- Ablation [$ABLATION]: τ-Bench Retail ---"
  $PYTHON scripts/run_evotool.py \
    --benchmark tau_bench_retail \
    --ablation "$ABLATION" \
    --output "$RESULTS_DIR/ablation_${ABLATION}_tau_bench_retail.json"
  $PYTHON scripts/run_eval.py \
    --benchmark tau_bench_retail \
    --policy "$RESULTS_DIR/ablation_${ABLATION}_tau_bench_retail.json" \
    --output "$RESULTS_DIR/eval_ablation_${ABLATION}_tau_bench_retail.json"

  echo "--- Ablation [$ABLATION]: BFCL Single ---"
  $PYTHON scripts/run_evotool.py \
    --benchmark bfcl_single \
    --ablation "$ABLATION" \
    --output "$RESULTS_DIR/ablation_${ABLATION}_bfcl_single.json"
  $PYTHON scripts/run_eval.py \
    --benchmark bfcl_single \
    --policy "$RESULTS_DIR/ablation_${ABLATION}_bfcl_single.json" \
    --output "$RESULTS_DIR/eval_ablation_${ABLATION}_bfcl_single.json"
done

# ----------------------------------------------------------------
# 5. Analysis: learning curves + results table
# ----------------------------------------------------------------
echo "--- Analysis ---"
$PYTHON scripts/analyze_results.py \
  --results-dir "$RESULTS_DIR" \
  --logs-dir "$LOG_DIR"

echo ""
echo "================================================================"
echo "All experiments complete. Results in $RESULTS_DIR/"
echo "================================================================"
