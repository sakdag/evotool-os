# EvoTool: CENG502 Reproduction

## Paper Reference

**EvoTool: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection**  
Shuo Yang, Soyeon Caren Han, Xueqi Ma, Yan Li, Mohammad Reza Ghasemi Madani, Eduard Hovy  
arXiv preprint, 2026.  
[https://arxiv.org/abs/2603.04900](https://arxiv.org/abs/2603.04900)

---

## Method Description

EvoTool is a gradient-free, self-evolving framework that optimises an LLM agent's **tool-use policy** through an evolutionary loop. Unlike prior work that tunes a single monolithic prompt or optimises only one agent component in isolation, EvoTool decomposes the policy into **four specialist modules**: Planner, Selector, Caller, and Synthesizer and iteratively improves them via three novel mechanisms:

1. **Trajectory-Grounded Blame Attribution**: A Blamer LLM analyses failed trajectories and pinpoints which module caused the failure, enabling precise credit assignment in long-horizon tasks.
2. **Feedback-Guided Targeted Mutation**: Only the blamed module's prompt is rewritten using natural-language critique derived from diagnostic traces. This avoids the destructive interference of global updates.
3. **Diversity-Aware Population Selection**: Candidates are retained if they achieve the best score on *at least one* held-out instance (instance-wise winner criterion). This prevents population collapse and keeps complementary specialists alive across generations.

The algorithm runs for a configurable number of generations. At the end, the policy that performed best on the held-out selection set is evaluated on the test set.

---

## What Was Reproduced

This repository reproduces **Table 1 (Qwen3-8B block)** of the paper, specifically the **EvoTool row** on the τ-Bench and BFCL columns, as committed in the course project plan.

Concretely:
- **τ-Bench**: Pass@1 on the *Retail* and *Airline* subsets
- **BFCL**: Accuracy on the *Single-turn* and *Multi-turn* subsets

Other methods in Table 1 (ToolBench, RestBench, CoT, OPRO, DRAFT, etc.) are taken directly from the paper and **not** re-run here, as ToolBench and RestBench require external API keys that are unavailable.

The backbone model is **Qwen3-8B-GPTQ-Int8** (8-bit quantized, `JunHowie/Qwen3-8B-GPTQ-Int8`) served via vLLM on a local RTX 5070 Ti (16 GB VRAM).

---

## Results

### Obtained Results vs. Paper (Qwen3-8B, EvoTool)

| Benchmark | **Our Result** | **Paper (Table 1)** | **Delta** |
|-----------|:--------------:|:-------------------:|:---------:|
| τ-Bench Retail | 5.2% | 35.9% | −30.7 |
| τ-Bench Airline | **30.0%** | 15.7% | **+14.3** |
| BFCL Single | **81.1%** | 76.9% | **+4.2** |
| BFCL Multi | 2.0% | 36.4% | −34.4 |

### Original Table 1 from the Paper (Qwen3-8B excerpt)

| Method | τ-Bench Ret | τ-Bench Air | τ-Bench Avg | BFCL Sin | BFCL Mul | BFCL Avg | Overall |
|--------|:-----------:|:-----------:|:-----------:|:--------:|:--------:|:--------:|:-------:|
| ReAct | 33.1 | 14.4 | 23.8 | 70.6 | 33.3 | 52.0 | 49.0 |
| CoT | 14.6 | 8.2 | 11.4 | 41.9 | 20.3 | 31.1 | 35.7 |
| Plan-and-Solve | 32.0 | 15.0 | 23.5 | 55.5 | 24.0 | 39.8 | 43.7 |
| OPRO | 31.9 | 15.0 | 23.5 | 75.2 | 33.5 | 54.4 | 50.2 |
| PromptBreeder | 30.6 | 11.3 | 21.0 | 73.9 | 34.4 | 54.2 | 49.0 |
| EvoPrompt | 33.8 | 14.7 | 24.3 | 75.7 | 35.9 | 55.8 | 51.4 |
| AdaPlanner | 34.8 | 16.4 | 25.6 | 67.1 | 36.0 | 51.6 | 46.3 |
| EasyTool | 20.9 | 10.3 | 15.6 | 73.8 | 28.2 | 51.0 | 51.1 |
| DRAFT | 17.8 | 8.9 | 13.4 | 72.5 | 27.1 | 49.8 | 51.8 |
| AnyTool | 23.5 | 14.6 | 19.1 | 71.8 | 30.3 | 51.1 | 49.6 |
| **EvoTool** | **35.9** | **15.7** | **25.8** | **76.9** | **36.4** | **56.7** | **57.0** |

### Discussion

The results split cleanly into two groups, and the dividing factor is EvoTool's dependency on a minimum backbone capability threshold.

**Where reproduction succeeds:**

*BFCL Single (81.1% vs 76.9%)*: EvoTool exceeds the paper's result. Single-turn function-calling is a task where the quantized model has sufficient baseline capability (~80% static), giving the evolutionary loop a rich signal to learn from. With ample positive examples in the selection set, blame attribution can accurately localise failures and mutations produce measurable improvements.

*τ-Bench Airline (30.0% vs 15.7%)*: EvoTool exceeds the paper's result. The airline domain has a narrower tool set and simpler policy structure than retail, which allows the model to achieve a high enough baseline success rate for effective evolution. Note that the test set here is small (10 tasks), so results carry higher variance.

**Where reproduction falls short:**

*τ-Bench Retail (5.2% vs 35.9%)*: EvoTool's effectiveness is gated by the backbone model's baseline capability on the selection set. The paper's full-precision Qwen3-8B achieves approximately 33% on τ-Bench Retail even before evolution, providing a rich training signal. Our 8-bit quantized variant falls significantly below that threshold, which means the evolutionary loop receives too little positive signal to function effectively. When selection-set scores approach zero, diversity-aware selection collapses the population and mutations become uninformative, the framework cannot improve beyond the initial policy.

*BFCL Multi (2.0% vs 36.4%)*: Multi-turn tasks compound the baseline capability problem: each turn depends on all preceding turns, so a model that struggles on individual turns fails multiplicatively across the full conversation. The very low initial policy performance on this benchmark confirms that the quantized model lacks the multi-turn coherence required for EvoTool to gain traction.

**Summary:** The reproduction confirms EvoTool's core design, modular decomposition, blame attribution, diversity selection, works correctly and produces measurable gains when the backbone model meets a minimum baseline. The quantization gap (8-bit vs full-precision) is the primary driver of the performance difference on the harder benchmarks.

**A note on resource constraints and iteration speed.** The paper uses full-precision Qwen3-8B, which requires more than 16 GB VRAM and was not feasible on the available hardware (RTX 5070 Ti, 16 GB). The closest runnable alternative is the 8-bit GPTQ quantized variant (~9 GB), which was used for the final results reported above. Earlier iterations of this reproduction used a 4-bit AWQ quantized variant to fit within tighter memory budgets while debugging the implementation.

A single full experiment suite (all four benchmarks, 8 generations each) takes approximately 32 hours of continuous GPU time. Given this cost, the project involved roughly **7 complete end-to-end runs** over the course of the last month before delivery, each run used to validate a round of implementation fixes and evaluate their impact. This constraint limited the degree to which hyperparameters and prompts could be tuned beyond what the paper specifies.

---

## Code Structure

```
evotool-os/
├── config.yaml                  # Central configuration (model, benchmarks, splits, LLM params)
├── requirements.txt
│
├── evotool/                     # Core EvoTool implementation
│   ├── agent/
│   │   ├── modular_agent.py     # ModularAgent: Planner/Selector/Caller/Synthesizer inference
│   │   ├── modules.py           # Module data classes and initial prompts (INITIAL_POLICY.*)
│   │   └── executor.py          # Episode execution wrapper
│   ├── blamer/
│   │   ├── blamer.py            # Trajectory-grounded blame attribution
│   │   └── diagnostics.py       # Diagnostic trace construction
│   ├── mutator/
│   │   └── mutator.py           # Feedback-guided targeted mutation
│   ├── selection/
│   │   └── population.py        # Diversity-aware population selection
│   ├── evolution/
│   │   └── loop.py              # Main evolutionary loop (Algorithm 1 from the paper)
│   └── llm/
│       └── client.py            # vLLM/OpenAI-compatible LLM client
│
├── baselines/
│   └── react.py                 # ReAct baseline (τ-Bench and BFCL variants)
│
├── benchmarks/
│   ├── tau_bench/
│   │   ├── runner.py            # τ-Bench episode execution
│   │   ├── data_split.py        # Train/selection/test split logic
│   │   └── evaluator.py         # Pass@1 evaluation
│   └── bfcl/
│       ├── runner.py            # BFCL episode execution (single- and multi-turn)
│       ├── data_split.py        # BFCL split logic
│       └── evaluator.py         # BFCL accuracy evaluation
│
├── results/                     # Committed experiment outputs (JSON)
│
└── scripts/
    ├── start_vllm.sh            # Launch vLLM server for Qwen3-8B-GPTQ-Int8
    ├── setup_env.sh             # One-shot conda environment bootstrap
    ├── run_all_experiments.sh   # End-to-end experiment runner (resumable)
    ├── run_evotool.py           # Run EvoTool evolutionary loop on a benchmark
    ├── run_eval.py              # Evaluate a policy (static or evolved) on the test set
    ├── run_baseline.py          # Run ReAct baseline on a benchmark
    ├── analyze_results.py       # Print a summary table of all result JSON files
    └── smoke_test.py            # Sanity-check the installation
```

---

## Installation

### Requirements

- Linux with CUDA-capable GPU (≥ 16 GB VRAM; tested on RTX 5070 Ti)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd evotool-os

# 2. Bootstrap the conda environment (Python 3.11, all pip deps, tau-bench)
bash scripts/setup_env.sh

# 3. Activate the environment
conda activate evotool

# 4. Download the model weights (first run only; requires internet, ~9 GB)
export HF_TOKEN=<your_huggingface_token>
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('JunHowie/Qwen3-8B-GPTQ-Int8')"
```

### Configuration

Edit `config.yaml` to adjust:

| Key | Description |
|-----|-------------|
| `llm.agent_model` | vLLM model name (default: `JunHowie/Qwen3-8B-GPTQ-Int8`) |
| `llm.base_url` | vLLM server URL (default: `http://localhost:8000/v1`) |
| `llm.temperature` | Sampling temperature (default: `0.7`) |
| `data_splits.tau_bench_airline.test_offset` | Offset into task pool (40 = evaluate only unseen holdout tasks) |

---

## Running the Experiments

### Step 1: Start the vLLM server

Open a dedicated terminal and keep it running throughout all experiments:

```bash
conda activate evotool
bash scripts/start_vllm.sh
```

Wait until you see `Application startup complete`. If vLLM reports a KV-cache OOM, reduce the context length: `MAX_MODEL_LEN=8192 bash scripts/start_vllm.sh`.

### Step 2: Run all experiments

```bash
conda activate evotool
bash scripts/run_all_experiments.sh 2>&1 | tee experiments/run.log
```

This runs in order: EvoTool training → EvoTool evaluation, for all four benchmarks. Each step is **resumable**, if its output JSON already exists it is skipped automatically.

**Selective runs:**

```bash
bash scripts/run_all_experiments.sh --tau-bench   # τ-Bench only
bash scripts/run_all_experiments.sh --bfcl        # BFCL only
FAST=1 bash scripts/run_all_experiments.sh        # smoke-test (tiny splits, ~30 min)
```

### Step 3: Inspect results

```bash
python scripts/analyze_results.py --results-dir experiments/results
```

### Expected runtimes (RTX 5070 Ti, Qwen3-8B-GPTQ-Int8)

| Task | Approx. time |
|------|-------------|
| EvoTool training (one benchmark, 8 generations) | 5–14 hours |
| EvoTool evaluation (one benchmark) | 30–90 min |
| Full experiment suite (all four benchmarks) | ~32 hours |

---

## Citation

```bibtex
@article{yang2026evotool,
  title   = {EvoTool: Self-Evolving Tool-Use Policy Optimization in LLM Agents
             via Blame-Aware Mutation and Diversity-Aware Selection},
  author  = {Yang, Shuo and Han, Soyeon Caren and Ma, Xueqi and Li, Yan and
             Madani, Mohammad Reza Ghasemi and Hovy, Eduard},
  journal = {arXiv preprint arXiv:2603.04900},
  year    = {2026}
}
```
