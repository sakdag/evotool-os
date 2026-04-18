# CENG502 Project Plan

Q1: Paper Reference

**Title:** EVOTOOL: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection  
**Authors:** Shuo Yang, Soyeon Caren Han, Xueqi Ma, Yan Li, Mohammad Reza Ghasemi Madani, Eduard Hovy  
**Venue:** arXiv preprint (under review), 2026  
**Link:** https://arxiv.org/abs/2603.04900

Q2: What I Will Reproduce

*Table 1, Qwen3-8B block, the last row (the paper's method, EVOTOOL) on the τ-Bench and BFCL columns**, compared against the paper's reported numbers for those same columns.

Concretely, I will implement EVOTOOL from scratch and run it on two benchmarks using Qwen3-8B as the backbone:
- **τ-Bench**: Pass@1 on the Retail and Airline subsets
- **BFCL**: Accuracy on the Single-turn and Multi-turn subsets

Baseline numbers (the other 10 rows in those columns) are taken directly from the paper rather than re-run.

**Resources needed:**
- **Model:** Qwen3-8B (open-source, Apache 2.0, runs locally at Q4 quantization on ~8 GB VRAM via vLLM, will run it locally on RTX 5070TI)
- **Environments:** Both benchmarks are fully self-contained: τ-Bench uses a simulated local database (no external API keys), BFCL uses deterministic ground-truth checking (installable via `pip install bfcl-eval`)
- **API budget:** Blamer and Mutator LLM calls under EVOTOOL implementation might need a stronger reasoning model, planning to test with local Qwen3-8B but if needed I will utilize API based LLM calls, such as GPT-4/5. 

Q3: Justification

τ-Bench and BFCL are the two most self-contained benchmarks in the paper. τ-Bench runs entirely against a local simulated database with deterministic success checking, and BFCL is a pip-installable package with ground-truth evaluation. Neither requires external API keys, a running server, or an LLM-as-judge. Together they also cover complementary aspects of the paper's claims: τ-Bench stresses long-horizon stateful planning where EVOTOOL's blame attribution matters most (and where its margin over baselines is largest), while BFCL stresses precise function-calling syntax in shorter interactions. Using Qwen3-8B locally keeps most agent rollout costs at zero.

ToolBench and RestBench were considered but deprioritized. ToolBench requires a gated RapidAPI server key that has been unresponsive to new requests based on discussion forums. It also requires keys for third-party APIs on RapidAPI, and its evaluation calls GPT-3.5/4 as a judge for every test instance. RestBench is simpler but still requires live API keys for both TMDB and Spotify (including OAuth for Spotify), and with only 157 total instances across both subsets it is too small to split cleanly into the training pool, selection set, and test set that EVOTOOL's evolutionary loop requires. Due to these limitations 2/4 benchmark datasets are selected to run the experiments on. 
