#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for Qwen3-8B (GPTQ-Int8 quantized).
#
# Model quality ladder (RTX 5070 Ti, 16 GiB GDDR7):
#   Qwen/Qwen3-8B-AWQ         ~4-5 GiB weights  (4-bit,  lowest quality, current)
#   JunHowie/Qwen3-8B-GPTQ-Int8 ~8-9 GiB weights (8-bit,  significantly better)
#   Qwen/Qwen3-8B             ~15-16 GiB weights (BF16,   too tight with KV cache)
#   Qwen/Qwen3-14B-AWQ        ~23 GiB total      (too large for 16 GiB card)
#
# First-time model download (run once, requires internet — comment out HF_HUB_OFFLINE below):
#   huggingface-cli download JunHowie/Qwen3-8B-GPTQ-Int8
#
# Fallback to AWQ if GPTQ-Int8 is not yet downloaded:
#   MODEL=Qwen/Qwen3-8B-AWQ bash scripts/start_vllm.sh
#
# Usage:
#   bash scripts/start_vllm.sh
#   PORT=8001 MODEL=JunHowie/Qwen3-8B-GPTQ-Int8 bash scripts/start_vllm.sh

set -e

PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.90}
# GPTQ-Int8: ~8-9 GiB weights, leaves ~6-7 GiB for KV cache on a 16 GiB card.
# Reduce MAX_MODEL_LEN if you see KV-cache OOM at startup.
MODEL=${MODEL:-"JunHowie/Qwen3-8B-GPTQ-Int8"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

echo "=== Starting vLLM server ==="
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "GPU util:     $GPU_UTIL"
echo "Max seq len:  $MAX_MODEL_LEN"
echo ""
echo "Tip: if you see KV-cache OOM, reduce MAX_MODEL_LEN or lower GPU_UTIL."
echo ""

export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="dummy"

# Run fully offline — model is already cached locally.
# Without these, vLLM/transformers tries to reach HuggingFace on startup and
# crashes when there is no network (e.g. after a machine reboot).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS gives more accurate memory accounting
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
