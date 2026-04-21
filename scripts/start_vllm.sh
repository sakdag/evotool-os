#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for Qwen3-8B (AWQ quantized).
#
# Qwen3-8B in bfloat16 (~15.3 GiB) does NOT fit with KV-cache headroom on a
# 16 GiB card.  Use the AWQ-quantized variant (~4-5 GiB) instead.
#
# First-time model download:
#   huggingface-cli download Qwen/Qwen3-8B-AWQ
#
# Fallback – Ollama (no extra download needed if already installed):
#   ollama pull qwen3:8b && ollama serve
#   Then point config.yaml base_url to http://localhost:11434/v1
#
# Usage:
#   bash scripts/start_vllm.sh
#   PORT=8001 MODEL=Qwen/Qwen3-8B-AWQ bash scripts/start_vllm.sh

set -e

PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.90}
# AWQ model: ~4-5 GiB weights, leaves ~10 GiB for KV cache on a 16 GiB card.
MODEL=${MODEL:-"Qwen/Qwen3-8B-AWQ"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

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

# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS gives more accurate memory accounting
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization awq_marlin \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
