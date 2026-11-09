#!/usr/bin/env bash
# Benchmark real local LLMs served by Ollama on the golden dataset.
# Usage: ./scripts/run_ollama_benchmark.sh [model ...]
set -euo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("qwen2.5:7b" "llama3.1:8b" "llama3.2:3b")   # all support tool calling
fi

command -v ollama >/dev/null || { echo "Ollama not found: install it from ollama.com"; exit 1; }
command -v uv >/dev/null || { echo "uv not found: run 'pip install uv'"; exit 1; }
curl -s http://localhost:11434/api/tags >/dev/null || { echo "Start Ollama first (open the Ollama app)"; exit 1; }

cd "$(dirname "$0")/.."
uv sync --extra dev -q

SPECS=()
for m in "${MODELS[@]}"; do
  echo ">> pulling $m"
  ollama pull "$m"
  SPECS+=("ollama:$m")
done

uv run agent-eval \
  --models "${SPECS[@]}" \
  --prompts baseline tool_first \
  --judge "ollama:${MODELS[0]}" \
  --out results/ollama

echo
echo "Done. Results: results/ollama/summary.md"
