#!/usr/bin/env bash
# aios-scripts-version: 6
# Download (if needed) and launch Qwen3.8-27B on llama-server, tuned for the
# detected GPUs. Defaults = "max decode speed" profile for 2x RTX 5060 Ti 16 GB.
#
# Usage: ./run-qwen.sh [options]
#   --quant <Q>       GGUF quant tag from the HF repo (default auto: UD-Q4_K_XL for 2x16 GB, Q8_0 for >=40 GB, UD-Q3_K_XL for one 16 GB card)
#   --ctx <n>         context length, default 65536
#   --kv <t>          KV cache type f16|q8_0|q4_0, default q8_0 (use q4_0 for 256K ctx)
#   --split <m>       layer|row, default layer (benchmark both with --bench)
#   --no-mtp          disable MTP speculative decoding (required for --parallel > 1)
#   --parallel <n>    request slots, default 1
#   --vision          also load the mmproj vision encoder
#   --port <p>        default 8080      --host <h>  default 0.0.0.0
#   --bench           run llama-bench instead of the server
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BIN_DIR:-$SCRIPT_DIR/bin}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
HF_REPO="unsloth/Qwen3.8-27B-GGUF"   # ships MTP head tensors (blk.*.nextn.*) + fixed chat template

QUANT=auto; CTX=65536; KV=q8_0; SPLIT=layer; MTP=1; PARALLEL=1; VISION=0; PORT=8080; HOST=0.0.0.0; BENCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quant) QUANT="$2"; shift ;;   --ctx) CTX="$2"; shift ;;   --kv) KV="$2"; shift ;;
    --split) SPLIT="$2"; shift ;;   --parallel) PARALLEL="$2"; shift ;;
    --port) PORT="$2"; shift ;;     --host) HOST="$2"; shift ;;
    --no-mtp) MTP=0 ;;  --vision) VISION=1 ;;  --bench) BENCH=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac; shift
done

# --- hardware -----------------------------------------------------------------
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
TOTAL_VRAM=0; NAMES=()
for row in "${GPU_ROWS[@]}"; do IFS=',' read -r n v <<<"$row"; NAMES+=("${n## }"); TOTAL_VRAM=$(( TOTAL_VRAM + ${v## } / 1024 )); done
PHYS_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
printf '\033[1;36mGPUs: %d x [%s]  total VRAM %d GB   CPU cores %s\033[0m\n' "${#GPU_ROWS[@]}" "$(IFS=,; echo "${NAMES[*]}")" "$TOTAL_VRAM" "$PHYS_CORES"

# --- pick quant ---------------------------------------------------------------
# Sizes (GB): Q3_K_M 13.8 | IQ4_XS 15.7 | Q4_K_M 17.1 | Q5_K_M 19.8 | Q6_K 22.9 | Q8_0 29.0
if [[ "$QUANT" == auto ]]; then
  if   (( TOTAL_VRAM >= 40 )); then QUANT=Q8_0
  elif (( TOTAL_VRAM >= 26 )); then QUANT=UD-Q4_K_XL   # 2x16 GB: Unsloth dynamic 4-bit, ~Q4_K_M size, better quality
  elif (( TOTAL_VRAM >= 15 )); then QUANT=UD-Q3_K_XL   # single 16 GB card
  else QUANT=UD-IQ2_XXS; fi
fi
printf '\033[1;36mQuant: %s   ctx: %s   kv: %s   split: %s   mtp: %s\033[0m\n' "$QUANT" "$CTX" "$KV" "$SPLIT" "$MTP"

# --- model source -------------------------------------------------------------
# llama.cpp downloads (with resume) into $LLAMA_CACHE on first use and reuses it after.
# Requires a build with HTTPS support (build-llama.sh sets LLAMA_OPENSSL=ON + libssl-dev).
mkdir -p "$MODEL_DIR"
export LLAMA_CACHE="$MODEL_DIR"
HF_MODEL="$HF_REPO:$QUANT"
printf '\033[1;36mModel: -hf %s  (cache: %s)\033[0m\n' "$HF_MODEL" "$MODEL_DIR"

# --- launch -------------------------------------------------------------------
export CUDA_MODULE_LOADING=LAZY
if [[ $BENCH -eq 1 ]]; then
  # llama-bench takes a file path, so use the cached GGUF (download it first via a normal run)
  find_cached() { find "$MODEL_DIR" -name "*$QUANT*.gguf" ! -name '*mmproj*' 2>/dev/null | head -1; }
  CACHED="$(find_cached)"
  if [[ -z "$CACHED" ]]; then
    # No GGUF yet: let llama-server download it (resumable), then stop it once it reports healthy.
    echo "Downloading $HF_MODEL via llama-server (first time only, ~17 GB)..."
    "$BIN_DIR/llama-server" -hf "$HF_MODEL" --no-mmproj -ngl 0 -c 512 --host 127.0.0.1 --port "$PORT" &
    DL=$!
    while kill -0 "$DL" 2>/dev/null; do
      sleep 3
      curl -fs "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    done
    kill "$DL" 2>/dev/null; wait "$DL" 2>/dev/null || true
    CACHED="$(find_cached)"
    [[ -z "$CACHED" ]] && { echo "Download did not produce a $QUANT GGUF in $MODEL_DIR - check the llama-server output above." >&2; exit 1; }
  fi
  exec "$BIN_DIR/llama-bench" -m "$CACHED" -ngl 999 -fa 1 -sm "$SPLIT" -p 512 -n 128 -r 3
fi

ARGS=(
  -hf "$HF_MODEL"
  --jinja                         # mandatory: Qwen3.8 chat template / stop tokens
  -ngl 999                        # everything on GPU
  -sm "$SPLIT"                    # layer split across both cards (row = try for extra bandwidth)
  -c "$CTX"
  -fa on
  --cache-type-k "$KV" --cache-type-v "$KV"
  -b 2048 -ub 512                 # prompt-processing batch sizes
  -t "$PHYS_CORES"                # CPU threads only matter for the tiny non-offloaded bits
  --parallel "$PARALLEL"
  --host "$HOST" --port "$PORT"
  --metrics
)
if [[ $MTP -eq 1 ]]; then
  [[ $PARALLEL -ne 1 ]] && { echo "MTP requires --parallel 1 (use --no-mtp for multi-slot)" >&2; exit 1; }
  ARGS+=(--spec-type draft-mtp --spec-draft-n-max 2)   # +33-39% decode on consumer GPUs
fi
[[ $VISION -eq 0 ]] && ARGS+=(--no-mmproj)   # -hf auto-downloads the vision encoder unless told not to

echo "llama-server ${ARGS[*]}"
exec "$BIN_DIR/llama-server" "${ARGS[@]}"
