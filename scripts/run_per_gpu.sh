#!/usr/bin/env bash
# Launch one xgpu_burn process per Level Zero GPU.
# Use this when a single process only heats GPU 0 (common on multi-Arc).
#
# Usage:
#   ./scripts/run_per_gpu.sh 600
#   ./scripts/run_per_gpu.sh 7200 -x

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi

DUR=3600
EXTRA=()
for a in "$@"; do
    if [[ "$a" =~ ^[0-9]+$ ]] && [[ ${#EXTRA[@]} -eq 0 ]]; then
        DUR="$a"
    else
        EXTRA+=("$a")
    fi
done

[[ -x "$ROOT/xgpu_burn" ]] || { echo "build first: make CXX=icpx"; exit 1; }

# Do not inherit a mask that hides cards.
unset ZE_AFFINITY_MASK || true
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"

mapfile -t DEVS < <(ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls 2>/dev/null | grep -E 'level_zero|ext_oneapi_level_zero' || true)
N=${#DEVS[@]}
if [[ "$N" -eq 0 ]]; then
    echo "sycl-ls saw no Level Zero GPUs. Output:"
    sycl-ls || true
    exit 1
fi

echo "Launching $N per-GPU processes for ${DUR}s"
mkdir -p "$ROOT/logs"
STAMP=$(date +%Y%m%d-%H%M%S)
PIDS=()
for i in $(seq 0 $((N - 1))); do
    LOG="$ROOT/logs/gpu${i}-${STAMP}.log"
    echo "  GPU $i -> $LOG"
    # Affinity first so this process only sees one physical card.
    # After the mask, that card is Level Zero device 0.
    env ZE_AFFINITY_MASK="$i" \
        ONEAPI_DEVICE_SELECTOR=level_zero:0 \
        "$ROOT/xgpu_burn" "${EXTRA[@]}" "$DUR" >"$LOG" 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Watch:  xpu-smi dump -d 0,1,2,3,4,5,6,7 -m 0,1,2,3 -i 1"
echo "        tail -f logs/gpu0-${STAMP}.log"

RC=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "GPU $i exited non-zero"
        RC=1
    fi
done
echo "done (rc=$RC)"
exit "$RC"
