#!/usr/bin/env bash
# 8x Intel Arc Pro B60 burn-in runner
# Ubuntu 24.04.x + Xe KMD + oneAPI / Level Zero
#
# Usage:
#   ./scripts/run_8gpu_burnin.sh              # 1 hour, FP32, 90% VRAM
#   ./scripts/run_8gpu_burnin.sh 7200 -x      # 2 hours, BF16/XMX
#   ./scripts/run_8gpu_burnin.sh 300 -i 0     # 5 min, GPU 0 only
#   ./scripts/run_8gpu_burnin.sh --preflight  # checks only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DUR=3600
PREFLIGHT=0
EXTRA=()
for a in "$@"; do
    if [[ "$a" == "--preflight" ]]; then
        PREFLIGHT=1
    elif [[ "$a" =~ ^[0-9]+$ ]] && [[ ${#EXTRA[@]} -eq 0 ]]; then
        DUR="$a"
    else
        EXTRA+=("$a")
    fi
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

echo "=============================================================="
echo " 8x Arc Pro B60 burn-in pre-check"
echo "=============================================================="
echo " host     : $(hostname)"
echo " kernel   : $(uname -r)"
echo " os       : $(. /etc/os-release; echo "$PRETTY_NAME")"
echo " time     : $(date -Is)"
echo " workdir  : $ROOT"
echo

# oneAPI env
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
    log "sourced /opt/intel/oneapi/setvars.sh"
elif [[ -f /opt/intel/oneapi/compiler/latest/env/vars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null
    log "sourced compiler vars.sh"
else
    log "WARNING: oneAPI setvars.sh not found. icpx/sycl-ls may be missing."
fi

echo
echo "-- render nodes --"
ls -l /dev/dri/renderD* 2>/dev/null || die "no /dev/dri/renderD* nodes (Xe/i915 not bound?)"

echo
echo "-- Intel PCI GPUs (expect 8 x 0xe211 for B60) --"
mapfile -t GPUS < <(lspci -nn | grep -iE 'VGA|Display|3D' | grep -i '8086' || true)
printf '%s\n' "${GPUS[@]:-none}"
B60=$(printf '%s\n' "${GPUS[@]:-}" | grep -ci 'e211' || true)
echo " B60 count (PCI ID 0xe211): ${B60}"

echo
echo "-- Xe/i915 driver --"
if lsmod | grep -q '^xe'; then
    log "xe module loaded"
elif lsmod | grep -q '^i915'; then
    log "i915 module loaded (B60 should be on xe; check dmesg)"
else
    log "WARNING: neither xe nor i915 is loaded"
fi

echo
echo "-- groups --"
id
if ! id -nG | grep -qw render; then
    log "WARNING: user is not in 'render' group. You may need:"
    echo "         sudo gpasswd -a \$USER render && newgrp render"
fi

echo
echo "-- sycl-ls --"
if command -v sycl-ls >/dev/null; then
    ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:*}" sycl-ls || true
else
    log "WARNING: sycl-ls not in PATH"
fi

echo
echo "-- xpu-smi --"
if command -v xpu-smi >/dev/null; then
    xpu-smi discovery || xpu-smi || true
else
    log "xpu-smi not installed (optional). Install: sudo apt install xpu-smi"
    log "sysfs hwmon fallback will be used."
fi

echo
echo "-- power / thermal headroom reminder --"
echo " Each Arc Pro B60 TBP is 120-200 W. Eight cards => 960-1600 W GPU only."
echo " Add CPU, DRAM, fans, NVMe. Size PSU, PDUs, and chassis airflow for the"
echo " full load before starting a multi-hour run."
echo

if [[ "$PREFLIGHT" -eq 1 ]]; then
    log "preflight only; exiting"
    exit 0
fi

if [[ ! -x "$ROOT/xgpu_burn" ]]; then
    log "building xgpu_burn"
    if command -v icpx >/dev/null; then
        if make -C "$ROOT" 2>/tmp/xgpu_make.err; then
            log "built with oneMKL"
        else
            log "oneMKL build failed; building tiled SYCL fallback"
            cat /tmp/xgpu_make.err || true
            make -C "$ROOT" NOMKL=1
        fi
    else
        die "icpx not found. Install Intel oneAPI / OMIX and source setvars.sh"
    fi
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="${XGPU_LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR"
BURN_LOG="$LOGDIR/burn-${STAMP}.log"
TEL_LOG="$LOGDIR/telemetry-${STAMP}.csv"

log "starting telemetry -> $TEL_LOG"
bash "$ROOT/scripts/monitor_xpu.sh" "$DUR" "$TEL_LOG" >/dev/null &
MON_PID=$!
cleanup() {
    g_stop=1
    kill "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "starting xgpu_burn ${DUR}s ${EXTRA[*]:-}"
log "console log -> $BURN_LOG"
set +e
ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:*}" \
    "$ROOT/xgpu_burn" "${EXTRA[@]}" "$DUR" 2>&1 | tee "$BURN_LOG"
RC=${PIPESTATUS[0]}
set -e

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo
log "burn exit code: $RC"
log "logs: $BURN_LOG"
log "telemetry: $TEL_LOG"

if [[ -f "$TEL_LOG" ]]; then
    echo
    echo "-- telemetry peaks (best-effort) --"
    python3 - "$TEL_LOG" <<'PY' || true
import csv, sys, re
path = sys.argv[1]
rows = []
with open(path, errors="replace") as f:
    sample = f.read(4096)
    f.seek(0)
    if "GPU Power" in sample or "DeviceId" in sample:
        # xpu-smi csv; headers vary
        rdr = csv.DictReader(f)
        temps, powers = [], []
        for row in rdr:
            for k,v in row.items():
                if v is None: continue
                kl = k.lower()
                try:
                    x = float(v)
                except Exception:
                    continue
                if "temp" in kl: temps.append(x)
                if "power" in kl or k.strip().endswith("(W)"): powers.append(x)
        if temps: print(f"  max temp listed : {max(temps):.1f}")
        if powers: print(f"  max power listed: {max(powers):.1f} W")
    else:
        rdr = csv.DictReader((ln for ln in f if not ln.startswith("#") and ln.strip()))
        temps, powers = [], []
        for row in rdr:
            try:
                if row.get("temp_c"): temps.append(float(row["temp_c"]))
                if row.get("power_w"): powers.append(float(row["power_w"]))
            except Exception:
                pass
        if temps: print(f"  max core temp : {max(temps):.1f} C")
        if powers: print(f"  max card power: {max(powers):.1f} W")
PY
fi

exit "$RC"
