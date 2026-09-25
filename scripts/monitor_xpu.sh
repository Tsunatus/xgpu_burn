#!/usr/bin/env bash
# Live telemetry logger for Intel Arc / Xe GPUs.
# Prefers xpu-smi dump; falls back to sysfs hwmon.
#
# Usage: monitor_xpu.sh [duration_seconds] [outfile.csv]

set -u
DUR="${1:-3600}"
OUT="${2:-xgpu_burn_telemetry.csv}"

have_xpumi=0
if command -v xpu-smi >/dev/null 2>&1; then
    have_xpumi=1
fi

echo "# xgpu-burn telemetry  start=$(date -Is)  duration=${DUR}s" | tee "$OUT"

if [[ "$have_xpumi" -eq 1 ]]; then
    echo "# source=xpu-smi" | tee -a "$OUT"
    # metrics: 0 util, 1 power, 2 freq, 3 core temp, 4 mem temp, 5 mem util
    # device list: all
    # -i 1 second, --time duration
    if xpu-smi dump -h 2>&1 | grep -q -- '--time'; then
        xpu-smi dump -d 0,1,2,3,4,5,6,7 -m 0,1,2,3,4,5 -i 1 --time "$DUR" | tee -a "$OUT"
        exit 0
    fi
    # older xpu-smi: -n samples
    xpu-smi dump -d 0,1,2,3,4,5,6,7 -m 0,1,2,3,4,5 -i 1 -n "$DUR" | tee -a "$OUT"
    exit 0
fi

echo "# source=sysfs-hwmon" | tee -a "$OUT"
echo "timestamp,card,pci,device_id,temp_c,mem_temp_c,power_w,freq_mhz" | tee -a "$OUT"

end=$((SECONDS + DUR))
while (( SECONDS < end )); do
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    for card in /sys/class/drm/card[0-9]*; do
        [[ -e "$card/device/vendor" ]] || continue
        name=$(basename "$card")
        [[ "$name" == *-* ]] && continue
        vend=$(cat "$card/device/vendor" 2>/dev/null || true)
        [[ "$vend" == *8086* ]] || continue
        pci=$(grep PCI_SLOT_NAME "$card/device/uevent" 2>/dev/null | cut -d= -f2)
        devid=$(cat "$card/device/device" 2>/dev/null || echo na)
        hwmon_dir=$(ls -d "$card/device/hwmon/hwmon"* 2>/dev/null | head -1 || true)
        temp=""; mtemp=""; pwr=""; freq=""
        if [[ -n "${hwmon_dir:-}" ]]; then
            [[ -f "$hwmon_dir/temp1_input" ]] && temp=$(awk '{printf "%.1f", $1/1000}' "$hwmon_dir/temp1_input")
            [[ -f "$hwmon_dir/temp2_input" ]] && mtemp=$(awk '{printf "%.1f", $1/1000}' "$hwmon_dir/temp2_input")
            if [[ -f "$hwmon_dir/power1_average" ]]; then
                pwr=$(awk '{printf "%.1f", $1/1e6}' "$hwmon_dir/power1_average")
            elif [[ -f "$hwmon_dir/power1_input" ]]; then
                pwr=$(awk '{printf "%.1f", $1/1e6}' "$hwmon_dir/power1_input")
            fi
        fi
        if [[ -f "$card/device/gt/gt0/freq0_cur_freq" ]]; then
            freq=$(cat "$card/device/gt/gt0/freq0_cur_freq")
        elif [[ -f "$card/gt_cur_freq_mhz" ]]; then
            freq=$(cat "$card/gt_cur_freq_mhz")
        fi
        echo "$ts,$name,${pci:-},$devid,${temp:-},${mtemp:-},${pwr:-},${freq:-}" | tee -a "$OUT"
    done
    sleep 1
done
