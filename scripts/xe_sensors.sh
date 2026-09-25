#!/usr/bin/env bash
# Live Xe/B60 sensors from sysfs. No extra packages.
# Usage: ./scripts/xe_sensors.sh [interval_seconds]

set -euo pipefail
INT="${1:-1}"

echo "timestamp pci drm hwmon freq_mhz temp_max_c energy1_uJ energy2_uJ power_w cap_w"

declare -A PREV_E PREV_T

while true; do
    ts=$(date +%H:%M:%S)
    sum_w=0
    n=0
    for card in /sys/class/drm/card[0-9]*; do
        name=$(basename "$card")
        [[ "$name" == *-* ]] && continue
        [[ -f "$card/device/vendor" ]] || continue
        grep -q 8086 "$card/device/vendor" || continue
        [[ -f "$card/device/device" ]] || continue
        devid=$(cat "$card/device/device")
        [[ "$devid" == *e211* || "$devid" == *E211* ]] || continue
        pci=$(grep PCI_SLOT_NAME "$card/device/uevent" 2>/dev/null | cut -d= -f2)
        freq=$(cat "$card/device/tile0/gt0/freq0/cur_freq" 2>/dev/null || echo)
        real=$(readlink -f "$card/device")
        hwmon=""
        for h in /sys/class/hwmon/hwmon*; do
            [[ -f "$h/name" ]] || continue
            [[ $(cat "$h/name") == xe ]] || continue
            href=$(readlink -f "$h/device" 2>/dev/null || true)
            [[ "$href" == "$real" ]] && { hwmon=$h; break; }
        done
        tmax=""
        e1=""; e2=""; cap=""
        if [[ -n "$hwmon" ]]; then
            tmax=$(cat "$hwmon"/temp*_input 2>/dev/null | awk 'BEGIN{m=-1} {if($1>m)$1=$1; if($1>m)m=$1} END{if(m>0) printf "%.1f", m/1000}')
            # max of plausible GPU temps
            tmax=$(for t in "$hwmon"/temp*_input; do awk '{v=$1/1000; if(v>=1 && v<=125) print v}' "$t"; done | awk 'BEGIN{m=-1} {if($1>m)m=$1} END{if(m>0) printf "%.1f", m}')
            e1=$(cat "$hwmon/energy1_input" 2>/dev/null || echo)
            e2=$(cat "$hwmon/energy2_input" 2>/dev/null || echo)
            cap=$(awk '{printf "%.0f", $1/1e6}' "$hwmon/power1_cap" 2>/dev/null || echo)
        fi
        pw=""
        now=$(date +%s.%N)
        key=${pci:-$name}
        if [[ -n "$e1" && -n "${PREV_E[$key]:-}" ]]; then
            pw=$(python3 - "$e1" "${PREV_E[$key]}" "$now" "${PREV_T[$key]}" <<'PY'
import sys
e, pe, t, pt = map(float, sys.argv[1:])
dt = t-pt
if dt > 0.2:
    print(f"{((e-pe)/dt)/1e6:.1f}")
PY
)
        fi
        PREV_E[$key]="$e1"
        PREV_T[$key]="$now"
        if [[ -n "$pw" ]]; then
            sum_w=$(python3 -c "print($sum_w + $pw)")
            n=$((n+1))
        fi
        printf '%s %s %s %s %s %s %s %s %s %s\n' \
            "$ts" "${pci:-?}" "$name" "${hwmon:-?}" "${freq:-?}" "${tmax:-?}" \
            "${e1:-?}" "${e2:-?}" "${pw:-?}" "${cap:-?}"
    done
    echo "-- total_power_w=${sum_w:-?} cards_with_power=$n --"
    sleep "$INT"
done
