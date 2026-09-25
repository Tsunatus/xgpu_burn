# xgpu-burn

GPU_Burn equivalent for **Intel Arc / Xe** GPUs. Written for an 8× **Intel Arc Pro B60** (Battlemage Xe2, PCI ID `0xE211`) burn-in on **Ubuntu 24.04.5 LTS** / **kernel 7.0.0-34-generic**, but it will run on any Level Zero GPU `sycl-ls` can see.

It does what `wilicc/gpu-burn` does on NVIDIA:

- fill most of VRAM
- hammer large GEMMs continuously
- compare results for silent compute errors
- print per-GPU throughput, temperature, power, clock

Original `gpu-burn` is CUDA/cuBLAS only. This uses **SYCL + oneMKL** (XMX) or a tiled SYCL GEMM fallback.

## What an 8× B60 system should look like

| Item | Per card | ×8 |
|---|---|---|
| Architecture | Xe2-HPG, 20 Xe-cores, 160 XMX | — |
| Memory | 24 GB GDDR6, 456 GB/s | 192 GB |
| Peak FP32 | 12.28 TFLOPS | ~98 TFLOPS |
| Peak INT8 | 197 TOPS | ~1576 TOPS |
| Boost clock | 2400 MHz | — |
| Board power | 120–200 W TBP | **960–1600 W GPU only** |
| PCIe | Gen5 x8 electrical | watch lane bifurcation |

That 1600 W figure is GPU board power only. Add CPU, DIMMs, fans, NVMe, and conversion loss before you call the chassis or PDU “sized.” The point of this tool is to find whether the platform can hold frequency and stay thermally stable at that load.

## Software stack on Ubuntu 24.04.5

B60 (`0xe211`) wants the **Xe** KMD. Kernel 7.0 is new enough.

You need a user-mode compute stack on top of that:

1. Level Zero + Compute Runtime (`libze1`, `libze-intel-gpu1`, `intel-opencl-icd`)
2. Intel oneAPI DPC++ (`icpx`) so this binary can compile
3. oneMKL (optional but recommended — this is how you exercise **XMX**)
4. `xpu-smi` for clean telemetry (optional; sysfs hwmon is the fallback)

Two supported install paths:

- **Intel OMIX** — pinned driver + compiler + oneMKL for Arc Pro B50/B60/B65/B70 on Ubuntu 24.04. This is the least painful path for 8-card servers.
- **Intel graphics PPA + oneAPI Base Toolkit** — also works. Do not mix PPA GPU packages with OMIX on the same image.

Verify before building:

```bash
uname -r
lspci -nn | grep -iE 'VGA|Display|3D' | grep 8086
ls /dev/dri/renderD*
groups            # you want 'render'
source /opt/intel/oneapi/setvars.sh
sycl-ls
xpu-smi discovery   # if installed
```

You want **eight** Level Zero GPU devices and eight `0xe211` PCI functions. If `sycl-ls` shows OpenCL copies of the same cards, the binary prefers Level Zero and dedups.

If a card is invisible, check `xe.force_probe=e211` on the kernel command line (needed on some 24.04 + Core Series 2 combinations) and that the user is in the `render` group:

```bash
sudo gpasswd -a $USER render
# log out / newgrp render
```

## Build

```bash
cd xgpu-burn
source /opt/intel/oneapi/setvars.sh
make                # oneMKL + XMX
# or
make NOMKL=1        # tiled SYCL only, no MKL
```

`icpx` is required. gcc cannot compile the SYCL source.

## Run

```bash
# list cards + current sysfs temp/power
./xgpu_burn -l

# 10 minute smoke test, all GPUs, FP32, 90% VRAM
./xgpu_burn 600

# 2 hour XMX/BF16 burn (highest power on B60 if oneMKL built)
./xgpu_burn -x -m 90% 7200

# FP64 (native on Xe2, lower power than XMX, good for data-path check)
./xgpu_burn -d 1800

# single card
./xgpu_burn -i 0 300
```

Flags match gpu-burn as closely as they can:

| Flag | Meaning |
|---|---|
| `TIME` | seconds (default 60) |
| `-m X` | X MiB per GPU |
| `-m N%` | percent of device memory (default 90%) |
| `-d` | FP64 |
| `-x` / `-tc` | BF16 GEMM via oneMKL XMX |
| `-n SIZE` | square matrix dim (default 8192) |
| `-i N` | one GPU |
| `-l` | list and exit |

### 8-card burn-in wrapper

```bash
chmod +x scripts/*.sh
./scripts/run_8gpu_burnin.sh --preflight
./scripts/run_8gpu_burnin.sh 3600 -x
```

The wrapper sources oneAPI, builds if needed, runs `xgpu_burn`, and logs `xpu-smi dump` (or sysfs) to `logs/`.

Standalone monitor:

```bash
./scripts/monitor_xpu.sh 3600 logs/telemetry.csv
```

`xpu-smi` metrics used: engine util, power, frequency, core temp, memory temp, memory util.

## How to read the result

A healthy 8× B60 run looks like:

- all 8 GPUs stay in the burn (no SYCL abort, no device lost)
- `err=0` on every card for the whole window
- clocks hold near 2.0–2.4 GHz instead of collapsing after a few minutes
- per-card power sits in the 150–200 W band under `-x` / large FP32 GEMM
- core / memory temps plateau, they do not walk into throttle or shutdown
- chassis inlet-to-outlet and PDU watts match ~1.0–1.8 kW GPU load plus the rest of the box

Red flags that mean the *system* is the limit, not the kernel:

- frequency droop on some slots only (PCIe / VRM / local airflow)
- one or two cards many tens of watts below the others
- compute mismatches (`FAULT`) after the box is hot — possible rail sag or thermal error
- device reset in `dmesg` (`xe`, `drm`, `AER`)
- BMC / PSU fault while the burn is running

FP32 peak on paper is 12.28 TFLOPS/card. oneMKL will get much closer than the tiled fallback. The tiled `NOMKL` build is still enough to heat the cards; do not treat its TFLOPS number as a spec check.

Suggested sequence:

1. `./scripts/run_8gpu_burnin.sh --preflight`
2. 5 minutes, one GPU: `./xgpu_burn -i 0 300`
3. 10 minutes, all 8, FP32: `./xgpu_burn 600`
4. 2 hours, all 8, XMX: `./xgpu_burn -x 7200`
5. Overnight if the 2 hour plateau is clean: `./xgpu_burn -x 28800`

Watch `dmesg -w`, BMC sensors, and PDU watts in parallel.

## Implementation notes

- One in-order SYCL queue per GPU, one host thread per GPU. Single process, not one process per card (Battlemage has had multi-process `zeInit` pain on some compute-runtime versions).
- Default GEMM is 8192×8192, same ballpark as gpu-burn. Remaining VRAM is filled with extra result slots so the memory subsystem stays busy.
- Every 8th iteration compares the latest result matrix against the first. Mismatches, NaNs, and infinities count as errors.
- Telemetry is read from Xe/i915 hwmon sysfs. It is best-effort; `xpu-smi` is more complete when installed.
- `-x` needs the oneMKL build. Without MKL the binary falls back to FP32 tiled GEMM.

## Safety

This will drive eight 200 W boards toward TBP for as long as you let it. Confirm PSU headroom, cable seating, case airflow, and inlet temperature first. The tool will not protect you from an undersized power budget.
