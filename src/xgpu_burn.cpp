/*
 * xgpu-burn — GPU_Burn-style stress test for Intel Arc / Xe GPUs
 *
 * Repeated large GEMMs (oneMKL when available, tiled SYCL otherwise),
 * VRAM fill, result comparison, and per-GPU temp/power/clock readout.
 *
 * Built for Intel Arc Pro B60 (Xe2 Battlemage, PCI 0xE211) multi-GPU
 * burn-in on Ubuntu 24.04 + Xe KMD. Works on other Level Zero GPUs too.
 *
 * License: BSD-2-Clause (same spirit as original gpu-burn by Ville Timonen)
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits.h>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif

using clock_tp = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};

static void on_signal(int) { g_stop.store(true); }

enum class Prec { FP32, FP64, BF16 };

struct Options {
    int seconds = 60;
    int device_id = -1;          // -1 = all
    double mem_frac = 0.90;
    long mem_mb = -1;
    Prec prec = Prec::FP32;
    bool list_only = false;
    bool use_xmx = false;
    int matrix_n = 8192;
};

struct Telemetry {
    double temp_c = -1;
    double mem_temp_c = -1;
    double power_w = -1;
    double freq_mhz = -1;
};

struct GpuStatus {
    std::string name;
    std::string pci;
    std::atomic<uint64_t> iters{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<double> tflops{0};
    std::atomic<int> alive{0};
    std::atomic<size_t> used_bytes{0};
    std::atomic<int> n{0};
};

static void usage(const char* argv0) {
    std::cerr
        << "xgpu-burn — Intel Arc / Xe GPU stress test (GPU_Burn equivalent)\n"
        << "Usage: " << argv0 << " [OPTIONS] [TIME_SECONDS]\n\n"
        << "  -m X      Use X MiB of device memory per GPU\n"
        << "  -m N%     Use N% of device memory (default 90%)\n"
        << "  -d        Use FP64 (DGEMM)\n"
        << "  -x, -tc   Use BF16 / XMX path (oneMKL) when available\n"
        << "  -n SIZE   Square matrix dimension (default 8192)\n"
        << "  -i N      Run only on GPU N\n"
        << "  -l        List GPUs and exit\n"
        << "  -h        Help\n\n"
        << "Examples:\n"
        << "  " << argv0 << " -l\n"
        << "  " << argv0 << " 3600\n"
        << "  " << argv0 << " -x -m 90% 7200\n"
        << "  " << argv0 << " -d -i 0 300\n";
}

static bool starts_with(const std::string& s, const char* p) {
    return s.rfind(p, 0) == 0;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "-l") {
            o.list_only = true;
        } else if (a == "-d") {
            o.prec = Prec::FP64;
        } else if (a == "-x" || a == "-tc") {
            o.use_xmx = true;
            o.prec = Prec::BF16;
        } else if (a == "-i" && i + 1 < argc) {
            o.device_id = std::atoi(argv[++i]);
        } else if (a == "-n" && i + 1 < argc) {
            o.matrix_n = std::atoi(argv[++i]);
        } else if (a == "-m" && i + 1 < argc) {
            std::string v = argv[++i];
            if (!v.empty() && v.back() == '%') {
                o.mem_frac = std::atof(v.c_str()) / 100.0;
            } else {
                o.mem_mb = std::atol(v.c_str());
            }
        } else if (a[0] != '-') {
            o.seconds = std::atoi(a.c_str());
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (o.matrix_n < 256) o.matrix_n = 256;
    if (o.mem_frac < 0.10) o.mem_frac = 0.10;
    if (o.mem_frac > 0.95) o.mem_frac = 0.95;
    if (o.seconds < 1) o.seconds = 1;
    return o;
}

static const char* backend_name(sycl::backend b) {
    switch (b) {
        case sycl::backend::ext_oneapi_level_zero: return "level_zero";
        case sycl::backend::opencl: return "opencl";
        default: return "other";
    }
}

static std::vector<sycl::device> intel_gpus() {
    std::vector<sycl::device> l0, ocl;
    for (auto& p : sycl::platform::get_platforms()) {
        for (auto& d : p.get_devices(sycl::info::device_type::gpu)) {
            auto backend = d.get_backend();
            if (backend == sycl::backend::ext_oneapi_level_zero)
                l0.push_back(d);
            else if (backend == sycl::backend::opencl)
                ocl.push_back(d);
        }
    }
    // Never mix Level Zero and OpenCL views of the same cards.
    // Name+VRAM dedup is wrong on 8x B60: every card has the same name
    // and the same 24 GiB, so it collapses the set.
    if (!l0.empty()) return l0;
    return ocl;
}

static std::string pci_hint(const sycl::device& d) {
    try {
        return d.get_info<sycl::info::device::name>();
    } catch (...) {
        return "unknown";
    }
}

/* ---- sysfs telemetry (Xe / i915 hwmon) ---- */

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static std::string first_line(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string s;
    std::getline(f, s);
    return s;
}

static double read_num(const std::string& path) {
    auto s = read_file(path);
    if (s.empty()) return -1;
    try {
        return std::stod(s);
    } catch (...) {
        return -1;
    }
}

struct HwmonCard {
    std::string drm;     // cardN
    std::string pci;     // 0000:xx:yy.z
    std::string hwmon;   // /sys/class/hwmon/hwmonX or drm .../hwmon/hwmonX
    std::string device_id;
    std::string freq_path;
};

static std::string first_existing(const std::vector<std::string>& paths) {
    for (auto& p : paths) {
        std::ifstream f(p);
        if (f) return p;
    }
    return {};
}

static std::vector<HwmonCard> scan_hwmon() {
    std::vector<HwmonCard> cards;
    DIR* drm = opendir("/sys/class/drm");
    if (!drm) return cards;
    while (auto* ent = readdir(drm)) {
        std::string name = ent->d_name;
        if (!starts_with(name, "card")) continue;
        if (name.find('-') != std::string::npos) continue;
        std::string base = "/sys/class/drm/" + name + "/device";
        std::string vendor = read_file(base + "/vendor");
        if (vendor.find("0x8086") == std::string::npos) continue;
        // Skip integrated GPUs when discrete B60s are present (0xe211).
        std::string devid = first_line(base + "/device");
        HwmonCard c;
        c.drm = name;
        c.device_id = devid;
        std::string uevent = read_file(base + "/uevent");
        auto pos = uevent.find("PCI_SLOT_NAME=");
        if (pos != std::string::npos) {
            auto line = uevent.substr(pos + 14);
            auto nl = line.find_first_of("\r\n");
            c.pci = line.substr(0, nl);
        } else {
            c.pci = name;
        }

        char card_real[PATH_MAX];
        std::string card_dev;
        if (realpath(base.c_str(), card_real)) card_dev = card_real;

        DIR* hm = opendir((base + "/hwmon").c_str());
        if (hm) {
            while (auto* h = readdir(hm)) {
                std::string hn = h->d_name;
                if (starts_with(hn, "hwmon")) {
                    c.hwmon = base + "/hwmon/" + hn;
                    break;
                }
            }
            closedir(hm);
        }
        {
            DIR* cls = opendir("/sys/class/hwmon");
            if (cls) {
                while (auto* h = readdir(cls)) {
                    std::string hn = h->d_name;
                    if (!starts_with(hn, "hwmon")) continue;
                    std::string hp = std::string("/sys/class/hwmon/") + hn;
                    std::string nm = first_line(hp + "/name");
                    if (nm != "xe" && nm != "i915") continue;
                    char hreal[PATH_MAX];
                    std::string href = hp + "/device";
                    bool same = false;
                    if (!card_dev.empty() && realpath(href.c_str(), hreal))
                        same = (card_dev == hreal);
                    if (!same) {
                        std::string slot = read_file(href + "/uevent");
                        if (!c.pci.empty() && slot.find(c.pci) != std::string::npos)
                            same = true;
                    }
                    if (same) {
                        c.hwmon = hp;
                        break;
                    }
                }
                closedir(cls);
            }
        }

        c.freq_path = first_existing({
            base + "/tile0/gt0/freq0/cur_freq",
            base + "/gt/gt0/freq0_cur_freq",
            base + "/gt/gt0/freq0/cur_freq",
            "/sys/class/drm/" + name + "/gt_cur_freq_mhz",
            base + "/gt_cur_freq_mhz",
        });
        cards.push_back(c);
    }
    closedir(drm);

    // Prefer discrete B60 (0xe211) if mixed with iGPU
    std::vector<HwmonCard> dgpu;
    for (auto& c : cards)
        if (c.device_id.find("e211") != std::string::npos ||
            c.device_id.find("E211") != std::string::npos)
            dgpu.push_back(c);
    if (!dgpu.empty()) return dgpu;
    return cards;
}

static Telemetry read_telemetry(const HwmonCard& c) {
    Telemetry t;
    if (!c.hwmon.empty()) {
        // Xe on Battlemage: temp2 package, temp3 VRAM on some kernels;
        // temp1 may be missing or labeled unused.
        for (int i = 2; i <= 17; ++i) {
            double v = read_num(c.hwmon + "/temp" + std::to_string(i) + "_input");
            if (v <= 0) continue;
            v /= 1000.0;
            if (v < 1.0 || v > 120.0) continue;
            std::string lab = first_line(c.hwmon + "/temp" + std::to_string(i) + "_label");
            if (t.temp_c < 0 || v > t.temp_c) t.temp_c = v;
            if (lab.find("vram") != std::string::npos ||
                lab.find("mem") != std::string::npos || i == 3)
                t.mem_temp_c = v;
        }
        double pavg = read_num(c.hwmon + "/power1_average");
        if (pavg < 0) pavg = read_num(c.hwmon + "/power2_average");
        double pinput = read_num(c.hwmon + "/power1_input");
        if (pinput < 0) pinput = read_num(c.hwmon + "/power2_input");
        if (pavg > 0) t.power_w = pavg / 1e6;
        else if (pinput > 0) t.power_w = pinput / 1e6;
        else {
            // Instantaneous power often missing; derive W from energy µJ.
            static std::mutex mu;
            static std::map<std::string, std::pair<double, uint64_t>> prev;
            double e = read_num(c.hwmon + "/energy1_input");
            if (e < 0) e = read_num(c.hwmon + "/energy2_input");
            if (e > 0) {
                auto now = std::chrono::steady_clock::now().time_since_epoch();
                double sec = std::chrono::duration<double>(now).count();
                std::lock_guard<std::mutex> g(mu);
                auto it = prev.find(c.pci.empty() ? c.drm : c.pci);
                uint64_t eu = uint64_t(e);
                if (it != prev.end()) {
                    double dt = sec - it->second.first;
                    uint64_t de = eu - it->second.second;
                    if (dt > 0.2) t.power_w = (double(de) / dt) / 1e6;
                }
                prev[c.pci.empty() ? c.drm : c.pci] = {sec, eu};
            }
        }
    }
    if (!c.freq_path.empty()) {
        double f = read_num(c.freq_path);
        if (f > 0) t.freq_mhz = f;
    }
    return t;
}

/* ---- kernels ---- */

template <typename T>
void fill_host(std::vector<T>& v, unsigned seed) {
    // Cheap deterministic values in (0,1] — stable enough for compare.
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = T((s >> 8) & 0xFFFF) / T(65536);
        v[i] += T(0.125);
    }
}

#ifdef USE_ONEMKL
template <typename T>
sycl::event mkl_gemm(sycl::queue& q, int n, T alpha, const T* A, const T* B,
                     T beta, T* C, const std::vector<sycl::event>& deps) {
    return oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        n, n, n, alpha, A, n, B, n, beta, C, n, deps);
}
#endif

// Tiled SGEMM / DGEMM. Not peak, but keeps Xe-cores + caches + VRAM busy
// when oneMKL is not linked.
template <typename T>
sycl::event tiled_gemm(sycl::queue& q, int n, const T* A, const T* B, T* C) {
    constexpr int TS = 16;
    const int ng = (n + TS - 1) / TS;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 2> As({TS, TS}, h);
        sycl::local_accessor<T, 2> Bs({TS, TS}, h);
        h.parallel_for(sycl::nd_range<2>({size_t(ng * TS), size_t(ng * TS)},
                                         {size_t(TS), size_t(TS)}),
                       [=](sycl::nd_item<2> it) {
                           const int row = int(it.get_global_id(0));
                           const int col = int(it.get_global_id(1));
                           const int lr = int(it.get_local_id(0));
                           const int lc = int(it.get_local_id(1));
                           T acc = T(0);
                           const int tiles = (n + TS - 1) / TS;
                           for (int t = 0; t < tiles; ++t) {
                               const int a_col = t * TS + lc;
                               const int b_row = t * TS + lr;
                               As[lr][lc] = (row < n && a_col < n) ? A[a_col * n + row] : T(0);
                               Bs[lr][lc] = (b_row < n && col < n) ? B[col * n + b_row] : T(0);
                               sycl::group_barrier(it.get_group());
                               for (int k = 0; k < TS; ++k) acc += As[lr][k] * Bs[k][lc];
                               sycl::group_barrier(it.get_group());
                           }
                           if (row < n && col < n) C[col * n + row] = acc;
                       });
    });
}

template <typename T>
sycl::event compare_results(sycl::queue& q, const T* C0, const T* Ci, size_t elems,
                            int* faults, T eps) {
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> i) {
            T a = C0[i];
            T b = Ci[i];
            T d = a > b ? a - b : b - a;
            if (!(d <= eps)) {
                // NaN / Inf / mismatch
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    at(*faults);
                at.fetch_add(1);
            }
        });
    });
}

static const char* prec_name(Prec p) {
    switch (p) {
        case Prec::FP64: return "fp64";
        case Prec::BF16: return "bf16-xmx";
        default: return "fp32";
    }
}

template <typename T>
void burn_typed(sycl::queue& q, const Options& opt, GpuStatus& st, size_t budget) {
    const int n = opt.matrix_n;
    const size_t elems = size_t(n) * size_t(n);
    const size_t mat_bytes = elems * sizeof(T);

    // A + B + Nresult * C
    if (budget < mat_bytes * 3) {
        std::cerr << "Not enough memory for 3 matrices of " << n << "x" << n << "\n";
        return;
    }
    size_t nC = (budget - 2 * mat_bytes) / mat_bytes;
    if (nC < 2) nC = 2;
    // Compare only uses C[0] vs C[1]; extra slots exist to occupy VRAM.
    if (nC > 512) nC = 512;

    size_t used = mat_bytes * (2 + nC);
    size_t leftover = budget > used + (8ull << 20) ? budget - used : 0;
    leftover &= ~size_t(4095);

    st.used_bytes.store(used + leftover);
    st.n.store(n);

    T* A = sycl::malloc_device<T>(elems, q);
    T* B = sycl::malloc_device<T>(elems, q);
    std::vector<T*> C(nC);
    for (size_t i = 0; i < nC; ++i) C[i] = sycl::malloc_device<T>(elems, q);
    int* faults = sycl::malloc_device<int>(1, q);
    char* scratch = leftover ? sycl::malloc_device<char>(leftover, q) : nullptr;
    if (!A || !B || !faults) {
        std::cerr << "Device malloc failed\n";
        return;
    }
    for (auto* p : C) {
        if (!p) {
            std::cerr << "Device malloc failed (C)\n";
            return;
        }
    }

    std::vector<T> hA(elems), hB(elems);
    fill_host(hA, 0xA5A5u);
    fill_host(hB, 0x5A5Au);
    q.memcpy(A, hA.data(), mat_bytes).wait();
    q.memcpy(B, hB.data(), mat_bytes).wait();
    q.memset(faults, 0, sizeof(int)).wait();
    if (scratch) q.memset(scratch, 0x5A, leftover).wait();

    const T alpha = T(1);
    const T beta = T(0);
    T eps;
    if constexpr (std::is_same_v<T, double>) eps = T(1e-6);
    else eps = T(1e-2);

    auto t0 = clock_tp::now();
    uint64_t local_iters = 0;
    size_t slot = 0;

    auto do_gemm = [&](T* out) -> sycl::event {
#ifdef USE_ONEMKL
        return mkl_gemm<T>(q, n, alpha, A, B, beta, out, {});
#else
        (void)alpha;
        (void)beta;
        return tiled_gemm<T>(q, n, A, B, out);
#endif
    };

    // Prime reference result into C[0]
    do_gemm(C[0]).wait();
    local_iters++;

    const int inflight = 8;
    while (!g_stop.load()) {
        for (int k = 0; k < inflight; ++k) {
            slot = (slot + 1) % nC;
            if (slot == 0) slot = 1;
            do_gemm(C[slot]);
            local_iters++;
        }
        // Keep GDDR6 busy with a bulk copy of leftover VRAM.
        if (scratch && leftover >= 2 * (8ull << 20)) {
            size_t half = leftover / 2;
            q.memcpy(scratch + half, scratch, half);
        }
        q.wait();

        if ((local_iters % 32) < size_t(inflight)) {
            compare_results<T>(q, C[0], C[1], elems, faults, eps).wait();
            int f = 0;
            q.memcpy(&f, faults, sizeof(int)).wait();
            if (f) {
                st.errors.store(uint64_t(f));
                q.memset(faults, 0, sizeof(int)).wait();
            }
        }

        auto now = clock_tp::now();
        double sec = std::chrono::duration<double>(now - t0).count();
        // 2*n^3 FLOPs per GEMM
        double flops = double(local_iters) * 2.0 * double(n) * double(n) * double(n);
        st.iters.store(local_iters);
        if (sec > 0.2) st.tflops.store((flops / sec) / 1e12);
    }

    sycl::free(A, q);
    sycl::free(B, q);
    for (auto* p : C) sycl::free(p, q);
    if (scratch) sycl::free(scratch, q);
    sycl::free(faults, q);
}

#ifdef USE_ONEMKL
using bf16 = oneapi::mkl::bfloat16;

void burn_bf16(sycl::queue& q, const Options& opt, GpuStatus& st, size_t budget) {
    const int n = opt.matrix_n;
    const size_t elems = size_t(n) * size_t(n);
    const size_t mat_bytes = elems * sizeof(bf16);
    if (budget < mat_bytes * 3) return;
    size_t nC = (budget - 2 * mat_bytes) / mat_bytes;
    if (nC < 2) nC = 2;
    if (nC > 512) nC = 512;
    size_t used = mat_bytes * (2 + nC);
    size_t leftover = budget > used + (8ull << 20) ? budget - used : 0;
    leftover &= ~size_t(4095);
    st.used_bytes.store(used + leftover);
    st.n.store(n);

    bf16* A = sycl::malloc_device<bf16>(elems, q);
    bf16* B = sycl::malloc_device<bf16>(elems, q);
    std::vector<bf16*> C(nC);
    for (size_t i = 0; i < nC; ++i) C[i] = sycl::malloc_device<bf16>(elems, q);
    int* faults = sycl::malloc_device<int>(1, q);
    char* scratch = leftover ? sycl::malloc_device<char>(leftover, q) : nullptr;

    std::vector<float> hAf(elems), hBf(elems);
    fill_host(hAf, 0xA5A5u);
    fill_host(hBf, 0x5A5Au);
    std::vector<bf16> hA(elems), hB(elems);
    for (size_t i = 0; i < elems; ++i) {
        hA[i] = bf16(hAf[i]);
        hB[i] = bf16(hBf[i]);
    }
    q.memcpy(A, hA.data(), mat_bytes).wait();
    q.memcpy(B, hB.data(), mat_bytes).wait();
    q.memset(faults, 0, sizeof(int)).wait();
    if (scratch) q.memset(scratch, 0x5A, leftover).wait();

    const bf16 alpha = bf16(1.0f);
    const bf16 beta = bf16(0.0f);

    auto t0 = clock_tp::now();
    uint64_t local_iters = 0;
    size_t slot = 0;
    mkl_gemm<bf16>(q, n, alpha, A, B, beta, C[0], {}).wait();
    local_iters++;

    const int inflight = 8;
    while (!g_stop.load()) {
        for (int k = 0; k < inflight; ++k) {
            slot = (slot + 1) % nC;
            if (slot == 0) slot = 1;
            mkl_gemm<bf16>(q, n, alpha, A, B, beta, C[slot], {});
            local_iters++;
        }
        if (scratch && leftover >= 2 * (8ull << 20)) {
            size_t half = leftover / 2;
            q.memcpy(scratch + half, scratch, half);
        }
        q.wait();
        if ((local_iters % 32) < size_t(inflight)) {
            compare_results<bf16>(q, C[0], C[1], elems, faults, bf16(0.25f)).wait();
            int f = 0;
            q.memcpy(&f, faults, sizeof(int)).wait();
            if (f) {
                st.errors.store(uint64_t(f));
                q.memset(faults, 0, sizeof(int)).wait();
            }
        }
        auto now = clock_tp::now();
        double sec = std::chrono::duration<double>(now - t0).count();
        double flops = double(local_iters) * 2.0 * double(n) * double(n) * double(n);
        st.iters.store(local_iters);
        if (sec > 0.2) st.tflops.store((flops / sec) / 1e12);
    }
    sycl::free(A, q);
    sycl::free(B, q);
    for (auto* p : C) sycl::free(p, q);
    if (scratch) sycl::free(scratch, q);
    sycl::free(faults, q);
}
#endif

static void burn_one(sycl::device dev, const Options& opt, GpuStatus& st) {
    try {
        sycl::queue q(dev, sycl::property_list{sycl::property::queue::in_order()});
        size_t global = dev.get_info<sycl::info::device::global_mem_size>();
        size_t budget = opt.mem_mb > 0 ? size_t(opt.mem_mb) * 1024ull * 1024ull
                                       : size_t(opt.mem_frac * double(global));
        // Leave a little headroom for runtime / kernels
        if (budget > global - (128ull << 20) && global > (128ull << 20))
            budget = global - (128ull << 20);

        st.alive.store(1);
        switch (opt.prec) {
            case Prec::FP64:
                burn_typed<double>(q, opt, st, budget);
                break;
            case Prec::BF16:
#ifdef USE_ONEMKL
                burn_bf16(q, opt, st, budget);
#else
                std::cerr << "BF16/XMX requested but xgpu-burn was built without oneMKL; "
                             "falling back to FP32 tiled GEMM\n";
                burn_typed<float>(q, opt, st, budget);
#endif
                break;
            default:
                burn_typed<float>(q, opt, st, budget);
                break;
        }
        st.alive.store(0);
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error on " << pci_hint(dev) << ": " << e.what() << "\n";
        st.alive.store(0);
    } catch (const std::exception& e) {
        std::cerr << "Error on " << pci_hint(dev) << ": " << e.what() << "\n";
        st.alive.store(0);
    }
}

static void print_banner(const Options& opt, const std::vector<sycl::device>& devs) {
    std::cout << "==============================================================\n";
    std::cout << " xgpu-burn  Intel Arc / Xe GPU burn-in\n";
    std::cout << "==============================================================\n";
#ifdef USE_ONEMKL
    std::cout << " GEMM backend : oneMKL (XMX-capable)\n";
#else
    std::cout << " GEMM backend : tiled SYCL (rebuild with USE_ONEMKL=1 for peak)\n";
#endif
    std::cout << " Precision    : " << prec_name(opt.prec) << "\n";
    std::cout << " Matrix       : " << opt.matrix_n << " x " << opt.matrix_n << "\n";
    std::cout << " Duration     : " << opt.seconds << " s\n";
    std::cout << " Devices      : " << devs.size() << "\n";
    for (size_t i = 0; i < devs.size(); ++i) {
        auto& d = devs[i];
        double gb = d.get_info<sycl::info::device::global_mem_size>() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "   [" << i << "] " << d.get_info<sycl::info::device::name>()
                  << "  " << std::fixed << std::setprecision(1) << gb << " GiB  backend="
                  << backend_name(d.get_backend())
                  << "  CUs=" << d.get_info<sycl::info::device::max_compute_units>()
                  << "\n";
    }
    std::cout << " B60 peak (ref): 12.28 TFLOPS FP32, 200 W TBP, 24 GB, 2400 MHz\n";
    std::cout << " 8x B60 budget : ~98 TFLOPS FP32, ~1600 W GPU TBP\n";
    std::cout << "==============================================================\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Options opt = parse_args(argc, argv);
    auto all = intel_gpus();
    if (all.empty()) {
        std::cerr << "No SYCL GPU devices found.\n"
                  << "Check:  source /opt/intel/oneapi/setvars.sh && sycl-ls\n"
                  << "        ls /dev/dri/renderD*\n";
        return 1;
    }

    if (opt.list_only) {
        std::cout << "Found " << all.size() << " GPU(s):\n";
        for (size_t i = 0; i < all.size(); ++i) {
            auto& d = all[i];
            double gb = d.get_info<sycl::info::device::global_mem_size>() /
                        (1024.0 * 1024.0 * 1024.0);
            std::cout << "  [" << i << "] " << d.get_info<sycl::info::device::name>()
                      << "  " << std::fixed << std::setprecision(2) << gb << " GiB"
                      << "  backend=" << backend_name(d.get_backend())
                      << "  CUs=" << d.get_info<sycl::info::device::max_compute_units>()
                      << "\n";
        }
        auto hw = scan_hwmon();
        if (!hw.empty()) {
            std::cout << "\nSysfs cards:\n";
            for (auto& c : hw) {
                auto t = read_telemetry(c);
                std::cout << "  " << c.drm << "  pci=" << c.pci
                          << "  id=" << c.device_id
                          << "  hwmon=" << (c.hwmon.empty() ? "(none)" : c.hwmon)
                          << "  T=" << t.temp_c << "C  P=" << t.power_w << "W"
                          << "  f=" << t.freq_mhz << "MHz\n";
            }
        }
        return 0;
    }

    std::vector<sycl::device> devs;
    if (opt.device_id >= 0) {
        if (opt.device_id >= int(all.size())) {
            std::cerr << "GPU index " << opt.device_id << " out of range\n";
            return 1;
        }
        devs.push_back(all[size_t(opt.device_id)]);
    } else {
        devs = all;
    }

    print_banner(opt, devs);

    std::vector<GpuStatus> status(devs.size());
    for (size_t i = 0; i < devs.size(); ++i)
        status[i].name = devs[i].get_info<sycl::info::device::name>();

    auto hw = scan_hwmon();

    std::vector<std::thread> workers;
    workers.reserve(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        workers.emplace_back(burn_one, devs[i], opt, std::ref(status[i]));
    }

    const auto start = clock_tp::now();
    double peak_power = 0;
    double peak_temp = 0;

    while (!g_stop.load()) {
        auto now = clock_tp::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= opt.seconds) {
            g_stop.store(true);
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
        elapsed = std::chrono::duration<double>(clock_tp::now() - start).count();

        std::ostringstream line;
        line << "\n[" << std::setw(5) << int(elapsed) << "s / " << opt.seconds << "s]  "
             << prec_name(opt.prec) << "\n";
        double sum_w = 0, sum_tf = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            Telemetry t;
            if (i < hw.size()) t = read_telemetry(hw[i]);
            if (t.power_w > 0) sum_w += t.power_w;
            if (t.power_w > peak_power) peak_power = t.power_w;
            if (t.temp_c > peak_temp) peak_temp = t.temp_c;
            sum_tf += status[i].tflops.load();

            const double used_gb = status[i].used_bytes.load() / (1024.0 * 1024.0 * 1024.0);
            line << "  GPU " << i << "  "
                 << std::setw(6) << std::fixed << std::setprecision(1)
                 << status[i].tflops.load() << " TFLOPS  "
                 << "iters=" << std::setw(6) << status[i].iters.load() << "  "
                 << "err=" << status[i].errors.load() << "  "
                 << std::setprecision(1) << used_gb << "GiB";
            if (t.temp_c > 0)
                line << "  " << std::setprecision(0) << t.temp_c << "C";
            if (t.mem_temp_c > 0)
                line << "/" << t.mem_temp_c << "Cmem";
            if (t.power_w > 0)
                line << "  " << std::setprecision(0) << t.power_w << "W";
            if (t.freq_mhz > 0)
                line << "  " << t.freq_mhz << "MHz";
            line << (status[i].errors.load() ? "  FAULT" : "  OK") << "\n";
        }
        line << "  TOTAL  " << std::setprecision(1) << sum_tf << " TFLOPS";
        if (sum_w > 0) line << "  " << std::setprecision(0) << sum_w << "W";
        line << "  peakGPU " << peak_temp << "C / " << std::setprecision(0) << peak_power << "W\n";
        std::cout << line.str() << std::flush;
    }

    g_stop.store(true);
    for (auto& th : workers) th.join();

    std::cout << "\n============== SUMMARY ==============\n";
    uint64_t tot_err = 0, tot_it = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        tot_err += status[i].errors.load();
        tot_it += status[i].iters.load();
        std::cout << " GPU " << i << "  iters=" << status[i].iters.load()
                  << "  errors=" << status[i].errors.load()
                  << (status[i].errors.load() ? "  FAIL" : "  PASS") << "\n";
    }
    std::cout << " All GPUs    iters=" << tot_it << "  errors=" << tot_err << "\n";
    std::cout << " Peak (sysfs) temp=" << peak_temp << "C  per-GPU power=" << peak_power << "W\n";
    std::cout << (tot_err ? " RESULT: FAIL (compute mismatches)\n"
                         : " RESULT: PASS\n");
    return tot_err ? 3 : 0;
}
