# xgpu-burn — Intel Arc / Xe GPU_Burn equivalent
#
# Build (preferred, oneMKL + XMX):
#   source /opt/intel/oneapi/setvars.sh
#   make
#
# Build without oneMKL (tiled SYCL fallback):
#   source /opt/intel/oneapi/setvars.sh
#   make NOMKL=1
#
# Requires: Intel oneAPI DPC++ (icpx) and a working Level Zero stack.

CXX      ?= icpx
CXXFLAGS ?= -fsycl -O3 -std=c++17 -Wall
LDFLAGS  ?=
TARGET   := xgpu_burn
SRC      := src/xgpu_burn.cpp

ifeq ($(NOMKL),1)
  CPPFLAGS +=
else
  CPPFLAGS += -DUSE_ONEMKL
  # oneAPI setvars.sh normally provides these. Explicit flags keep
  # headless server builds reproducible.
  CXXFLAGS += -qmkl=sequential
  LDFLAGS  += -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
endif

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

check:
	@command -v $(CXX) >/dev/null || { echo "icpx not found. source /opt/intel/oneapi/setvars.sh"; exit 1; }
	@echo "compiler: $$($(CXX) --version | head -1)"
	@command -v sycl-ls >/dev/null && sycl-ls || echo "sycl-ls not in PATH"
	@ls /dev/dri/renderD* 2>/dev/null || echo "no render nodes"
