# RAD50 isolated, single-executable test runner.
# Uses the actual existing lab implementations compiled into one HIP fat binary.
# No shell-command forwarding, arbitrary executable/plugin loading or generic
# subprocess execution. The IPC case only re-execs its own fixed child mode.
#
# Build: make -C tools/lab/rad42-bar-push OUT_DIR="$BC_WORK_ROOT/builds/rad42-bar-push" rad50-tester
# GPU architecture images: RX 7900 XTX (gfx1100) and R9700 (gfx1201).
# Not an executable permission workaround: obtain allowlist permission for
# rad50-tester once; each new test requires an intentional source/build change.
#
ROCM_PATH ?= /opt/rocm
ifeq ($(strip $(OUT_DIR)),)
$(error Set OUT_DIR outside the source tree; e.g. $$BC_WORK_ROOT/builds/rad42-bar-push)
endif
SRC := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
OBJ := $(OUT_DIR)/obj-rad50
HIPCC ?= hipcc
CXX ?= g++
ARCHES ?= --offload-arch=gfx1100 --offload-arch=gfx1201
# Proof-first build; rebuild with OPT=-O2 before performance evidence.
OPT ?= -O1
COMMON := -std=c++17 $(OPT) -Wall -Wextra -pthread
HSA := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lhsa-runtime64
HIP_OBJECTS := $(OBJ)/serial.o $(OBJ)/ipc.o $(OBJ)/vmem-cap.o $(OBJ)/dmabuf-map.o $(OBJ)/dmabuf-ab.o $(OBJ)/transfer.o $(OBJ)/live-ready.o
CPU_OBJECTS := $(OBJ)/tester.o $(OBJ)/inventory.o $(OBJ)/combine.o $(OBJ)/cpu-handoff.o
OBJECTS := $(CPU_OBJECTS) $(HIP_OBJECTS)

.PHONY: rad50-tester
rad50-tester: $(OUT_DIR)/rad50-tester

$(OBJ) $(OUT_DIR):
	mkdir -p "$@"

$(OBJ)/tester.o: $(SRC)/rad50_tester.cpp $(SRC)/tester_selfcheck.inc | $(OBJ)
	$(CXX) $(COMMON) -c "$<" -o "$@"

$(OBJ)/inventory.o: $(SRC)/pool_inventory.cpp | $(OBJ)
	$(CXX) $(COMMON) -I$(ROCM_PATH)/include -Dmain=rad50_pools_entry -c "$<" -o "$@"

$(OBJ)/combine.o: $(SRC)/combine_reference_test.cpp $(SRC)/../../../engines/radiance/components/moe-remote/combine_f16.h | $(OBJ)
	$(CXX) $(COMMON) -I$(SRC)/../../../engines/radiance/components/moe-remote -Dmain=rad50_combine_entry -c "$<" -o "$@"

$(OBJ)/cpu-handoff.o: $(SRC)/cpu_handoff_bench.cpp | $(OBJ)
	$(CXX) $(COMMON) -Dmain=rad50_cpu_handoff_entry -c "$<" -o "$@"

$(OBJ)/serial.o: $(SRC)/bar_write_canary.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_serial_entry -c "$<" -o "$@"

$(OBJ)/ipc.o: $(SRC)/bar_ipc_canary.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_ipc_entry -c "$<" -o "$@"

$(OBJ)/vmem-cap.o: $(SRC)/bar_vmem_cap.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_vmem_cap_entry -c "$<" -o "$@"

$(OBJ)/dmabuf-map.o: $(SRC)/bar_dmabuf_map.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_dmabuf_map_entry -c "$<" -o "$@"

$(OBJ)/dmabuf-ab.o: $(SRC)/bar_dmabuf_ab.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_dmabuf_ab_entry -c "$<" -o "$@"

$(OBJ)/transfer.o: $(SRC)/bar_transfer_ab.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_transfer_entry -c "$<" -o "$@"

$(OBJ)/live-ready.o: $(SRC)/bar_live_ready_canary.hip $(SRC)/bar_access.h | $(OBJ)
	$(HIPCC) $(COMMON) $(ARCHES) -Dmain=rad50_live_ready_entry -c "$<" -o "$@"

$(OUT_DIR)/rad50-tester: $(OBJECTS) | $(OUT_DIR)
	$(HIPCC) $(COMMON) $(ARCHES) $(OBJECTS) $(HSA) -o "$@"
