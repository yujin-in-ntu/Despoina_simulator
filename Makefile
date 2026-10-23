# SPDX-License-Identifier: Apache-2.0
CXX ?= c++
ROOT := $(CURDIR)
BUILD := $(ROOT)/build
SIM := $(ROOT)/src/NeuraSim
LIB := $(SIM)/src/lib
DRAMSIM := $(ROOT)/third_party/DRAMsim3-1.0.0
UNAME_S := $(shell uname -s)

CPPFLAGS := -I$(LIB) -I$(DRAMSIM)/src -I$(DRAMSIM)/ext/fmt/include -I$(DRAMSIM)/ext/headers
CXXFLAGS ?= -std=c++11 -O2 -Wall
CXXFLAGS += -DFMT_HEADER_ONLY=1 -DENABLE_CIM -DENABLE_PE_ARRAY \
	-DENABLE_ACTIVATION -DENABLE_HAGG_BUFFER -DENABLE_DRAMSIM3_PE \
	-DUSE_PE_ARRAY_MODE -DPE_L1_MRU_ENTRIES=16 -DPE_CLUSTER_OUTPUT_WRITE_LATENCY=50 \
	-DPE_CLUSTER_A_LOAD_LATENCY=50 -DPE_MSHR_SIZE=64 \
	-DPE_OUTPUT_EGRESS_BYTES_PER_CYCLE=512 -DPE_CLUSTER_OUTPUT_QUEUE_DEPTH=8 \
	-DPE_SHARED_F_CACHE_ENTRIES=4096 -DPE_SHARED_F_CACHE_BANKS=8 \
	-DPE_PHASE3_KEEP_SHARED_F_CACHE=1 -DINDEX_STORE_MAX_ENTRIES=4096 \
	-DPE_NUM_CLUSTERS=64 -DDESPOINA_V3_CACHE_EPOCH=1 \
	-DDESPOINA_V3_FUNCTIONAL_DCIM=1
LDLIBS := $(BUILD)/third_party/libdramsim3.a -lboost_program_options

ifeq ($(UNAME_S),Darwin)
BOOST_PREFIX := $(shell brew --prefix boost)
OMP_PREFIX := $(shell brew --prefix libomp)
CPPFLAGS += -I$(BOOST_PREFIX)/include -I$(OMP_PREFIX)/include
CXXFLAGS += -Xpreprocessor -fopenmp
LDFLAGS += -L$(BOOST_PREFIX)/lib -L$(OMP_PREFIX)/lib -Wl,-rpath,$(BOOST_PREFIX)/lib
LDLIBS += -lomp
else
CXXFLAGS += -fopenmp
LDLIBS += -fopenmp
endif

SOURCES := \
	$(SIM)/src/code/main.cpp \
	$(LIB)/functions.cpp \
	$(LIB)/debugger/debugger.cpp \
	$(LIB)/neuracore/neuracore.cpp \
	$(LIB)/neuramem/neuramem.cpp \
	$(LIB)/neurarouter/neurarouter.cpp \
	$(LIB)/memreq/memrequest.cpp \
	$(LIB)/memresp/memresponse.cpp \
	$(LIB)/counters/globalcounters.cpp \
	$(LIB)/neuratile/neuratile.cpp \
	$(LIB)/connector/conn.cpp \
	$(LIB)/instx/instx.cpp \
	$(LIB)/dispatcher/dispatcher.cpp \
	$(LIB)/hashreq/hashrequest.cpp \
	$(LIB)/hashresp/hashresponse.cpp \
	$(LIB)/writer/writer.cpp \
	$(LIB)/mc/memorycontroller.cpp \
	$(LIB)/hashpad/hashpad.cpp \
	$(LIB)/hashline/hashline.cpp \
	$(LIB)/cache/cache.cpp \
	$(LIB)/cacheline/cacheline.cpp \
	$(LIB)/result_sink/result_sink.cpp \
	$(LIB)/argparse/argparse.cpp \
	$(LIB)/neuracim/neuracim.cpp \
	$(LIB)/pe/pe.cpp \
	$(LIB)/pe/pe_cluster.cpp \
	$(LIB)/pe/work_stealing_scheduler.cpp \
	$(LIB)/pe/pe_array_spmm.cpp \
	$(LIB)/pe/delta_f_aware_indexer.cpp \
	$(LIB)/pe/delta_converter.cpp \
	$(LIB)/pe/dual_mode_spmm.cpp \
	$(LIB)/activation/activation.cpp \
	$(LIB)/buffer/sram_buffer.cpp
OBJECTS := $(patsubst $(ROOT)/%.cpp,$(BUILD)/obj/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)
DUAL_MODE_OBJECT := $(BUILD)/obj/src/NeuraSim/src/lib/pe/dual_mode_spmm.o
FAULT_DUAL_MODE_OBJECT := $(BUILD)/fault-obj/src/NeuraSim/src/lib/pe/dual_mode_spmm.o
FAULT_OBJECTS := $(filter-out $(DUAL_MODE_OBJECT),$(OBJECTS)) $(FAULT_DUAL_MODE_OBJECT)
FAULT_DEPS := $(FAULT_DUAL_MODE_OBJECT:.o=.d)
UNIT_FLAGS := -std=c++17 -O0 -Wall -I$(LIB) -I$(SIM)/src

.PHONY: all build fault-build test smoke verify maintainer-units release-static \
	fault-smoke check-public-scope package package-hygiene maintainer-check clean
all: build
build: $(BUILD)/despoina_sim
fault-build: $(BUILD)/fault/despoina_sim_fault

$(BUILD)/third_party/libdramsim3.a:
	cmake -S $(DRAMSIM) -B $(BUILD)/third_party -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD)/third_party --target dramsim3 --parallel 2

$(BUILD)/obj/%.o: $(ROOT)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/fault-obj/%.o: $(ROOT)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DDESPOINA_ENABLE_TEST_FAULT_INJECTION=1 \
		-MMD -MP -c $< -o $@

$(BUILD)/despoina_sim: $(BUILD)/third_party/libdramsim3.a $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD)/fault/despoina_sim_fault: $(BUILD)/third_party/libdramsim3.a $(FAULT_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $(FAULT_OBJECTS) $(LDLIBS)

$(BUILD)/unit/a_load_burst_model_unit: $(SIM)/tests/a_load_burst_model_unit.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(UNIT_FLAGS) $< -o $@

$(BUILD)/unit/dynamic_row_mean_operator_unit: $(SIM)/tests/dynamic_row_mean_operator_unit.cpp \
		$(LIB)/pe/delta_converter.cpp $(LIB)/pe/delta_f_aware_indexer.cpp \
		$(BUILD)/third_party/libdramsim3.a
	@mkdir -p $(dir $@)
	$(CXX) $(UNIT_FLAGS) $(CPPFLAGS) $^ -o $@

$(BUILD)/unit/hagg_staging_policy_test: $(SIM)/tests/hagg_staging_policy_test.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(UNIT_FLAGS) $< -o $@

$(BUILD)/unit/official_contract_unit: $(SIM)/tests/official_contract_unit.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(UNIT_FLAGS) $< -o $@

test: build $(BUILD)/unit/official_contract_unit \
	$(BUILD)/unit/dynamic_row_mean_operator_unit
	$(BUILD)/unit/official_contract_unit
	$(BUILD)/unit/dynamic_row_mean_operator_unit
	python3 tests/test_cli_contract.py

smoke: build
	@python3 -c 'import shutil; shutil.rmtree("build/smoke", ignore_errors=True)'
	python3 scripts/run_synthetic_smoke.py --binary $(BUILD)/despoina_sim --output $(BUILD)/smoke

verify: test smoke

maintainer-units: $(BUILD)/unit/a_load_burst_model_unit \
	$(BUILD)/unit/hagg_staging_policy_test
	$(BUILD)/unit/a_load_burst_model_unit
	$(BUILD)/unit/hagg_staging_policy_test

release-static: build
	python3 tests/test_release.py

fault-smoke: build fault-build
	python3 scripts/check_fault_binary_isolation.py \
		--normal $(BUILD)/despoina_sim --fault $(BUILD)/fault/despoina_sim_fault
	@python3 -c 'import shutil; shutil.rmtree("build/fault-smoke", ignore_errors=True)'
	DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT=1 \
		python3 scripts/run_synthetic_smoke.py --binary $(BUILD)/despoina_sim \
		--output $(BUILD)/fault-smoke/normal-isolation
	python3 scripts/run_synthetic_smoke.py --binary $(BUILD)/fault/despoina_sim_fault \
		--output $(BUILD)/fault-smoke/fault-negative --negative-only

check-public-scope:
	python3 scripts/check_public_scope.py

package: check-public-scope
	python3 scripts/package.py

package-hygiene: package
	python3 scripts/check_package_hygiene.py

maintainer-check: maintainer-units release-static check-public-scope fault-smoke \
	package-hygiene

clean:
	python3 -c 'import shutil; shutil.rmtree("build", ignore_errors=True); shutil.rmtree("dist", ignore_errors=True)'

-include $(DEPS) $(FAULT_DEPS)
