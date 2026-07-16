-include Make.user

HELP_GOALS := help
.DEFAULT_GOAL := all

ifeq ($(filter $(HELP_GOALS),$(MAKECMDGOALS)),)
ifndef MFEM_DIR
$(error MFEM_DIR is not set. Please create a Make.user file to set MFEM_DIR)
endif
endif

ifdef MFEM_DIR
MFEM_BUILD_DIR = $(MFEM_DIR)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
-include $(CONFIG_MK)
endif

BUILD_DIR = build
DEPS = $(BUILD_DIR)/deps.d
SOURCES = $(wildcard src/*.cpp)
EXE_NAMES = $(SOURCES:src/%.cpp=%)
EXES = $(EXE_NAMES:%=$(BUILD_DIR)/%)

CXXFLAGS = $(MFEM_FLAGS)
LDLIBS = $(filter-out -lmetis,$(MFEM_LIBS)) -lmetis

.PHONY: help all clean deps run-smoke run-smoke-mpi run-smoke-3d

help:
	@printf '%s\n' \
	  "MFEM_PhaseField make targets" \
	  "" \
	  "Targets:" \
	  "  make                         Build all executables." \
	  "  make all                     Build all executables." \
	  "  make build/<name>            Build one executable, e.g. build/ac2d_conservative_wetting." \
	  "  make run-smoke               Build and run a short 2D AC wetting smoke test." \
	  "  make run-smoke-mpi           Build and run the same smoke test on 2 MPI ranks." \
	  "  make run-smoke-3d            Build and run a short 3D AC wetting smoke test." \
	  "  make deps                    Regenerate source dependency file." \
	  "  make clean                   Remove build outputs and smoke-test output." \
	  "" \
	  "Configuration:" \
	  "  MFEM_DIR                     Set in Make.user; must point to an MFEM build/source tree." \
	  "" \
	  "Example:" \
	  "  mpirun -np 12 ./build/ac2d_conservative_wetting -theta 45" \
	  "  mpirun -np 12 ./build/ch2d_conservative_wetting -theta 45" \
	  "  mpirun -np 12 ./build/ac3d_conservative_wetting -theta 45"

all: $(EXES)

$(BUILD_DIR)/%: src/%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | $(BUILD_DIR)
	$(MFEM_CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

BUILD_DEPS = $(MFEM_CXX) $(CXXFLAGS) -M $(SOURCES) \
	| sed -E 's@^(.*\.o: *)(.*/)?(.*\.cpp)@build/\2\1\2\3@' > $(DEPS)

$(DEPS): | $(BUILD_DIR)
	$(BUILD_DEPS)

deps:
	$(BUILD_DEPS)

ifeq ($(filter $(HELP_GOALS),$(MAKECMDGOALS)),)
-include $(DEPS)
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run-smoke: $(BUILD_DIR)/ac2d_conservative_wetting
	rm -rf Saves/smoke_theta45
	./$(BUILD_DIR)/ac2d_conservative_wetting \
	  -nx 40 -ny 20 -delta 0.05 -dt 0.05 -tf 0.05 \
	  -theta 45 -no-vis -out Saves/smoke_theta45
	cat Saves/smoke_theta45/time_specialized.csv

run-smoke-mpi: $(BUILD_DIR)/ac2d_conservative_wetting
	rm -rf Saves/smoke_theta45_np2
	mpirun -np 2 ./$(BUILD_DIR)/ac2d_conservative_wetting \
	  -nx 40 -ny 20 -delta 0.05 -dt 0.05 -tf 0.05 \
	  -theta 45 -no-vis -out Saves/smoke_theta45_np2
	cat Saves/smoke_theta45_np2/time_specialized.csv

run-smoke-3d: $(BUILD_DIR)/ac3d_conservative_wetting
	rm -rf Saves/smoke3d_theta45_np2
	mpirun -np 2 ./$(BUILD_DIR)/ac3d_conservative_wetting \
	  -nx 10 -ny 10 -nz 10 -delta 0.1 -dt 0.02 -tf 0.02 \
	  -theta 45 -no-vis -out Saves/smoke3d_theta45_np2
	cat Saves/smoke3d_theta45_np2/time_specialized.csv

clean:
	rm -rf $(BUILD_DIR) Saves/smoke_theta45 Saves/smoke_theta45_np2 Saves/smoke3d_theta45_np2
