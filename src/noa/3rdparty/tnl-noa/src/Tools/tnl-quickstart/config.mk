# configure the include path(s) according to your TNL installation
TNL_INCLUDE_DIRS := -I ~/.local/include

WITH_OPENMP := yes
WITH_DEBUG := no

# If TNL is installed on your system, the CUDA architecture can be detected
# automatically by tnl-cuda-arch. This is done if CUDA_ARCH is set to "auto".
# Otherwise, CUDA_ARCH has to be set manually to the desired CUDA architecture
# number, e.g. 60, 61, etc.
CUDA_ARCH := auto

# compilers
CXX := g++
CUDA_CXX := nvcc

# host compiler flags
CXXFLAGS := -std=c++14 $(TNL_INCLUDE_DIRS)
ifeq ($(WITH_DEBUG),yes)
    CXXFLAGS += -O0 -g
else
    CXXFLAGS += -O3 -DNDEBUG
endif

# CUDA compiler flags
CUDA_CXXFLAGS := -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda $(TNL_INCLUDE_DIRS)
CUDA_CXXFLAGS += -DHAVE_CUDA
ifeq ($(CUDA_ARCH),auto)
    CUDA_CXXFLAGS += $(shell tnl-cuda-arch)
else
    CUDA_CXXFLAGS += -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
endif

# determine path to the CUDA toolkit installation
# (autodetection is attempted, set it manually if it fails)
CUDA_PATH ?= $(abspath $(dir $(shell command -v nvcc))/..)
#$(info Detected CUDA_PATH: $(CUDA_PATH))

# flags for linking CUDA with the host compiler
CUDA_LDFLAGS := -L $(CUDA_PATH)/lib64
CUDA_LDLIBS := -lcudart -ldl -lrt

# enable OpenMP
ifeq ($(WITH_OPENMP),yes)
    CXXFLAGS += -fopenmp -DHAVE_OPENMP
    LDLIBS += -lgomp
    CUDA_CXXFLAGS += -Xcompiler -fopenmp -DHAVE_OPENMP
    CUDA_LDLIBS += -lgomp
endif
