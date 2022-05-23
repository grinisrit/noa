include config.mk

SOURCES := $(wildcard *.cpp)
TARGETS := $(SOURCES:%.cpp=%)
CUDA_SOURCES := $(wildcard *.cu)
CUDA_TARGETS := $(CUDA_SOURCES:%.cu=%)

## targets definitions follow
.PHONY: all host cuda
all: host cuda
host: $(TARGETS)
cuda: $(CUDA_TARGETS)

.PHONY: clean
clean:
	rm -f *.d *.o *.cuo $(TARGETS) $(CUDA_TARGETS)

$(TARGETS): % : %.o
	$(CXX) $(LDFLAGS) -o $@ $< $(LDLIBS)

# use .cuo instead of .cu.o to avoid problems with the implicit rules: https://stackoverflow.com/q/62967939
# (and use the host compiler for linking CUDA, nvcc does not understand that .cuo is an object file)
$(CUDA_TARGETS): % : %.cuo
	$(CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS)

$(SOURCES:%.cpp=%.o): %.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(CUDA_SOURCES:%.cu=%.cuo): %.cuo : %.cu
	$(CUDA_CXX) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -c -o $@ $<
