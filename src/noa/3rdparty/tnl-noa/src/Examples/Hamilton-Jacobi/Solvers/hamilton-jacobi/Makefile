TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = hamiltonJacobiSolver
CONFIG_FILE = $(TARGET).cfg.desc
INSTALL_DIR = ${HOME}/local
CXX = g++
CUDA_CXX = nvcc
OMP_FLAGS = -DHAVE_OPENMP -fopenmp
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR) -O3 $(OMP_FLAGS)
LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-0.1 -lgomp

SOURCES = main.cpp
HEADERS = 
OBJECTS = main.o
DIST = $(SOURCES) Makefile

all: $(TARGET)
clean: 
	rm -f $(OBJECTS)
	rm -f $(TARGET)-conf.h	

dist: $(DIST)
	tar zcvf $(TARGET).tgz $(DIST) 

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/bin
	cp $(CONFIG_FILE) $(INSTALL_DIR)/share

uninstall: $(TARGET)
	rm -f $(INSTALL_DIR)/bin/$(TARGET) 
	rm -f $(CONFIG_FILE) $(INSTALL_DIR)/share

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $(CXX_FLAGS) $<


