TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = quad-test
CONFIG_FILE = $(TARGET).cfg.desc
INSTALL_DIR = ${HOME}/local
CXX = g++-4.7
CUDA_CXX = nvcc
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR)
LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-0.1 -lbz2

SOURCES = main.cpp Quad.cpp Quadcpp.cpp
HEADERS = Quad.h Quadcpp.h
OBJECTS = main.o Quad.o Quadcpp.o
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

%.o: %.cpp $(TARGET)-conf.h $(HEADERS)
	$(CXX) -c -o $@ $(CXX_FLAGS) $<

$(TARGET)-conf.h:
	echo "#define CONFIG_FILE \"${INSTALL_DIR}/share/${CONFIG_FILE}\" " > $(TARGET)-conf.h 

format:
	astyle --style=java --indent=force-tab --unpad-paren --close-templates $(SOURCES) $(HEADERS)
