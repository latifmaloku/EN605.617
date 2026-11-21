CXX ?= g++
STD := -std=c++17
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

CXXFLAGS := $(STD) $(OPENCV_CFLAGS) -O2
CPU_OBJECTS := main.o contour_selector.o snake_CPU.o

.PHONY: all cpu clean help

all: cpu

cpu: snake_app

snake_app: $(CPU_OBJECTS)
	$(CXX) $(STD) $^ -o $@ $(OPENCV_LIBS)

main.o: main.cpp snake_base.hpp snake_constants.hpp snake_CPU.hpp contour_selector.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

contour_selector.o: contour_selector.cpp contour_selector.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

snake_CPU.o: snake_CPU.cpp snake_CPU.hpp snake_base.hpp snake_constants.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o snake_app

help:
	@echo "Targets:"
	@echo "  make cpu        Build the CPU executable (snake_app)"
	@echo "  make clean      Remove build outputs"
