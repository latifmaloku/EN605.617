
CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++

NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS := -m64 -std=c++17 -arch=sm_120

INCLUDES := -Icommon/UtilNPP
INCLUDES += -Icommon/FreeImage/include

LIBRARIES := -L$(CUDA_PATH)/targets/x86_64-linux/lib
LIBRARIES += -Lcommon/FreeImage/lib
LIBRARIES += -Lcommon/FreeImage/lib/linux
LIBRARIES += -Lcommon/FreeImage/lib/linux/x86_64
LIBRARIES += -lnppidei -lnppial -lnppisu -lnppicc -lnppif -lnppc -lcufft -lfreeimage

all: assignment.exe

assignment.o: assignment.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $< -o $@

assignment.exe: assignment.o
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LIBRARIES)

run: assignment.exe
	./assignment.exe data/Lena.bmp

clean:
	rm -f assignment.exe assignment.o

clobber: clean

.PHONY: all run clean clobber
