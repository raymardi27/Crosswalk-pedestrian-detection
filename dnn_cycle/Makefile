# Specify the path to your CUDA installation
CUDA_PATH ?= /usr/local/cuda

# Compiler
CXX := g++

# NVCC Compiler from CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc

# Include directories for CUDA
INCLUDES := -I$(CUDA_PATH)/include $(shell pkg-config --cflags opencv4)

# Library paths for CUDA and OpenCV
LIBRARIES := -L$(CUDA_PATH)/lib64 $(shell pkg-config --libs opencv4)

# Compiler flags
CXXFLAGS := -std=c++11 -Wall
NVCCFLAGS := -std=c++11 -Xcompiler -Wall

# Determine architecture-specific flags
SMS ?= 50 52 61 75 # Example architectures, specify your target architectures
GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

# Target executable
TARGET := dnn_cycle

all: build

build: $(TARGET)

$(TARGET): dnn_cycle.o
	$(CXX) $^ $(LIBRARIES) -o $@

dnn_cycle.o: dnn_cycle.cpp
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

run: build
	./$(TARGET)

.PHONY: all build clean run

