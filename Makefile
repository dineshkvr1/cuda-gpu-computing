# Makefile for CUDA programs

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70

# Target executable
TARGET = vector_add

# Source file
SOURCE = vector_add.cu

# Default target
all: $(TARGET)

# Compile CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

# Clean compiled files
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Check CUDA installation and devices
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found"
	@echo "Available CUDA devices:"
	@nvidia-smi || echo "nvidia-smi not found"

.PHONY: all clean run check-cuda
