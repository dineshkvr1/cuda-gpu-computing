# CUDA Testing

This folder contains CUDA programs for testing GPU computation capabilities.

## Files

- `vector_add.cu` - A simple CUDA program that performs vector addition on the GPU
- `Makefile` - Build configuration for compiling CUDA programs
- `README.md` - This file

## Requirements

- NVIDIA GPU with CUDA capability
- CUDA toolkit installed (detected: CUDA 12.2)
- NVIDIA drivers (detected: 535.230.02)

## Hardware Detected

- GPU: Tesla V100-PCIE-16GB
- Memory: 16384 MiB
- CUDA Version: 12.2

## Usage

### Compile the program:
```bash
make
```

### Run the program:
```bash
make run
# or directly:
./vector_add
```

### Clean compiled files:
```bash
make clean
```

### Check CUDA installation:
```bash
make check-cuda
```

## Program Description

The `vector_add.cu` program demonstrates:
- Memory allocation on GPU
- Data transfer between host (CPU) and device (GPU)
- CUDA kernel execution with proper grid and block configuration
- Error checking for CUDA operations
- Result verification

The program adds two vectors of 1024 elements each using GPU parallel processing.
