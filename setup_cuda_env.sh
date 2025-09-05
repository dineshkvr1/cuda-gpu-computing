#!/bin/bash

# CUDA Environment Setup Script

echo "Setting up CUDA environment..."

# Add CUDA to PATH
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

echo "CUDA paths configured for current session:"
echo "PATH includes: /usr/local/cuda-12.2/bin"
echo "LD_LIBRARY_PATH includes: /usr/local/cuda-12.2/lib64"

# Test CUDA installation
echo ""
echo "Testing CUDA installation..."
nvcc --version

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

echo ""
echo "CUDA environment is ready!"
echo "You can now use 'nvcc' command directly."
