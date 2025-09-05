#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

int main() {
    // Vector size
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    
    // Host vectors
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_a, bytes), "Failed to allocate device memory for vector a");
    checkCudaError(cudaMalloc(&d_b, bytes), "Failed to allocate device memory for vector b");
    checkCudaError(cudaMalloc(&d_c, bytes), "Failed to allocate device memory for vector c");
    
    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy vector a to device");
    checkCudaError(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy vector b to device");
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    std::cout << "Launching CUDA kernel with " << gridSize << " blocks and " << blockSize << " threads per block" << std::endl;
    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Failed to copy result to host");
    
    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition completed successfully!" << std::endl;
        std::cout << "First 10 results:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
        }
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
