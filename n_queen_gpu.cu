#include "n_queen_gpu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cmath>

// --- STRUCTS & TYPES ---
struct GpuResult {
    int n_value;
    unsigned long long solutions;
    double time_ms;
};

struct BoardState {
    unsigned int col_mask;
    unsigned int diag1_mask;
    unsigned int diag2_mask;
};

// --- THE GPU KERNEL ---
// Executed in parallel by thousands of CUDA cores across the GPU grid layout.
__device__ void solve_n_queens_cuda_device(int row, int n, unsigned int col_mask, unsigned int diag1_mask, unsigned int diag2_mask, unsigned long long& count) {
    if (row == n) {
        count++;
        return;
    }
    unsigned int target_mask = (1U << n) - 1U;
    unsigned int available_positions = target_mask & ~(col_mask | diag1_mask | diag2_mask);

    while (available_positions > 0) {
        // Isolate the lowest set bit (extract the next available valid position)
        unsigned int position = available_positions & -static_cast<int>(available_positions);
        available_positions -= position;

        // Recurse deeper into the board tree matrix configuration
        solve_n_queens_cuda_device(row + 1, n, col_mask | position, (diag1_mask | position) << 1, (diag2_mask | position) >> 1, count);
    }
}

// Global kernel entry point to map individual device threads to unique starting boards
__global__ void nqueens_deep_kernel(int n, int start_row, const BoardState* d_states, int total_states, unsigned long long* d_global_counts) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= total_states) return;

    // Load the pre-calculated, unique starting configuration assigned to this thread
    BoardState state = d_states[thread_id];
    unsigned long long local_gpu_count = 0;

    // Continue execution deeply parallelized across the physical hardware cores
    solve_n_queens_cuda_device(start_row, n, state.col_mask, state.diag1_mask, state.diag2_mask, local_gpu_count);

    // Save the computed backtracking leaves sum into global device VRAM
    d_global_counts[thread_id] = local_gpu_count;
}

// --- HOST COMPANION SEED ENGINE ---
// Generates valid path roots down to a target row limit to flood the GPU with workload threads
void generate_seeds_cpu(int row, int target_row, int n, unsigned int col, unsigned int d1, unsigned int d2, std::vector<BoardState>& seeds) {
    if (row == target_row) {
        seeds.push_back({ col, d1, d2 });
        return;
    }
    unsigned int target_mask = (1U << n) - 1U;
    unsigned int available = target_mask & ~(col | d1 | d2);
    while (available > 0) {
        unsigned int pos = available & -static_cast<int>(available);
        available -= pos;
        generate_seeds_cpu(row + 1, target_row, n, col | pos, (d1 | pos) << 1, (d2 | pos) >> 1, seeds);
    }
}

// --- DRIVER PIPELINE ---
GpuResult run_nqueen_on_gpu(int n) {
    // Warmup call to load context handles natively 
    cudaFree(0);

    auto start = std::chrono::high_resolution_clock::now();

    // Determine how deep to pre-seed based on workload matrix complexity
    int target_start_row = (n > 12) ? 3 : 2;
    std::vector<BoardState> h_seeds;
    generate_seeds_cpu(0, target_start_row, n, 0, 0, 0, h_seeds);

    int total_states = static_cast<int>(h_seeds.size());

    // Safety check for trivial instances where no valid paths match up to target row
    if (total_states == 0) {
        return { n, 0, 0.0 };
    }

    size_t allocation_size_counts = total_states * sizeof(unsigned long long);
    size_t allocation_size_states = total_states * sizeof(BoardState);

    unsigned long long* d_global_counts = nullptr;
    BoardState* d_states = nullptr;

    // Memory Allocation on GPU Device VRAM
    cudaMalloc(&d_global_counts, allocation_size_counts);
    cudaMalloc(&d_states, allocation_size_states);

    // Copy setup vectors across PCIe lines to Device memory spaces
    cudaMemcpy(d_states, h_seeds.data(), allocation_size_states, cudaMemcpyHostToDevice);
    cudaMemset(d_global_counts, 0, allocation_size_counts);

    // Dynamic grid block mapping to scale pipeline occupancy efficiently
    int threads_per_block = 256;
    int blocks_per_grid = (total_states + threads_per_block - 1) / threads_per_block;

    // Fire the GPU Grid Pipeline!
    nqueens_deep_kernel << <blocks_per_grid, threads_per_block >> > (n, target_start_row, d_states, total_states, d_global_counts);

    // Synchronize to trap execution state checks securely before hitting WDDM TDR limits
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "❌ GPU Kernel Execution Failure on N=" << n << ": " << cudaGetErrorString(err) << "\n";
        cudaFree(d_global_counts);
        cudaFree(d_states);
        return { n, 0, 0.0 };
    }

    // Allocate host storage vector to retrieve the computed thread buffers
    std::vector<unsigned long long> h_counts(total_states, 0);
    cudaMemcpy(h_counts.data(), d_global_counts, allocation_size_counts, cudaMemcpyDeviceToHost);

    // Clean up explicit GPU resource allocations
    cudaFree(d_global_counts);
    cudaFree(d_states);

    // Final Host Aggregation Stage
    unsigned long long total_solutions = 0;
    for (int i = 0; i < total_states; ++i) {
        total_solutions += h_counts[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return { n, total_solutions, elapsed.count() };
}

// --- ROUTER ROUTINE BOUNDARY ---
std::string handle_nqueen_gpu_computation(int n) {
    std::vector<int> workloads = { n };
    std::vector<GpuResult> results;
    double combined_gpu_execution_time = 0.0;

    for (int n : workloads) {
        GpuResult res = run_nqueen_on_gpu(n);
        results.push_back(res);
        combined_gpu_execution_time += res.time_ms;
    }

    // --- DYNAMIC HARDWARE DETECTION ENGINE ---
    std::string hardware_name = "Generic CUDA-Accelerated GPU Hardware";
    int device_count = 0;
    
    // Check if any CUDA-capable devices exist to prevent initialization crashes
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp device_properties;
        // Interrogate device 0 (the primary active execution GPU node)
        if (cudaGetDeviceProperties(&device_properties, 0) == cudaSuccess) {
            // device_properties.name contains the exact string returned by the NVIDIA driver
            hardware_name = std::string(device_properties.name) + " (CUDA C++)";
        }
    } else {
        hardware_name = "No Active CUDA Device Found / Driver Emulation Mode";
    }

    std::stringstream json;
    json << "{\n"
        << "  \"hardware_accelerator\": \"" << hardware_name << "\",\n"
        << "  \"total_pipeline_gpu_time_ms\": " << combined_gpu_execution_time << ",\n"
        << "  \"computed_benchmarks\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\n"
            << "      \"N\": " << results[i].n_value << ",\n"
            << "      \"distinct_solutions\": " << results[i].solutions << ",\n"
            << "      \"gpu_compute_time_ms\": " << results[i].time_ms << "\n"
            << "    }";
        if (i < results.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";

    return json.str();
}