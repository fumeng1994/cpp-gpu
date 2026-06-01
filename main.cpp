#include <iostream>
#include <string>
#include <vector>
#include "n_queen_gpu.h"

int main(int argc, char* argv[]) {
    int n = 12; // Default fallback dimension
    
    // Check if the parameter argument was passed via the terminal execution string
    if (argc > 1) {
        try {
            n = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid N integer parameter provided. Falling back to N=12.\n";
        }
    }

    // Call your exact native CUDA calculation orchestration pipeline
    std::string json_output = handle_nqueen_gpu_computation(n);
    
    // Output the resulting text payload straight to stdout for Python to grab
    std::cout << json_output;
    
    return 0;
}