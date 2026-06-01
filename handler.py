import os
import sys
import json
import subprocess
import runpod

def handler(job):
    # Extracts the target N value from the incoming payload (defaults to 12)
    job_input = job.get("input", {})
    n_value = job_input.get("N", 12)
    
    try:
        # Fire your high-performance compiled C++ CUDA binary
        process = subprocess.run(
            ["./gpu_server", str(n_value)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the C++ stdout string output directly back into a JSON object
        result_json = json.loads(process.stdout)
        return result_json

    except subprocess.CalledProcessError as e:
        return {"error": f"CUDA Kernel Execution Fault: {e.stderr}"}
    except Exception as e:
        return {"error": f"Handler execution error: {str(e)}"}

# --- DYNAMIC HYBRID ENVIRONMENT DETECTOR ---
if __name__ == "__main__":
    # Check if we are running locally on Docker Desktop
    if not os.environ.get("RUNPOD_POD_ID"):
        print("\n=== 🖥️ LOCAL DOCKER DESKTOP ENVIRONMENT DETECTED ===")
        
        # Parse command line arguments if passed via "docker run"
        # sys.argv[0] is the script name, sys.argv[1] would be the user input
        if len(sys.argv) > 1:
            try:
                target_n = int(sys.argv[1])
            except ValueError:
                print(f"⚠️ Warning: '{sys.argv[1]}' is not a valid integer. Falling back to N=12.")
                target_n = 12
        else:
            print("💡 Tip: You can pass N as an argument. Example: docker run ... image_name 14")
            target_n = 12 # Default fallback
        
        # Define the dynamic local mock job payload
        mock_job = {"input": {"N": target_n}}
        
        print(f"Executing N-Queen GPU solver for N = {mock_job['input']['N']}...\n")
        local_result = handler(mock_job)
        
        print("=== 📋 SYSTEM RESPONSE PAYLOAD ===")
        print(json.dumps(local_result, indent=2))
        print("================================================\n")
    else:
        # RunPod cloud engine execution line
        runpod.serverless.start({"handler": handler})