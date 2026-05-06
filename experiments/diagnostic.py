import subprocess

def check_gpus():
    try:
        # This is the exact command the script uses
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('utf-8')
        print("--- RAW OUTPUT FROM NVIDIA-SMI ---")
        print(output)
        print("----------------------------------")
        
        free_gpus = []
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            idx, free_mem = line.split(',')
            # Check if it sees the 80GB (80000 MB) correctly
            print(f"GPU {idx}: {free_mem} MiB free")
            if int(free_mem) > 10000:
                free_gpus.append(idx.strip())
                
        print(f"\nSCRIPT DETECTED: {free_gpus}")
    except Exception as e:
        print(f"Error: {e}")

check_gpus()