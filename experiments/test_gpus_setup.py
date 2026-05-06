import os
import subprocess
import torch

def get_best_gpu():
    """Returns the index of the GPU with the most free memory."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('utf-8')
        lines = output.strip().split('\n')
        # Sort by free memory (descending)
        lines.sort(key=lambda x: int(x.split(',')[1]), reverse=True)
        # Return the ID of the winner
        return lines[0].split(',')[0] 
    except:
        return "0"

# 1. Pick the winner
best_gpu_id = get_best_gpu()
print(f"Auto-selected GPU ID: {best_gpu_id}")

# 2. Isolate it (Make PyTorch think this is the ONLY GPU that exists)
os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu_id

# 3. Normal PyTorch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")