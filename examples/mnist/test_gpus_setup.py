# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Must be before importing torch

# import torch
# print(torch.cuda.get_device_name(0)) # This will now show "GPU 0" but physically it is GPU 1


# import torch

# print(f"{'ID':<3} | {'GPU Name':<25} | {'Memory Usage (Used/Total)':<30}")
# print("-" * 60)

# for i in range(torch.cuda.device_count()):
#     # mem_get_info() returns (free, total) in bytes
#     free_mem, total_mem = torch.cuda.mem_get_info(i)
    
#     used_mem = total_mem - free_mem
    
#     # Convert to GB
#     total_gb = total_mem / (1024**3)
#     used_gb = used_mem / (1024**3)
    
#     print(f"{i:<3} | {torch.cuda.get_device_name(i):<25} | {used_gb:5.1f}GB / {total_gb:5.1f}GB")


# import GPUtil

# # Get all GPUs
# gpus = GPUtil.getGPUs()

# print(f"{'ID':<3} | {'Load':<6} | {'Memory Used':<12} | {'Total':<10}")
# for gpu in gpus:
#     print(f"{gpu.id:<3} | {gpu.load*100:<3.0f}%  | {gpu.memoryUsed:<5.0f} MB    | {gpu.memoryTotal:<5.0f} MB")

# # BONUS: Automatically give me the ID of the first free GPU
# first_free_id = GPUtil.getFirstAvailable(order='memory', maxLoad=0.1, maxMemory=0.1, attempts=1, interval=900, verbose=False)
# print(f"\nRecommended GPU ID: {first_free_id}")



# import os
# import subprocess
# import torch
# import torch.nn as nn

# def get_best_gpus(n=3):
#     """
#     Returns a list of the top 'n' GPU indices with the most free memory.
#     """
#     try:
#         # Query nvidia-smi for index and free memory (in MiB)
#         cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
#         output = subprocess.check_output(cmd.split()).decode('utf-8')
        
#         # Parse output: create a list of (index, free_memory) tuples
#         gpu_info = []
#         for line in output.strip().split('\n'):
#             idx, free_mem = line.split(',')
#             gpu_info.append((int(idx), int(free_mem)))
            
#         # Sort by free memory (descending) and take top n
#         gpu_info.sort(key=lambda x: x[1], reverse=True)
#         best_gpus = [str(x[0]) for x in gpu_info[:n]]
        
#         print(f"Selected GPUs: {best_gpus} (Free Mem: {[f'{x[1]} MiB' for x in gpu_info[:n]]})")
#         return best_gpus
        
#     except Exception as e:
#         print(f"Error auto-selecting GPUs: {e}")
#         return ["0"] # Fallback to GPU 0

# # --- CRITICAL SECTION ---
# # 1. Get the best 3 GPUs
# top_gpus = get_best_gpus(3)

# # 2. Set the environment variable to make ONLY these GPUs visible
# # This maps them to indices 0, 1, 2 inside PyTorch
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(top_gpus)

# # 3. NOW import PyTorch (must be after setting the env var)
# import torch
# print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# # Standard PyTorch setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"PyTorch sees {torch.cuda.device_count()} GPUs.")
# # Output should be 3 (even if your server has 8)



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