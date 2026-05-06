# import csv
# import itertools
# import os
# import re
# import sys
# import subprocess
# import random
# import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from datetime import datetime
# from pathlib import Path
# from multiprocessing import Manager

# # ================= CONFIGURATION =================

# # Path to your training script
# SCRIPT_PATH = Path("experiments/qurift_main.py")

# # SPEED SETTING: How many scripts to run on ONE GPU at the same time?
# # Since MNIST 4-wire circuits are small, you can likely run 6-8 per GPU.
# # If you hit OOM errors, lower this to 4.
# JOBS_PER_GPU = 6 

# # CPU SAFETY: Limit threads per worker.
# CPU_THREADS_PER_WORKER = "2"

# # FIXED ARGUMENTS (Passed to every run)
# BASE_ARGS = [
#     "--model-type", "qnn",
#     "--dataset", "mnist",
#     "--n-wires", "4",               # Fixed for 4-class MNIST
#     "--random-ops", "0",
    
#     # Data Density (The "Overfitting Regime")
#     "--vector-train", "100",
#     "--vector-valid", "100",
#     "--vector-test", "100",
#     "--batch-size", "10",
#     "--epochs", "50",
    
#     # Flags
#     "--train_target",
    
#     # Fixed Q-Layer Topology (Simple is sufficient for structural analysis)
#     "--qlayer-ent-kind", "linear",
# ]

# # === SWEEP RANGES (Targeted for Structural Privacy) ===

# # 1. Feature Map (The Independent Variable)
# FM_KINDS = ["z", "zz", "eff_su2"]

# # 2. Re-uploading Depth (The "Trap" Variable)
# REPS_LIST = [1, 2, 3, 4]

# # 3. Ansatz Depth (To prove depth independence)
# DEPTHS = [2, 4]

# # 4. Ansatz Operation (To test Geometric Resonance)
# QLAYER_OPS = ["cx", "swap"] 

# # Fixed settings for FMs
# PAD_MODE = "wrap"          # Images always wrap
# FM_ENTANGLEMENT = "linear" # Simple entanglement for FMs
# FM_EFF_OP = "cx"           # Standardize eff_su2 op

# # ================= HELPER FUNCTIONS =================

# def get_free_gpus(min_free_mem_mb=8000):
#     """Detects GPUs with at least 'min_free_mem_mb' available."""
#     try:
#         cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
#         output = subprocess.check_output(cmd.split()).decode('utf-8')
#         free_gpus = []
#         for line in output.strip().split('\n'):
#             if not line.strip(): continue
#             idx, free_mem = line.split(',')
#             if int(free_mem) > min_free_mem_mb:
#                 free_gpus.append(idx.strip())
#         return free_gpus
#     except Exception as e:
#         print(f"Warning: Could not detect GPUs via nvidia-smi: {e}")
#         return []

# def parse_metrics(log_text: str):
#     """Parses output logs for Train/Test Accuracy and Loss."""
#     metrics = {}
    
#     # 1. Parse Test Results (Printed at end)
#     # Looking for: "Without Saving: Test | loss 0.1234 acc 0.890"
#     test_match = re.search(r"Test\s+\|\s+loss\s+([\d\.]+)\s+acc\s+([\d\.]+)", log_text)
#     if test_match:
#         metrics["test_loss"] = float(test_match.group(1))
#         metrics["acc_test"] = float(test_match.group(2))
#     else:
#         metrics["test_loss"] = 0.0
#         metrics["acc_test"] = 0.0

#     # 2. Parse Last Epoch Training Stats
#     # Format: "   50 |        0.1234 / 0.5678 |       0.990 / 0.880"
#     try:
#         lines = log_text.strip().split('\n')
#         # Find lines that look like data rows
#         data_lines = [l for l in lines if '|' in l and l.strip()[0].isdigit()]
#         if data_lines:
#             last_line = data_lines[-1]
#             parts = last_line.split('|')
#             # parts[0]: Epoch, parts[1]: Loss, parts[2]: Acc
            
#             loss_parts = parts[1].strip().split('/')
#             acc_parts = parts[2].strip().split('/')
            
#             metrics["loss_train"] = float(loss_parts[0])
#             metrics["loss_val"] = float(loss_parts[1]) if len(loss_parts) > 1 else 0.0
#             metrics["acc_train"] = float(acc_parts[0])
#             metrics["acc_val"] = float(acc_parts[1]) if len(acc_parts) > 1 else 0.0
#             metrics["last_epoch"] = int(parts[0])
#     except:
#         metrics["loss_train"] = 0.0
#         metrics["acc_train"] = 0.0
#         metrics["acc_val"] = 0.0

#     return metrics

# def run_worker(args):
#     """
#     Worker process that claims a GPU, runs the script, and releases the GPU.
#     """
#     (
#         run_id, fm_kind, reps, depth, ql_op, 
#         script_path_str, base_dir_str, gpu_queue
#     ) = args

#     # 1. Claim a GPU
#     gpu_id = gpu_queue.get() 

#     try:
#         script_path = Path(script_path_str)
#         base_dir = Path(base_dir_str)
        
#         # Directory Structure: base/fm_kind
#         kind_dir = base_dir / fm_kind
#         kind_dir.mkdir(parents=True, exist_ok=True)

#         # --- BUILD COMMAND ---
#         cmd = [sys.executable, str(script_path)] + BASE_ARGS + [
#             "--run-id", str(run_id),
#             "--fm-kind", fm_kind,
#             "--depth", str(depth),
#             "--qlayer-twoq-op", ql_op
#         ]

#         # --- FM SPECIFIC FLAGS ---
#         if fm_kind == "z":
#             cmd += ["--fm-z-reps", str(reps), "--fm-z-pad-mode", PAD_MODE]
#         elif fm_kind == "zz":
#             cmd += [
#                 "--fm-zz-reps", str(reps), 
#                 "--fm-zz-pad-mode", PAD_MODE, 
#                 "--fm-zz-entanglement", FM_ENTANGLEMENT
#             ]
#         elif fm_kind == "eff_su2":
#             cmd += [
#                 "--fm-eff-reps", str(reps), 
#                 "--fm-eff-pad-mod", PAD_MODE, 
#                 "--fm-eff-ent-kind", FM_ENTANGLEMENT, 
#                 "--fm-eff-twoq-op", FM_EFF_OP
#             ]

#         # Log File
#         log_name = f"id{run_id}_{fm_kind}_r{reps}_d{depth}_{ql_op}.txt"
#         log_path = kind_dir / log_name

#         # --- ENV SETUP ---
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#         env["OMP_NUM_THREADS"] = CPU_THREADS_PER_WORKER
#         env["MKL_NUM_THREADS"] = CPU_THREADS_PER_WORKER
#         env["TORCH_NUM_THREADS"] = CPU_THREADS_PER_WORKER

#         # --- EXECUTE ---
#         result = subprocess.run(
#             cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
#         )

#         # Save Log
#         full_log = (
#             f"=== GPU: {gpu_id} | FM: {fm_kind} ===\n"
#             f"=== CMD: {' '.join(cmd)} ===\n\n"
#             f"=== STDOUT ===\n{result.stdout}\n\n"
#             f"=== STDERR ===\n{result.stderr}\n"
#         )
#         log_path.write_text(full_log, encoding="utf-8")

#         if result.returncode != 0:
#             return {
#                 "status": "error", "error_msg": f"Return Code {result.returncode}",
#                 "acc_test": 0, "acc_train": 0
#             }

#         # Parse
#         metrics = parse_metrics(result.stdout)
#         metrics["status"] = "ok"
#         metrics["error_msg"] = ""
#         return metrics
    
#     except Exception as e:
#         return {"status": "error", "error_msg": str(e), "acc_test": 0, "acc_train": 0}
    
#     finally:
#         # 2. Return GPU
#         gpu_queue.put(gpu_id)


# # ================= MAIN EXECUTION =================

# def main():
#     start_time = time.time()
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     base_dir = Path(f"mnist_structural_sweep_{timestamp}")
#     base_dir.mkdir(parents=True, exist_ok=True)

#     print(f"--- STARTING MNIST STRUCTURAL PRIVACY SWEEP ---")
#     print(f"Output Directory: {base_dir}")

#     # 1. GENERATE JOB LIST
#     jobs = []
#     run_counter = 1

#     # Product of all sweep ranges
#     combos = list(itertools.product(FM_KINDS, REPS_LIST, DEPTHS, QLAYER_OPS))
    
#     for fm, reps, depth, ql_op in combos:
#         job = (
#             run_counter, fm, reps, depth, ql_op,
#             str(SCRIPT_PATH), str(base_dir)
#         )
#         jobs.append(job)
#         run_counter += 1

#     total_runs = len(jobs)
#     print(f"Total Configurations: {total_runs}")

#     # 2. GPU SETUP
#     available_gpus = get_free_gpus(min_free_mem_mb=4000)
#     if not available_gpus:
#         print("NO GPUs FOUND! Running in CPU Mode (Slow).")
#         gpu_tickets = [""] # Empty string for CPU
#         max_workers = 2
#     else:
#         print(f"GPUs Found: {available_gpus}")
#         gpu_tickets = []
#         for gpu in available_gpus:
#             for _ in range(JOBS_PER_GPU):
#                 gpu_tickets.append(gpu)
#         random.shuffle(gpu_tickets)
#         max_workers = len(gpu_tickets)

#     print(f"Parallel Workers: {max_workers}")

#     m = Manager()
#     gpu_queue = m.Queue()
#     for t in gpu_tickets: gpu_queue.put(t)

#     # 3. EXECUTION
#     csv_path = base_dir / "mnist_structural_results.csv"
#     errors_path = base_dir / "errors.txt"

#     with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
#             errors_path.open("w", encoding="utf-8") as f_err:

#         fieldnames = [
#             "run_id", "fm_kind", "reps", "depth", "ql_op",
#             "acc_train", "acc_test", "acc_val", 
#             "loss_train", "loss_val", "test_loss", 
#             "last_epoch", "status", "error_msg", "gpu_used"
#         ]
#         writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
#         writer.writeheader()

#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             # Pass queue to worker
#             jobs_with_queue = [j + (gpu_queue,) for j in jobs]
#             futures = {executor.submit(run_worker, j): j for j in jobs_with_queue}

#             completed = 0
#             for future in as_completed(futures):
#                 # Get original job params (first 5 args)
#                 job_data = futures[future]
#                 rid, fm, r, d, op = job_data[:5]
                
#                 cfg_str = f"[{fm}] r{r} d{d} op{op}"

#                 try:
#                     metrics = future.result()
                    
#                     row = {
#                         "run_id": rid, "fm_kind": fm, "reps": r, "depth": d, "ql_op": op,
#                         **metrics
#                     }
#                     writer.writerow(row)
#                     f_csv.flush()

#                     if metrics["status"] == "ok":
#                         print(f"[OK] {cfg_str} -> Test Acc: {metrics.get('acc_test')}")
#                     else:
#                         print(f"[ERR] {cfg_str} -> {metrics.get('error_msg')}")
#                         f_err.write(f"{cfg_str} | {metrics.get('error_msg')}\n")
                        
#                 except Exception as e:
#                     print(f"[CRASH] {cfg_str} -> {e}")

#                 completed += 1
#                 if completed % 20 == 0:
#                     print(f"Progress: {completed}/{total_runs} ({completed/total_runs:.1%})")

#     print(f"=== SWEEP COMPLETE ===")
#     print(f"Results: {csv_path}")

# if __name__ == "__main__":
#     main()



import csv
import itertools
import os
import re
import sys
import subprocess
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from multiprocessing import Manager

# ================= CONFIGURATION =================

# Path to your training script
SCRIPT_PATH = Path("experiments/qurift_main.py")

# SPEED SETTING: Jobs per GPU (Adjust based on VRAM/Stability)
JOBS_PER_GPU = 6 

# CPU SAFETY
CPU_THREADS_PER_WORKER = "2"

# FIXED ARGUMENTS (Passed to every run)
BASE_ARGS = [
    "--model-type", "qnn",
    "--dataset", "mnist",
    "--random-ops", "0",
    
    # Increased Sample Size (200) for more robustness
    "--vector-train", "200",
    "--vector-valid", "200",
    "--vector-test", "200",
    
    "--batch-size", "16", # Slightly larger batch for 200 samples
    "--epochs", "60",
    
    # Flags
    "--train_target",
]

# === EXTENSIVE SWEEP RANGES ===

# 1. Feature Map Params
FM_KINDS = ["z", "zz", "eff_su2"]
N_WIRES_LIST = [4, 6, 8, 10] 
REPS_LIST    = [1, 2, 3, 4, 5]
PAD_MODE     = "wrap"
FM_ENTANGLEMENT = "linear"
FM_EFF_OPS   = ["cx", "cz"] # Only for eff_su2

# 2. Q-Layer Params
DEPTHS       = [2, 4, 6]
QLAYER_ENTS  = ["linear", "pairwise", "full"]
QLAYER_OPS   = ["cx", "crz", "rzz", "swap"]
QLAYER_REV   = False # Fixed

# ================= HELPER FUNCTIONS =================

def get_free_gpus(min_free_mem_mb=8000):
    """Detects GPUs with at least 'min_free_mem_mb' available."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('utf-8')
        free_gpus = []
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            idx, free_mem = line.split(',')
            if int(free_mem) > min_free_mem_mb:
                free_gpus.append(idx.strip())
        return free_gpus
    except Exception as e:
        print(f"Warning: Could not detect GPUs via nvidia-smi: {e}")
        return []

def parse_metrics(log_text: str):
    """Parses output logs for Train/Test Accuracy and Loss."""
    metrics = {}
    
    # 1. Parse Test Results
    test_match = re.search(r"Test\s+\|\s+loss\s+([\d\.]+)\s+acc\s+([\d\.]+)", log_text)
    if test_match:
        metrics["test_loss"] = float(test_match.group(1))
        metrics["acc_test"] = float(test_match.group(2))
    else:
        metrics["test_loss"] = 0.0
        metrics["acc_test"] = 0.0

    # 2. Parse Last Epoch Training Stats
    try:
        lines = log_text.strip().split('\n')
        data_lines = [l for l in lines if '|' in l and l.strip()[0].isdigit()]
        if data_lines:
            last_line = data_lines[-1]
            parts = last_line.split('|')
            loss_parts = parts[1].strip().split('/')
            acc_parts = parts[2].strip().split('/')
            
            metrics["loss_train"] = float(loss_parts[0])
            metrics["loss_val"] = float(loss_parts[1]) if len(loss_parts) > 1 else 0.0
            metrics["acc_train"] = float(acc_parts[0])
            metrics["acc_val"] = float(acc_parts[1]) if len(acc_parts) > 1 else 0.0
            metrics["last_epoch"] = int(parts[0])
    except:
        metrics["loss_train"] = 0.0
        metrics["acc_train"] = 0.0
        metrics["acc_val"] = 0.0

    return metrics

def run_worker(args):
    """
    Worker process that claims a GPU, runs the script, and releases the GPU.
    """
    (
        run_id, fm_kind, n_wires, reps, fm_op, 
        depth, ql_ent, ql_op,
        script_path_str, base_dir_str, gpu_queue
    ) = args

    gpu_id = gpu_queue.get() 

    try:
        script_path = Path(script_path_str)
        base_dir = Path(base_dir_str)
        
        # Directory Structure: base/fm_kind
        kind_dir = base_dir / fm_kind
        kind_dir.mkdir(parents=True, exist_ok=True)

        # --- BUILD COMMAND ---
        cmd = [sys.executable, str(script_path)] + BASE_ARGS + [
            "--run-id", str(run_id),
            "--fm-kind", fm_kind,
            "--n-wires", str(n_wires),
            "--depth", str(depth),
            "--qlayer-ent-kind", ql_ent,
            "--qlayer-twoq-op", ql_op
        ]
        
        # Reverse arg (Fixed False here, but logical check included)
        if QLAYER_REV:
             cmd.append("--qlayer-ent-wire-reverse")

        # --- FM SPECIFIC FLAGS ---
        if fm_kind == "z":
            cmd += ["--fm-z-reps", str(reps), "--fm-z-pad-mode", PAD_MODE]
        elif fm_kind == "zz":
            cmd += [
                "--fm-zz-reps", str(reps), 
                "--fm-zz-pad-mode", PAD_MODE, 
                "--fm-zz-entanglement", FM_ENTANGLEMENT
            ]
        elif fm_kind == "eff_su2":
            cmd += [
                "--fm-eff-reps", str(reps), 
                "--fm-eff-pad-mod", PAD_MODE, 
                "--fm-eff-ent-kind", FM_ENTANGLEMENT, 
                "--fm-eff-twoq-op", fm_op
            ]

        # Log File
        log_name = f"id{run_id}_{fm_kind}_w{n_wires}_r{reps}_d{depth}_{ql_op}.txt"
        log_path = kind_dir / log_name

        # --- ENV SETUP ---
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = CPU_THREADS_PER_WORKER
        env["MKL_NUM_THREADS"] = CPU_THREADS_PER_WORKER
        env["TORCH_NUM_THREADS"] = CPU_THREADS_PER_WORKER

        # --- EXECUTE ---
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        full_log = (
            f"=== GPU: {gpu_id} | FM: {fm_kind} ===\n"
            f"=== CMD: {' '.join(cmd)} ===\n\n"
            f"=== STDOUT ===\n{result.stdout}\n\n"
            f"=== STDERR ===\n{result.stderr}\n"
        )
        log_path.write_text(full_log, encoding="utf-8")

        if result.returncode != 0:
            return {
                "status": "error", "error_msg": f"Return Code {result.returncode}",
                "acc_test": 0, "acc_train": 0
            }

        metrics = parse_metrics(result.stdout)
        metrics["status"] = "ok"
        metrics["error_msg"] = ""
        return metrics
    
    except Exception as e:
        return {"status": "error", "error_msg": str(e), "acc_test": 0, "acc_train": 0}
    
    finally:
        gpu_queue.put(gpu_id)


# ================= MAIN EXECUTION =================

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(f"mnist_extensive_sweep_qnn{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- STARTING MNIST EXTENSIVE SWEEP (200 Samples) ---")
    print(f"Output Directory: {base_dir}")

    # 1. GENERATE JOB LIST (Conditional Logic)
    jobs = []
    run_counter = 1

    # Base combinations (Wires, Reps, Depth, QL_Ent, QL_Op)
    # We iterate these for ALL feature maps
    base_combos = list(itertools.product(
        N_WIRES_LIST, REPS_LIST, DEPTHS, QLAYER_ENTS, QLAYER_OPS
    ))
    
    # Iterate Feature Maps
    for fm in FM_KINDS:
        
        # Determine valid FM_OPs for this FM
        valid_fm_ops = ["NA"] # Default for z and zz
        if fm == "eff_su2":
            valid_fm_ops = FM_EFF_OPS # ["cx", "cz"]
            
        for fm_op in valid_fm_ops:
            for wires, reps, depth, ql_ent, ql_op in base_combos:
                
                job = (
                    run_counter, fm, wires, reps, fm_op,
                    depth, ql_ent, ql_op,
                    str(SCRIPT_PATH), str(base_dir)
                )
                jobs.append(job)
                run_counter += 1

    total_runs = len(jobs)
    print(f"Total Configurations: {total_runs}")

    # 2. GPU SETUP
    available_gpus = get_free_gpus(min_free_mem_mb=4000)
    if not available_gpus:
        print("NO GPUs FOUND! Running in CPU Mode.")
        gpu_tickets = [""] 
        max_workers = 2
    else:
        print(f"GPUs Found: {available_gpus}")
        gpu_tickets = []
        for gpu in available_gpus:
            for _ in range(JOBS_PER_GPU):
                gpu_tickets.append(gpu)
        random.shuffle(gpu_tickets)
        max_workers = len(gpu_tickets)

    print(f"Parallel Workers: {max_workers}")

    m = Manager()
    gpu_queue = m.Queue()
    for t in gpu_tickets: gpu_queue.put(t)

    # 3. EXECUTION
    csv_path = base_dir / "qnn_extensive_results.csv"
    errors_path = base_dir / "errors.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
            errors_path.open("w", encoding="utf-8") as f_err:

        fieldnames = [
            "run_id", "fm_kind", "n_wires", "reps", "fm_op", 
            "depth", "ql_ent", "ql_op",
            "acc_train", "acc_test", "acc_val", 
            "loss_train", "loss_val", "test_loss", 
            "last_epoch", "status", "error_msg", "gpu_used"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            jobs_with_queue = [j + (gpu_queue,) for j in jobs]
            futures = {executor.submit(run_worker, j): j for j in jobs_with_queue}

            completed = 0
            for future in as_completed(futures):
                job_data = futures[future]
                # Unpack critical display info
                rid, fm, wires, r, fop, d, qe, qo = job_data[:8]
                
                cfg_str = f"[{fm}:{fop}] w{wires} r{r} d{d} {qo}"

                try:
                    metrics = future.result()
                    
                    row = {
                        "run_id": rid, "fm_kind": fm, "n_wires": wires, 
                        "reps": r, "fm_op": fop, "depth": d, 
                        "ql_ent": qe, "ql_op": qo,
                        **metrics
                    }
                    writer.writerow(row)
                    f_csv.flush()

                    if metrics["status"] == "ok":
                        # Simple compact printing
                        print(f"[OK] {cfg_str} -> Acc: {metrics.get('acc_test')}")
                    else:
                        print(f"[ERR] {cfg_str} -> {metrics.get('error_msg')}")
                        f_err.write(f"{cfg_str} | {metrics.get('error_msg')}\n")
                        
                except Exception as e:
                    print(f"[CRASH] {cfg_str} -> {e}")

                completed += 1
                if completed % 50 == 0:
                    prog = completed/total_runs
                    elapsed = time.time() - start_time
                    est_total = elapsed / prog if prog > 0 else 0
                    rem = est_total - elapsed
                    print(f"--- Progress: {completed}/{total_runs} ({prog:.1%}) | Time: {elapsed/60:.1f}m | Rem: {rem/60:.1f}m ---")

    print(f"=== EXTENSIVE SWEEP COMPLETE ===")
    print(f"Results: {csv_path}")

if __name__ == "__main__":
    main()
