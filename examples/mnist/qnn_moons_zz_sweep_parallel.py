
# import time  # <--- Add this

# import csv
# import itertools
# import os
# import re
# import sys
# import traceback
# import subprocess
# import random  # <--- The key import for load balancing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from datetime import datetime
# from pathlib import Path
# from multiprocessing import Manager

# # ================= CONFIGURATION =================

# # Path to your training script
# SCRIPT_PATH = Path("examples/mnist/mnist_2qubit_4class.py")

# # SPEED SETTING: How many scripts to run on ONE GPU at the same time?
# # Your screenshot showed 98% Compute Load with 4 jobs. 
# # Do NOT go higher than 4, or they will slow each other down.
# JOBS_PER_GPU = 4 

# # CPU SAFETY: Limit threads per worker to prevent server freeze.
# # 32 workers * 2 threads = 64 CPU cores used. This is safe and stable.
# CPU_THREADS_PER_WORKER = "2"

# BASE_ARGS = [
#     "--model-type", "qnn",
#     "--n-wires", "4",
#     "--random-ops", "0",
#     "--dataset", "moons",
#     "--vector-train", "50",
#     "--vector-valid", "50",
#     "--vector-test", "50",
#     "--batch-size", "8",
#     "--epochs", "100",
#     "--moons-noise", "0.3",
#     "--train_target",
#     "--fm-kind", "zz",
#     "--extra-feats",
# ]

# DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
# ENTANGLEMENTS = ["linear", "ring", "full"]
# PAD_MODES = ["zero"]
# REPS_LIST = [1, 2, 3, 4, 5, 6]


# # ================= HELPER FUNCTIONS =================

# def get_free_gpus(min_free_mem_mb=10000):
#     """
#     Detects GPUs with at least 'min_free_mem_mb' available.
#     Returns a list of GPU IDs (strings).
#     """
#     try:
#         # Query nvidia-smi for index and free memory
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

# def parse_last_epoch_and_test(log_text: str):
#     """
#     Parses the output log for the final epoch metrics and test results.
#     """
#     epoch_pattern = re.compile(
#         r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)",
#         re.MULTILINE,
#     )
#     epoch_matches = epoch_pattern.findall(log_text)
#     if not epoch_matches:
#         raise ValueError("No epoch lines found in log")

#     last = epoch_matches[-1]
#     last_epoch = int(last[0])
#     loss_train = float(last[1])
#     loss_val = float(last[2])
#     acc_train = float(last[3])
#     acc_val = float(last[4])

#     test_pattern = re.compile(
#         r"Test\s*\|\s*loss\s*([0-9.]+)\s*acc\s*([0-9.]+)",
#         re.MULTILINE,
#     )
#     m_test = test_pattern.search(log_text)
#     if m_test:
#         test_loss = float(m_test.group(1))
#         test_acc = float(m_test.group(2))
#     else:
#         test_loss = float("nan")
#         test_acc = float("nan")

#     return {
#         "last_epoch": last_epoch,
#         "loss_train": loss_train,
#         "loss_val": loss_val,
#         "acc_train": acc_train,
#         "acc_val": acc_val,
#         "test_loss": test_loss,
#         "test_acc": test_acc,
#     }

# def run_config(args):
#     """
#     Worker process.
#     1. Gets a GPU ticket (ID) from the queue.
#     2. Runs the subprocess.
#     3. Returns the ticket to the queue.
#     """
#     (
#         run_id, depth, entanglement, pad_mode, reps,
#         script_path_str, base_dir_str, gpu_queue
#     ) = args

#     # 1. WAIT FOR AND CLAIM A GPU TICKET
#     gpu_id = gpu_queue.get() 

#     try:
#         script_path = Path(script_path_str)
#         base_dir = Path(base_dir_str)

#         cmd = [sys.executable, str(script_path)] + BASE_ARGS + [
#             "--depth", str(depth),
#             "--fm-zz-entanglement", entanglement,
#             "--fm-zz-pad-mode", pad_mode,
#             "--fm-zz-reps", str(reps),
#         ]

#         raw_log_name = (
#             f"log_run{run_id}_d{depth}_ent{entanglement}_"
#             f"pad{pad_mode}_reps{reps}.txt"
#         )
#         raw_log_path = base_dir / raw_log_name

#         # 2. CONFIGURE ENVIRONMENT (GPU + CPU LIMITS)
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
#         # Crucial for stability when running many jobs!
#         env["OMP_NUM_THREADS"] = CPU_THREADS_PER_WORKER
#         env["MKL_NUM_THREADS"] = CPU_THREADS_PER_WORKER
#         env["TORCH_NUM_THREADS"] = CPU_THREADS_PER_WORKER

#         result = subprocess.run(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             env=env,
#         )

#         full_log = (
#             f"=== RUNNING ON GPU: {gpu_id} ===\n"
#             "=== CMD ===\n"
#             + " ".join(cmd)
#             + "\n\n=== STDOUT ===\n"
#             + result.stdout
#             + "\n\n=== STDERR ===\n"
#             + result.stderr
#             + f"\n\n=== RETURN CODE: {result.returncode} ===\n"
#         )
#         raw_log_path.write_text(full_log, encoding="utf-8")

#         if result.returncode != 0:
#             raise RuntimeError(
#                 f"Run {run_id} failed with return code {result.returncode}. "
#                 f"See log: {raw_log_path}"
#             )

#         metrics = parse_last_epoch_and_test(result.stdout)
#         return metrics
    
#     finally:
#         # 3. RETURN THE TICKET (Crucial!)
#         gpu_queue.put(gpu_id)


# # ================= MAIN EXECUTION =================

# def main():
#     # --- TIMER START ---
#     start_time = time.time()
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     print("--- STARTING SWEEP (RANDOM BALANCED MODE) ---")
    
#     # 1. DETECT AND PREPARE GPUS
#     available_gpus = get_free_gpus(min_free_mem_mb=10000)
    
#     if not available_gpus:
#         print("NO GPUs FOUND! Falling back to CPU mode (1 worker).")
#         max_workers = 1
#         gpu_tickets = [""] 
#     else:
#         print(f"Found {len(available_gpus)} Physical GPUs: {available_gpus}")
#         print(f"Packing Strategy: {JOBS_PER_GPU} jobs per GPU")
        
#         gpu_tickets = []
#         for gpu in available_gpus:
#             for _ in range(JOBS_PER_GPU):
#                 gpu_tickets.append(gpu)
        
#         # SHUFFLE FOR LOAD BALANCING
#         random.shuffle(gpu_tickets)
        
#         max_workers = len(gpu_tickets)
#         print(f"Total Parallel Workers: {max_workers}")

#     # 2. FILL THE QUEUE
#     m = Manager()
#     gpu_queue = m.Queue()
#     for ticket in gpu_tickets:
#         gpu_queue.put(ticket)

#     # 3. PREPARE OUTPUTS
#     base_dir = Path(f"qnn_moons_zz_sweep_{timestamp}")
#     base_dir.mkdir(parents=True, exist_ok=True)

#     csv_path = base_dir / "results_last_epoch.csv"
#     errors_path = base_dir / "errors.txt"

#     combos = list(
#         itertools.product(DEPTHS, ENTANGLEMENTS, PAD_MODES, REPS_LIST)
#     )
#     total_runs = len(combos)

#     print(f"Total configurations to run: {total_runs}")
#     print(f"Output Directory: {base_dir}")

#     # 4. RUN EXECUTION LOOP
#     with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
#             errors_path.open("w", encoding="utf-8") as f_err:

#         writer = csv.DictWriter(
#             f_csv,
#             fieldnames=[
#                 "run_id", "depth", "fm_zz_entanglement", "fm_zz_pad_mode",
#                 "fm_zz_reps", "last_epoch", "acc_train", "acc_test",
#                 "acc_val", "loss_train", "loss_val", "test_loss",
#                 "status", "error_msg",
#             ],
#         )
#         writer.writeheader()

#         jobs = []
#         for run_id, (depth, ent, pad, reps) in enumerate(combos, start=1):
#             jobs.append(
#                 (
#                     run_id, depth, ent, pad, reps,
#                     str(SCRIPT_PATH), str(base_dir), gpu_queue
#                 )
#             )

#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             future_to_job = {
#                 executor.submit(run_config, job): job for job in jobs
#             }

#             completed = 0
#             for future in as_completed(future_to_job):
#                 job_data = future_to_job[future]
#                 run_id = job_data[0]
#                 cfg_str = f"Run {run_id}"

#                 try:
#                     metrics = future.result()
#                     status = "ok"
#                     error_msg = ""
#                     print(f"[OK]   {cfg_str} | Acc: {metrics.get('test_acc', 'N/A')}")
#                 except Exception as e:
#                     status = "error"
#                     error_msg = repr(e)
#                     print(f"[ERR]  {cfg_str} -> {error_msg}")

#                     f_err.write(f"{cfg_str}\n{error_msg}\n")
#                     # Empty metrics for failed runs
#                     metrics = {k: "" for k in ["last_epoch", "loss_train", "loss_val", "acc_train", "acc_val", "test_loss", "test_acc"]}

#                 writer.writerow({
#                     "run_id": run_id,
#                     # ... (Keep mapping same as before) ...
#                     "depth": job_data[1],
#                     "fm_zz_entanglement": job_data[2],
#                     "fm_zz_pad_mode": job_data[3],
#                     "fm_zz_reps": job_data[4],
#                     "last_epoch": metrics.get("last_epoch", ""),
#                     "acc_train": metrics.get("acc_train", ""),
#                     "acc_test": metrics.get("test_acc", ""),
#                     "acc_val": metrics.get("acc_val", ""),
#                     "loss_train": metrics.get("loss_train", ""),
#                     "loss_val": metrics.get("loss_val", ""),
#                     "test_loss": metrics.get("test_loss", ""),
#                     "status": status,
#                     "error_msg": error_msg,
#                 })
#                 f_csv.flush()

#                 completed += 1
                
#                 # Estimate remaining time
#                 elapsed_so_far = time.time() - start_time
#                 avg_time_per_job = elapsed_so_far / completed
#                 remaining_jobs = total_runs - completed
#                 est_remaining = remaining_jobs * avg_time_per_job
                
#                 # Format estimated time
#                 est_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining))
#                 print(f"Progress: {completed}/{total_runs} ({(completed/total_runs)*100:.1f}%) - Est. Remaining: {est_str}")

#     # --- TIMER END ---
#     total_seconds = time.time() - start_time
#     final_time_str = time.strftime("%H:%M:%S", time.gmtime(total_seconds))
    
#     print("\n" + "="*30)
#     print(f"=== SWEEP FINISHED ===")
#     print(f"Total Execution Time: {final_time_str}")
#     print(f"Results saved to: {csv_path}")
#     print("="*30)

# if __name__ == "__main__":
#     main()


import csv
import itertools
import os
import re
import sys
import traceback
import subprocess
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from multiprocessing import Manager

# ================= CONFIGURATION =================

# Path to your training script
SCRIPT_PATH = Path("examples/mnist/mnist_2qubit_4class.py")

# SPEED SETTING: How many scripts to run on ONE GPU at the same time?
# 4 is safe for A100 80GB. Do not increase beyond 4 to avoid compute bottleneck.
JOBS_PER_GPU = 4 

# CPU SAFETY: Limit threads per worker to prevent server freeze.
# 32 workers * 2 threads = 64 CPU cores used.
CPU_THREADS_PER_WORKER = "2"

BASE_ARGS = [
    "--model-type", "qnn",
    # "--n-wires" REMOVED from here, it is now a sweep parameter
    "--random-ops", "0",
    "--dataset", "moons",
    "--vector-train", "50",
    "--vector-valid", "50",
    "--vector-test", "50",
    "--batch-size", "8",
    "--epochs", "100",
    "--moons-noise", "0.3",
    "--train_target",
    "--fm-kind", "zz",
    "--extra-feats",
]

# === SWEEP PARAMETERS ===
N_WIRES_LIST = [2, 3, 4]  # <--- Now iterating over this
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
ENTANGLEMENTS = ["linear", "ring", "full"]
PAD_MODES = ["wrap", "repeatlast", "zero"] # <--- Expanded choices
REPS_LIST = [1, 2, 3, 4, 5, 6]


# ================= HELPER FUNCTIONS =================

def get_free_gpus(min_free_mem_mb=10000):
    """
    Detects GPUs with at least 'min_free_mem_mb' available.
    Returns a list of GPU IDs (strings).
    """
    try:
        # Query nvidia-smi for index and free memory
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

def parse_last_epoch_and_test(log_text: str):
    """
    Parses the output log for the final epoch metrics and test results.
    """
    epoch_pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)",
        re.MULTILINE,
    )
    epoch_matches = epoch_pattern.findall(log_text)
    if not epoch_matches:
        raise ValueError("No epoch lines found in log")

    last = epoch_matches[-1]
    last_epoch = int(last[0])
    loss_train = float(last[1])
    loss_val = float(last[2])
    acc_train = float(last[3])
    acc_val = float(last[4])

    test_pattern = re.compile(
        r"Test\s*\|\s*loss\s*([0-9.]+)\s*acc\s*([0-9.]+)",
        re.MULTILINE,
    )
    m_test = test_pattern.search(log_text)
    if m_test:
        test_loss = float(m_test.group(1))
        test_acc = float(m_test.group(2))
    else:
        test_loss = float("nan")
        test_acc = float("nan")

    return {
        "last_epoch": last_epoch,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

def run_config(args):
    """
    Worker process.
    1. Gets a GPU ticket (ID) from the queue.
    2. Runs the subprocess.
    3. Returns the ticket to the queue.
    """
    (
        run_id, n_wires, depth, entanglement, pad_mode, reps,
        script_path_str, base_dir_str, gpu_queue
    ) = args

    # 1. WAIT FOR AND CLAIM A GPU TICKET
    gpu_id = gpu_queue.get() 

    try:
        script_path = Path(script_path_str)
        base_dir = Path(base_dir_str)

        # Build command with dynamic n_wires
        cmd = [sys.executable, str(script_path)] + BASE_ARGS + [
            "--n-wires", str(n_wires),
            "--depth", str(depth),
            "--fm-zz-entanglement", entanglement,
            "--fm-zz-pad-mode", pad_mode,
            "--fm-zz-reps", str(reps),
        ]

        raw_log_name = (
            f"log_run{run_id}_w{n_wires}_d{depth}_ent{entanglement}_"
            f"pad{pad_mode}_reps{reps}.txt"
        )
        raw_log_path = base_dir / raw_log_name

        # 2. CONFIGURE ENVIRONMENT (GPU + CPU LIMITS)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Crucial for stability when running many jobs!
        env["OMP_NUM_THREADS"] = CPU_THREADS_PER_WORKER
        env["MKL_NUM_THREADS"] = CPU_THREADS_PER_WORKER
        env["TORCH_NUM_THREADS"] = CPU_THREADS_PER_WORKER

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        full_log = (
            f"=== RUNNING ON GPU: {gpu_id} ===\n"
            "=== CMD ===\n"
            + " ".join(cmd)
            + "\n\n=== STDOUT ===\n"
            + result.stdout
            + "\n\n=== STDERR ===\n"
            + result.stderr
            + f"\n\n=== RETURN CODE: {result.returncode} ===\n"
        )
        raw_log_path.write_text(full_log, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"Run {run_id} failed with return code {result.returncode}. "
                f"See log: {raw_log_path}"
            )

        metrics = parse_last_epoch_and_test(result.stdout)
        return metrics
    
    finally:
        # 3. RETURN THE TICKET (Crucial!)
        gpu_queue.put(gpu_id)


# ================= MAIN EXECUTION =================

def main():
    # --- TIMER START ---
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("--- STARTING SWEEP (RANDOM BALANCED MODE V2) ---")
    
    # 1. DETECT AND PREPARE GPUS
    available_gpus = get_free_gpus(min_free_mem_mb=10000)
    
    if not available_gpus:
        print("NO GPUs FOUND! Falling back to CPU mode (1 worker).")
        max_workers = 1
        gpu_tickets = [""] 
    else:
        print(f"Found {len(available_gpus)} Physical GPUs: {available_gpus}")
        print(f"Packing Strategy: {JOBS_PER_GPU} jobs per GPU")
        
        gpu_tickets = []
        for gpu in available_gpus:
            for _ in range(JOBS_PER_GPU):
                gpu_tickets.append(gpu)
        
        # SHUFFLE FOR LOAD BALANCING
        random.shuffle(gpu_tickets)
        
        max_workers = len(gpu_tickets)
        print(f"Total Parallel Workers: {max_workers}")

    # 2. FILL THE QUEUE
    m = Manager()
    gpu_queue = m.Queue()
    for ticket in gpu_tickets:
        gpu_queue.put(ticket)

    # 3. PREPARE OUTPUTS
    base_dir = Path(f"qnn_moons_zz_sweep_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "results_last_epoch.csv"
    errors_path = base_dir / "errors.txt"

    # Generate combinations (Order: n_wires, depths, ent, pad, reps)
    combos = list(
        itertools.product(N_WIRES_LIST, DEPTHS, ENTANGLEMENTS, PAD_MODES, REPS_LIST)
    )
    total_runs = len(combos)

    print(f"Total configurations to run: {total_runs}")
    print(f"Output Directory: {base_dir}")

    # 4. RUN EXECUTION LOOP
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
            errors_path.open("w", encoding="utf-8") as f_err:

        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "run_id", "n_wires", "depth", "fm_zz_entanglement", 
                "fm_zz_pad_mode", "fm_zz_reps", "last_epoch", 
                "acc_train", "acc_test", "acc_val", 
                "loss_train", "loss_val", "test_loss",
                "status", "error_msg",
            ],
        )
        writer.writeheader()

        jobs = []
        for run_id, (n_wires, depth, ent, pad, reps) in enumerate(combos, start=1):
            jobs.append(
                (
                    run_id, n_wires, depth, ent, pad, reps,
                    str(SCRIPT_PATH), str(base_dir), gpu_queue
                )
            )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(run_config, job): job for job in jobs
            }

            completed = 0
            for future in as_completed(future_to_job):
                job_data = future_to_job[future]
                # Unpack parameters to match the job tuple structure
                run_id, n_wires, depth, ent, pad, reps = job_data[:6]
                
                cfg_str = f"Run {run_id}: w={n_wires}, d={depth}, ent={ent}"

                try:
                    metrics = future.result()
                    status = "ok"
                    error_msg = ""
                    print(f"[OK]   {cfg_str} | Acc: {metrics.get('test_acc', 'N/A')}")
                except Exception as e:
                    status = "error"
                    error_msg = repr(e)
                    print(f"[ERR]  {cfg_str} -> {error_msg}")

                    f_err.write(f"{cfg_str}\n{error_msg}\n")
                    # Empty metrics for failed runs
                    metrics = {k: "" for k in ["last_epoch", "loss_train", "loss_val", "acc_train", "acc_val", "test_loss", "test_acc"]}

                writer.writerow({
                    "run_id": run_id,
                    "n_wires": n_wires,
                    "depth": depth,
                    "fm_zz_entanglement": ent,
                    "fm_zz_pad_mode": pad,
                    "fm_zz_reps": reps,
                    "last_epoch": metrics.get("last_epoch", ""),
                    "acc_train": metrics.get("acc_train", ""),
                    "acc_test": metrics.get("test_acc", ""),
                    "acc_val": metrics.get("acc_val", ""),
                    "loss_train": metrics.get("loss_train", ""),
                    "loss_val": metrics.get("loss_val", ""),
                    "test_loss": metrics.get("test_loss", ""),
                    "status": status,
                    "error_msg": error_msg,
                })
                f_csv.flush()

                completed += 1
                
                # Estimate remaining time
                elapsed_so_far = time.time() - start_time
                avg_time_per_job = elapsed_so_far / completed
                remaining_jobs = total_runs - completed
                est_remaining = remaining_jobs * avg_time_per_job
                
                # Format estimated time
                est_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining))
                print(f"Progress: {completed}/{total_runs} ({(completed/total_runs)*100:.1f}%) - Est. Remaining: {est_str}")

    # --- TIMER END ---
    total_seconds = time.time() - start_time
    final_time_str = time.strftime("%H:%M:%S", time.gmtime(total_seconds))
    
    print("\n" + "="*30)
    print(f"=== SWEEP FINISHED ===")
    print(f"Total Execution Time: {final_time_str}")
    print(f"Results saved to: {csv_path}")
    print("="*30)

if __name__ == "__main__":
    main()