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

# SPEED SETTING: Jobs per GPU 
# HQNN might use slightly more VRAM due to classical layers + quantum.
# If OOM occurs, lower this to 4.
JOBS_PER_GPU = 6 

# CPU SAFETY
CPU_THREADS_PER_WORKER = "2"

# FIXED ARGUMENTS (Passed to every run)
BASE_ARGS = [
    "--model-type", "hqnn",         # <--- CHANGED TO HQNN
    "--dataset", "mnist",
    "--random-ops", "0",
    
    # Data Settings (200 samples per split)
    "--vector-train", "200",
    "--vector-valid", "200",
    "--vector-test", "200",
    
    "--batch-size", "16", 
    "--epochs", "60",
    
    # Flags
    "--train_target",
]

# === EXTENSIVE SWEEP RANGES ===

# 1. Feature Map Params
FM_KINDS     = ["z", "zz", "eff_su2"]
N_WIRES_LIST = [4, 6, 8, 10]        # <--- Scaling the Quantum Bottleneck
REPS_LIST    = [1, 2, 3, 4, 5]
PAD_MODE     = "wrap"               # Fixed
FM_ENTANGLEMENT = "linear"          # Fixed
FM_EFF_OPS   = ["cx", "cz"]         # Only for eff_su2

# 2. Q-Layer Params
DEPTHS       = [2, 4, 6]
QLAYER_ENTS  = ["linear", "pairwise", "full"]
QLAYER_OPS   = ["cx", "crz", "rzz", "swap"]
QLAYER_REV   = False                # Fixed

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
        return [] # Fallback to CPU handling in main

def parse_metrics(log_text: str):
    """Parses output logs for Train/Test Accuracy and Loss."""
    metrics = {}
    
    # 1. Parse Test Results (Printed at end)
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
        metrics["last_epoch"] = 0

    return metrics

def run_worker(args):
    """
    Worker process: Claims GPU -> Runs Config -> Returns GPU.
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
        
        # Directory Structure: base/fm_kind/ql_op (organized for sanity)
        kind_dir = base_dir / fm_kind / ql_op
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

        # Save Log
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
    
    base_dir = Path(f"hqnn_sweep_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- STARTING HQNN EXTENSIVE SWEEP (200 Samples) ---")
    print(f"Output Directory: {base_dir}")

    # 1. GENERATE JOB LIST
    jobs = []
    run_counter = 1

    # Base combinations: Wires x Reps x Depth x QL_Ent x QL_Op
    base_combos = list(itertools.product(
        N_WIRES_LIST, REPS_LIST, DEPTHS, QLAYER_ENTS, QLAYER_OPS
    ))
    
    # Iterate Feature Maps
    for fm in FM_KINDS:
        
        # Determine valid FM_OPs for this FM
        # For z/zz, we use "NA" as a placeholder to keep structure consistent
        valid_fm_ops = ["NA"] 
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
    csv_path = base_dir / "hqnn_extensive_results.csv"
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
                # Unpack first 8 items to print nice status
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

    print(f"=== HQNN SWEEP COMPLETE ===")
    print(f"Results: {csv_path}")

if __name__ == "__main__":
    main()