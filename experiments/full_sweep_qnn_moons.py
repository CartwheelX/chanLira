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

# SPEED SETTING: How many scripts to run on ONE GPU at the same time?
JOBS_PER_GPU = 4 

# CPU SAFETY: Limit threads per worker.
CPU_THREADS_PER_WORKER = "2"

# FIXED ARGUMENTS (Passed to every run)
BASE_ARGS = [
    "--model-type", "qnn",
    "--random-ops", "0",
    "--dataset", "moons",
    "--vector-train", "50",
    "--vector-valid", "50",
    "--vector-test", "50",
    "--batch-size", "8",
    "--epochs", "100",
    "--moons-noise", "0.3",
    "--train_target",
    "--extra-feats",
    # "--plot-vector",
]




# === SWEEP RANGES ===

# 1. Feature Map params
FM_KINDS = ["z", "zz", "eff_su2"] # fm_kind
N_WIRES_LIST = [2, 3, 4] # n_wires
REPS_LIST    = [1, 2, 3, 4, 5] # reps
PAD_MODES    = ["wrap", "repeatlast", "zero"] # pad_mode
FM_ENTANGLEMENTS = ["linear", "ring", "full"] # fm_ent
REDUCED_FM_EFF_TWOQ_OPS = ["cx", "cz"] # fm_op

# 2. Q-Layer params
DEPTHS       = [2, 3, 4, 5, 6] # depth
QLAYER_ENT_KINDS = ["linear", "pairwise", "full"] # ql_ent
QLAYER_TWOQ_OPS  = ["cx", "crz", "rzz", "swap"]  # ql_op
QLAYER_REVERSE   = [False, True] # ql_rev


# ================= HELPER FUNCTIONS =================

def get_free_gpus(min_free_mem_mb=10000):
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

def parse_last_epoch_and_test(log_text: str):
    """Parses logs for metrics."""
    epoch_pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*/\s*([0-9.]+)",
        re.MULTILINE,
    )
    epoch_matches = epoch_pattern.findall(log_text)
    
    last_epoch = -1
    loss_train = loss_val = acc_train = acc_val = 0.0

    if epoch_matches:
        last = epoch_matches[-1]
        last_epoch = int(last[0])
        loss_train = float(last[1])
        loss_val = float(last[2])
        acc_train = float(last[3])
        acc_val = float(last[4])

    test_pattern = re.compile(r"Test\s*\|\s*loss\s*([0-9.]+)\s*acc\s*([0-9.]+)", re.MULTILINE)
    m_test = test_pattern.search(log_text)
    if m_test:
        test_loss = float(m_test.group(1))
        test_acc = float(m_test.group(2))
    else:
        test_loss = 0.0
        test_acc = 0.0

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
    Constructs the model save path and passes it to the subprocess.
    """
    (
        run_id, fm_kind, n_wires, depth, reps, pad_mode, fm_ent, fm_op, 
        ql_ent, ql_op, ql_rev, 
        script_path_str, base_dir_str, gpu_queue
    ) = args

    gpu_id = gpu_queue.get() 

    try:
        script_path = Path(script_path_str)
        base_dir = Path(base_dir_str)
        
        # --- MODEL PATH CONSTRUCTION (FOR TARGET SCRIPT) ---
        # Structure: base_dir/models/fm_kind/ql_op/id_{run_id}_{params}.pt
        model_save_dir = base_dir / "models" / fm_kind / ql_op
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique filename based on key parameters
        model_filename = f"id{run_id}_w{n_wires}d{depth}_qle{ql_ent[0]}{ql_ent[-1]}_qlo{ql_op}.pt"
        target_model_path = str(model_save_dir / model_filename)

        # --- LOG PATH CONSTRUCTION ---
        kind_dir = base_dir / fm_kind / ql_op
        kind_dir.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

        # --- BUILD COMMAND ---
        cmd = [sys.executable, str(script_path)] + BASE_ARGS + [
            # 1. MANDATORY RUN ID
            "--run-id", str(run_id),
            
            # 2. TARGET MODEL PATH (Passes explicit save path)
            "--target-model-path", target_model_path,
            
            # 3. FM TOPOLOGY
            "--fm-kind", fm_kind,
            "--n-wires", str(n_wires),
            "--depth", str(depth),
            
            # 4. Q-LAYER ARGS
            "--qlayer-ent-kind", ql_ent,
            "--qlayer-twoq-op", ql_op
        ]

        if ql_rev:
            cmd.append("--qlayer-ent-wire-reverse")

        # --- ADD FM-SPECIFIC FLAGS ---
        log_suffix = f"_w{n_wires}d{depth}r{reps}p{pad_mode[0]}"
        
        if fm_kind == "z":
            cmd += ["--fm-z-reps", str(reps), "--fm-z-pad-mode", pad_mode]
            
        elif fm_kind == "zz":
            cmd += ["--fm-zz-reps", str(reps), "--fm-zz-pad-mode", pad_mode, "--fm-zz-entanglement", fm_ent]
            log_suffix += f"_fe{fm_ent}"
            
        elif fm_kind == "eff_su2":
            cmd += ["--fm-eff-reps", str(reps), "--fm-eff-pad-mod", pad_mode, "--fm-eff-ent-kind", fm_ent, "--fm-eff-twoq-op", fm_op]
            log_suffix += f"_fe{fm_ent}_fo{fm_op}"

        # Append Q-Layer specifics to log name
        rev_str = "R" if ql_rev else "N"
        log_suffix += f"_qe{ql_ent[0]}{ql_ent[-1]}_qo{ql_op}_qr{rev_str}"

        log_name = f"id{run_id}{log_suffix}.txt"
        log_path = kind_dir / log_name

        # --- ENVIRONMENT SETTINGS ---
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
            raise RuntimeError(f"Return Code {result.returncode}. Check log: {log_path}")

        metrics = parse_last_epoch_and_test(result.stdout)
        return metrics
    
    finally:
        gpu_queue.put(gpu_id)


# ================= MAIN EXECUTION =================

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(f"sweep_full_pipeline_reduced_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- STARTING FULL PIPELINE SWEEP (Reduced Q-Layer) ---")
    print(f"Output Directory: {base_dir}")

    # 2. GENERATE JOB LIST
    jobs = []
    run_counter = 1

    # A. Q-Layer Combinations
    qlayer_combos = list(itertools.product(QLAYER_ENT_KINDS, QLAYER_TWOQ_OPS, QLAYER_REVERSE))
    
    # B. Base Topology
    base_combos = list(itertools.product(N_WIRES_LIST, DEPTHS, REPS_LIST, PAD_MODES))

    print("Generating Configuration List...")

    for ql_ent, ql_op, ql_rev in qlayer_combos:
        for wires, depth, reps, pad in base_combos:
            
            # --- FM: Z ---
            fm = "z"
            jobs.append((
                run_counter, fm, wires, depth, reps, pad, "NA", "NA", # FM specific
                ql_ent, ql_op, ql_rev, # Q-Layer specific
                str(SCRIPT_PATH), str(base_dir)
            ))
            run_counter += 1
            
            # --- FM: ZZ ---
            fm = "zz"
            for fm_ent in FM_ENTANGLEMENTS:
                jobs.append((
                    run_counter, fm, wires, depth, reps, pad, fm_ent, "NA",
                    ql_ent, ql_op, ql_rev,
                    str(SCRIPT_PATH), str(base_dir)
                ))
                run_counter += 1
                
            # --- FM: EFF_SU2 ---
            fm = "eff_su2"
            for fm_ent in FM_ENTANGLEMENTS:
                for fm_op in REDUCED_FM_EFF_TWOQ_OPS:
                    jobs.append((
                        run_counter, fm, wires, depth, reps, pad, fm_ent, fm_op,
                        ql_ent, ql_op, ql_rev,
                        str(SCRIPT_PATH), str(base_dir)
                    ))
                    run_counter += 1

    total_runs = len(jobs)
    
    N_QLAYER = len(qlayer_combos)
    N_BASE = len(base_combos)
    
    print(f"Total Q-Layer Combinations (3x4x2): {N_QLAYER}")
    print(f"Total Configurations: {total_runs}")
    print("This is a major reduction in complexity while still maintaining deep coverage.")

    # 3. GPU SETUP
    available_gpus = get_free_gpus(min_free_mem_mb=10000)
    if not available_gpus:
        print("NO GPUs FOUND! CPU Mode.")
        gpu_tickets = [""]
        max_workers = 1
    else:
        print(f"GPUs Found: {available_gpus}")
        gpu_tickets = []
        for gpu in available_gpus:
            for _ in range(JOBS_PER_GPU):
                gpu_tickets.append(gpu)
        random.shuffle(gpu_tickets)
        max_workers = len(gpu_tickets)

    m = Manager()
    gpu_queue = m.Queue()
    for t in gpu_tickets: gpu_queue.put(t)

    # 4. EXECUTION
    csv_path = base_dir / "master_results_full_pipeline_moons.csv"
    errors_path = base_dir / "errors.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
            errors_path.open("w", encoding="utf-8") as f_err:

        fieldnames = [
            "run_id", "fm_kind", "n_wires", "depth", "reps", "pad_mode", "fm_ent", "fm_op",
            "ql_ent", "ql_op", "ql_rev",
            "last_epoch", "acc_train", "acc_test", "acc_val", 
            "loss_train", "loss_val", "test_loss", "status", "error_msg"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            jobs_with_queue = [job + (gpu_queue,) for job in jobs]
            futures = {executor.submit(run_config, j): j for j in jobs_with_queue}

            completed = 0
            for future in as_completed(futures):
                job_data = futures[future]
                rid, fm, w, d, r, p, fe, fo, qe, qo, qr = job_data[:11]
                
                rev_mark = "R" if qr else "_"
                cfg_str = f"[{fm}] w{w} d{d} QL:{qe}/{qo}/{rev_mark}"

                try:
                    metrics = future.result()
                    status = "ok"
                    err_msg = ""
                    print(f"[OK] {rid} {cfg_str} -> Test: {metrics.get('test_acc')}")
                except Exception as e:
                    status = "error"
                    err_msg = str(e)
                    print(f"[ERR] {rid} {cfg_str} -> {err_msg}")
                    f_err.write(f"{cfg_str}\n{err_msg}\n")
                    metrics = {}

                row = {
                    "run_id": rid, "fm_kind": fm, "n_wires": w, "depth": d, 
                    "reps": r, "pad_mode": p, "fm_ent": fe, "fm_op": fo,
                    "ql_ent": qe, "ql_op": qo, "ql_rev": qr,
                    "last_epoch": metrics.get("last_epoch", ""),
                    "acc_train": metrics.get("acc_train", ""),
                    "acc_test": metrics.get("test_acc", ""),
                    "acc_val": metrics.get("acc_val", ""),
                    "loss_train": metrics.get("loss_train", ""),
                    "loss_val": metrics.get("loss_val", ""),
                    "test_loss": metrics.get("test_loss", ""),
                    "status": status, "error_msg": err_msg
                }
                writer.writerow(row)
                f_csv.flush()

                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    if rate > 0:
                        remaining = (total_runs - completed) / rate
                        rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                    else:
                        rem_str = "Unknown"
                    print(f"Progress: {completed}/{total_runs} ({completed/total_runs:.2%}) - Est Rem: {rem_str}")

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"=== SWEEP COMPLETE in {total_time} ===")

if __name__ == "__main__":
    main()