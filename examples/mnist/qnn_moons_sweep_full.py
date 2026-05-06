import subprocess
import sys
import csv
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================
# CONFIG — EDIT THESE
# =====================================================

PYTHON_EXE = "python.exe"
MAIN_SCRIPT = r".\examples\mnist\mnist_2qubit_4class.py"

# Base arguments that do NOT change
BASE_ARGS = [
    "--model-type", "qnn",
    "--random-ops", "0",
    "--dataset", "moons",
    "--vector-train", "200",
    "--vector-valid", "2000",
    "--vector-test", "2000",
    "--epochs", "100",
    # "--plot-vector",
    "--train_target",
]

# Hyperparameter sweep
N_WIRES_LIST = [2, 4, 5]       # change as you like
DEPTH_LIST = [2, 5]
MOONS_NOISE_LIST = [0.1, 0.3, 0.4]
BATCH_SIZE_LIST = [8, 16, 32]

# How many experiments to run in parallel.
# With a single GPU, start with 1–2. If you see OOM, reduce it to 1.
MAX_PARALLEL_JOBS = 4

# Output
OUT_DIR = Path("sweeps_parallel")
OUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = OUT_DIR / f"qnn_moons_sweep_parallel_{timestamp}.csv"

# =====================================================
# Regex patterns to parse logs
# =====================================================

# Matches lines like:
#  23 |       0.3709 / 0.3534 |       0.850 / 0.853
EPOCH_LINE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([\d.]+)\s*/\s*([\d.]+)\s*\|\s*([\d.]+)\s*/\s*([\d.]+)"
)

# Matches line like:
# Without Saving: Test  | loss 0.3478 acc 0.859
TEST_LINE_RE = re.compile(
    r"Test\s*\|\s*loss\s*([\d.]+)\s*acc\s*([\d.]+)"
)

# =====================================================
# Worker function (runs in separate processes)
# =====================================================


def run_experiment_worker(config):
    """
    Worker that:
      - runs one training process as a subprocess,
      - streams logs to stdout,
      - collects lines for parsing,
      - writes a per-run log file,
      - returns parsed metrics.
    """
    (run_idx, total_runs, n_wires, depth, noise, batch_size) = config

    cmd = [
        PYTHON_EXE,
        MAIN_SCRIPT,
        *BASE_ARGS,
        "--n-wires", str(n_wires),
        "--depth", str(depth),
        "--moons-noise", str(noise),
        "--batch-size", str(batch_size),
    ]

    header = (
        f"\n================ RUN {run_idx}/{total_runs} ================\n"
        f"wires={n_wires}, depth={depth}, noise={noise}, batch_size={batch_size}\n"
        f"CMD: {' '.join(map(str, cmd))}\n"
        f"===========================================================\n"
    )
    print(header, flush=True)

    all_lines = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
    )

    assert proc.stdout is not None
    # Stream output line-by-line
    for line in proc.stdout:
        line = line.rstrip("\n")
        all_lines.append(line)
        # Live progress (will interleave with other runs)
        print(f"[RUN {run_idx}] {line}", flush=True)

    proc.wait()

    stdout = "\n".join(all_lines)

    # Save raw log
    raw_log_name = (
        f"log_run{run_idx}_w{n_wires}_d{depth}_noise{noise}_bs{batch_size}.txt"
    )
    (OUT_DIR / raw_log_name).write_text(stdout, encoding="utf-8")

    # Parse epochs + test
    per_epoch = []
    final_test = {"test_loss": None, "test_acc": None}

    for line in all_lines:
        m = EPOCH_LINE_RE.match(line)
        if m:
            epoch = int(m.group(1))
            loss_train = float(m.group(2))
            loss_val = float(m.group(3))
            acc_train = float(m.group(4))
            acc_val = float(m.group(5))
            per_epoch.append(
                dict(
                    epoch=epoch,
                    loss_train=loss_train,
                    loss_val=loss_val,
                    acc_train=acc_train,
                    acc_val=acc_val,
                )
            )
            continue

        t = TEST_LINE_RE.search(line)
        if t:
            final_test["test_loss"] = float(t.group(1))
            final_test["test_acc"] = float(t.group(2))

    return dict(
        run_idx=run_idx,
        n_wires=n_wires,
        depth=depth,
        noise=noise,
        batch_size=batch_size,
        per_epoch=per_epoch,
        final_test=final_test,
    )


# =====================================================
# Main
# =====================================================


def main():
    # Build list of all configs
    configs = []
    run_idx = 0
    for n_wires in N_WIRES_LIST:
        for depth in DEPTH_LIST:
            for noise in MOONS_NOISE_LIST:
                for bs in BATCH_SIZE_LIST:
                    run_idx += 1
                    configs.append((run_idx, None, n_wires, depth, noise, bs))

    total_runs = len(configs)
    # Fill in total_runs in each config tuple
    configs = [
        (run_idx, total_runs, n_wires, depth, noise, bs)
        for (run_idx, _, n_wires, depth, noise, bs) in configs
    ]

    print(f"Total runs: {total_runs}")
    print(f"Running with up to {MAX_PARALLEL_JOBS} in parallel.\n")

    fieldnames = [
        "run_id",
        "n_wires",
        "depth",
        "moons_noise",
        "batch_size",
        "epoch",
        "loss_train",
        "loss_val",
        "acc_train",
        "acc_val",
        "test_loss",
        "test_acc",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
            future_to_cfg = {
                executor.submit(run_experiment_worker, cfg): cfg for cfg in configs
            }

            for future in as_completed(future_to_cfg):
                result = future.result()
                run_id = result["run_idx"]
                n_wires = result["n_wires"]
                depth = result["depth"]
                noise = result["noise"]
                bs = result["batch_size"]
                per_epoch = result["per_epoch"]
                final_test = result["final_test"]

                if not per_epoch:
                    print(f"[WARN] No epoch data parsed for run {run_id}")
                    continue

                for ep in per_epoch:
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "n_wires": n_wires,
                            "depth": depth,
                            "moons_noise": noise,
                            "batch_size": bs,
                            "epoch": ep["epoch"],
                            "loss_train": ep["loss_train"],
                            "loss_val": ep["loss_val"],
                            "acc_train": ep["acc_train"],
                            "acc_val": ep["acc_val"],
                            "test_loss": final_test["test_loss"],
                            "test_acc": final_test["test_acc"],
                        }
                    )

                print(
                    f"\n>>> [DONE] run {run_id}/{total_runs} "
                    f"(wires={n_wires}, depth={depth}, noise={noise}, bs={bs})"
                )
                if final_test["test_acc"] is not None:
                    print(
                        f"    Final test: loss={final_test['test_loss']:.4f}, "
                        f"acc={final_test['test_acc']:.3f}"
                    )

    print("\nAll runs completed.")
    print(f"Combined CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
