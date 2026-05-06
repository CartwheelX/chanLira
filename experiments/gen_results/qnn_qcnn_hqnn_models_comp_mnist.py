"""
Paper-ready results generator with ENHANCED STYLING:
- MNIST: QNN vs QCNN vs HQNN
  (1) Gap violin plot (styled)
  (2) Matched run_id train/test bar chart GROUPED BY ARCHITECTURE
      → [Baseline QNN, Stress QNN, Hard QNN] | [Baseline HQNN, ...] | [Baseline QCNN, ...]
      CLEAN VERSION: Simple labels, scenario-based legend
  (3) Summary CSV
  (4) Matched run_id LaTeX table (long format)

- Synthetic (Moons/Blobs/Circles): QNN-only
  Auto-picks 3 targets per dataset:
    baseline = max acc_test
    stress   = max gap_acc among acc_test >= threshold
    hard     = min |gap_acc| among top-k acc_test runs within threshold
  Exports CSV + LaTeX

Outputs saved to:
  .\experiments\gen_results\paper_arch_compare\
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =========================
# INPUT FILES (as given)
# =========================
from pathlib import Path

MNIST_FILES = {
    "QNN":  "experiments/gen_results/mnist_extensive_results_updated.csv",
    "HQNN": "experiments/gen_results/hqnn_extensive_results.csv",
    "QCNN": "experiments/gen_results/qcnn_extensive_results.csv",
}

SYNTHETIC_FILES = {
    "Moons":   "experiments/gen_results/master_results_full_pipeline_moon.csv",
    "Blobs":   "experiments/gen_results/master_results_full_pipeline_blobs.csv",
    "Circles": "experiments/gen_results/master_results_full_pipeline_circles.csv",
}

OUT_DIR = Path("experiments/gen_results/paper_arch_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# =========================
# ENHANCED STYLE CONFIGURATION
# =========================
# Professional color palette
MODEL_COLORS = {
    'QNN': '#E74C3C',      # Red
    'QCNN': '#2ECC71',     # Green
    'HQNN': '#3498DB',     # Blue
}

SCENARIO_COLORS = {
    'baseline': '#27AE60',  # Dark green
    'stress': '#E67E22',    # Orange
    'hard': '#9B59B6',      # Purple
}

DATASET_COLORS = {
    'Moons': '#E91E63',     # Pink
    'Blobs': '#2196F3',     # Blue
    'Circles': '#FF9800',   # Orange
}

# Font sizes
TITLE_FS = 28
LABEL_FS = 24
TICK_FS = 20
LEGEND_FS = 18

# Update matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    # "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 2,
    # "axes.titleweight": "bold",
    # "axes.labelweight": "bold",
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "grid.alpha": 0.3,
    "grid.linewidth": 1,
})
# this for the good of us

# =========================
# HELPERS
# =========================
def load_ok(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Filter OK runs if available
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    # Required columns
    required = {"run_id", "acc_train", "acc_test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns {} in {}".format(missing, csv_path))

    # Compute gaps
    df["gap_acc"] = df["acc_train"] - df["acc_test"]

    # Optional: loss gap if present
    loss_train_candidates = ["loss_train", "train_loss"]
    loss_test_candidates  = ["test_loss", "loss_test", "loss_val", "val_loss"]
    lt = next((c for c in loss_train_candidates if c in df.columns), None)
    lte = next((c for c in loss_test_candidates if c in df.columns), None)
    if lt and lte:
        df["gap_loss"] = df[lt] - df[lte]

    return df

def safe_get(row: pd.Series, col: str, default: str = "NA"):
    if col not in row.index:
        return default
    v = row[col]
    if pd.isna(v):
        return default
    return v

def _fmt_int(v):
    try:
        s = str(v).strip().lower()
        if s in {"na", "nan"}:
            return "NA"
        return str(int(float(v)))
    except Exception:
        return str(v)

def config_string(row: pd.Series, is_mnist: bool) -> str:
    """
    MNIST: pad_mode and fm_ent are fixed in your sweep (wrap, linear),
    may not be present in CSV; we still print them for paper clarity.
    """
    parts = [
        "fm_kind={}".format(safe_get(row, "fm_kind")),
        "n_wires={}".format(_fmt_int(safe_get(row, "n_wires"))),
        "reps={}".format(_fmt_int(safe_get(row, "reps"))),
    ]

    if is_mnist:
        parts += ["pad_mode=wrap", "fm_ent=linear"]
    else:
        parts += [
            "pad_mode={}".format(safe_get(row, "pad_mode")),
            "fm_ent={}".format(safe_get(row, "fm_ent")),
        ]

    parts += [
        "fm_op={}".format(safe_get(row, "fm_op")),
        "depth={}".format(_fmt_int(safe_get(row, "depth"))),
        "ql_ent={}".format(safe_get(row, "ql_ent")),
        "ql_op={}".format(safe_get(row, "ql_op")),
    ]
    return ", ".join(parts)

def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_runs": int(len(df)),
        "max_acc_test": float(df["acc_test"].max()),
        "mean_acc_test": float(df["acc_test"].mean()),
        "median_gap_acc": float(df["gap_acc"].median()),
        "mean_gap_acc": float(df["gap_acc"].mean()),
        "frac_acc_test_lt_0p60": float((df["acc_test"] < 0.60).mean()),
    }

def get_row_by_runid(df: pd.DataFrame, run_id: int) -> pd.Series:
    r = df.loc[df["run_id"] == run_id]
    if r.empty:
        raise ValueError("run_id={} not found in dataframe".format(run_id))
    return r.iloc[0]

# =========================
# SYNTH TARGET PICKER (baseline + stress + hard)
# =========================
def pick_synth_targets(df: pd.DataFrame,
                       min_test: Optional[float] = None,
                       topk_for_hard: int = 200) -> Dict[str, Any]:
    """
    baseline: max acc_test
    stress:   max gap_acc among runs with acc_test >= threshold
    hard:     min |gap_acc| among top-k highest acc_test runs within threshold
    """
    baseline = df.loc[df["acc_test"].idxmax()]

    # "credible" threshold to avoid choosing trivial low-test runs
    if min_test is None:
        thr = max(0.60, float(df["acc_test"].quantile(0.75)))
    else:
        thr = float(min_test)

    cand = df[df["acc_test"] >= thr].copy()
    if cand.empty:
        # relax if needed
        thr = max(0.55, float(df["acc_test"].quantile(0.60)))
        cand = df[df["acc_test"] >= thr].copy()
    if cand.empty:
        cand = df.copy()

    # stress: maximize gap among credible
    stress = cand.loc[cand["gap_acc"].idxmax()]

    # hard: minimize |gap| but keep test high
    hard_pool = cand.sort_values("acc_test", ascending=False).head(topk_for_hard)
    hard = hard_pool.loc[hard_pool["gap_acc"].abs().idxmin()]

    return {"baseline": baseline, "stress": stress, "hard": hard, "threshold_used": thr}

# =========================
# LOAD DATA
# =========================
print("\n" + "="*80)
print("📊 LOADING DATA")
print("="*80)

mnist = {}
for k, p in MNIST_FILES.items():
    mnist[k] = load_ok(p)
    print(f"✓ Loaded {k}: {len(mnist[k])} runs")

synthetic = {}
for k, p in SYNTHETIC_FILES.items():
    synthetic[k] = load_ok(p)
    print(f"✓ Loaded {k}: {len(synthetic[k])} runs")

# =========================
# (1) MNIST gap violin - ENHANCED STYLING
# =========================
print("\n" + "="*80)
print("🎻 GENERATING MNIST GAP VIOLIN PLOT")
print("="*80)

labels = ["QNN", "QCNN", "HQNN"]
data = [
    mnist["QNN"]["gap_acc"].values,
    mnist["QCNN"]["gap_acc"].values,
    mnist["HQNN"]["gap_acc"].values,
]

fig = plt.figure(figsize=(12, 7), facecolor="white")
ax = fig.add_subplot(111)

# Create violin plot with custom colors
positions = [1, 2, 3]
parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True, widths=0.7)

# Color each violin
colors = [MODEL_COLORS['QNN'], MODEL_COLORS['QCNN'], MODEL_COLORS['HQNN']]
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(2)

# Style the other elements
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    if partname in parts:
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)

# Add statistics annotations
for i, (label, d) in enumerate(zip(labels, data), 1):
    median = np.median(d)
    mean = np.mean(d)
    ax.text(i, ax.get_ylim()[1] * 0.95, 
            f'μ={mean:.3f}\nM={median:.3f}',
            ha='center', va='top', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=colors[i-1], linewidth=2, alpha=0.9))

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=TICK_FS+2, fontweight='bold')
ax.set_ylabel("Generalization Gap (Train Acc − Test Acc)", fontsize=LABEL_FS)
ax.set_title("MNIST: Generalization Gap Distribution Across Architectures", 
             fontsize=TITLE_FS, fontweight='bold', pad=20)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.grid(True, axis="y", alpha=0.15, linestyle='-', linewidth=0.8)
ax.set_facecolor('white')

# Professional spine styling: only left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_edgecolor('black')
    ax.spines[spine].set_linewidth(1.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_mnist_gap_violin.png", dpi=300, bbox_inches="tight", facecolor='white')
fig.savefig(OUT_DIR / "fig_mnist_gap_violin.pdf", bbox_inches="tight", facecolor='white')
plt.close(fig)
print("✅ Saved: fig_mnist_gap_violin.png/pdf")

# =========================
# (2) MNIST matched run_id bars + LaTeX table - CLEAN VERSION
# =========================
print("\n" + "="*80)
print("📊 GENERATING MNIST MATCHED RUN_ID COMPARISON (CLEAN VERSION)")
print("="*80)

RUN_IDS_MNIST = {
    "baseline": 1102,
    "stress":   647,
    "hard":     1456,
}

rows = []
for scen, rid in RUN_IDS_MNIST.items():
    for arch_key, arch_label in [("QNN", "QNN"), ("QCNN", "QCNN"), ("HQNN", "HQNN")]:
        r = get_row_by_runid(mnist[arch_key], rid)
        rows.append({
            "scenario": scen,
            "run_id": int(rid),
            "architecture": arch_label,
            "acc_train": float(r["acc_train"]),
            "acc_test": float(r["acc_test"]),
            "gap_acc": float(r["gap_acc"]),
            "config": config_string(r, is_mnist=True),
        })

mnist_matched = pd.DataFrame(rows)
mnist_matched.to_csv(OUT_DIR / "mnist_matched_runids_table.csv", index=False)
mnist_matched.to_latex(OUT_DIR / "mnist_matched_runids_table.tex", index=False, escape=False)
print("✅ Saved: mnist_matched_runids_table.csv/tex")

# =========================
# REORGANIZE: Group by architecture - CLEAN VERSION
# =========================
plot_df = mnist_matched.copy()

# Create ordering: architecture first, then scenario
order = []
for arch in ["QNN", "HQNN", "QCNN"]:  # Architecture groups
    for scen in ["baseline", "stress", "hard"]:  # Scenarios within each architecture
        order.append((scen, arch))

plot_df = plot_df.set_index(["scenario", "architecture"]).loc[order].reset_index()

x = np.arange(len(plot_df))
width = 0.35

fig = plt.figure(figsize=(16, 7), facecolor="white")
ax = fig.add_subplot(111)

# Create bars with scenario-specific colors
train_colors = [SCENARIO_COLORS[row['scenario']] for _, row in plot_df.iterrows()]
test_colors = train_colors

train_bars = ax.bar(x - width/2, plot_df["acc_train"].values, width, 
                    color=train_colors, alpha=0.7,
                    edgecolor='black', linewidth=2)
test_bars = ax.bar(x + width/2, plot_df["acc_test"].values, width, 
                   color=test_colors, alpha=0.9,
                   edgecolor='black', linewidth=2, hatch='//')

# CLEAN X-axis labels: Just scenario names
ax.set_xticks(x)
xlabels = []
for _, r in plot_df.iterrows():
    scen_label = r["scenario"].capitalize()
    xlabels.append(scen_label)  # Simple: just "Baseline", "Stress", "Hard"
ax.set_xticklabels(xlabels, fontsize=TICK_FS)

ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Accuracy", fontsize=LABEL_FS)
# ax.set_title("MNIST: Train vs Test Accuracy Across Architectures\n(Grouped by Model: QNN, HQNN, QCNN)", 
            #  fontsize=TITLE_FS, fontweight='bold', pad=20)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.grid(True, axis="y", alpha=0.15, linestyle='-', linewidth=0.8)

# CLEAN Separators: Just lines, no annotations
separator_positions = [2.5, 5.5]
for sep in separator_positions:
    ax.axvline(sep, color='black', linewidth=2.5, alpha=0.5, linestyle='--')

# CLEAN Architecture labels: No color boxes, just black text
arch_positions = [1, 4, 7]  # Middle of each group
for pos, arch in zip(arch_positions, ["QNN", "HQNN", "QCNN"]):
    ax.text(pos, -0.12, arch, transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=TITLE_FS-2,
            color='black')  # Just black text, no colored boxes

# UPDATED LEGEND: Show scenarios + pattern for train/test
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    # Scenario colors
    Patch(facecolor=SCENARIO_COLORS['baseline'], edgecolor='black', linewidth=2, 
          label='Baseline', alpha=0.7),
    Patch(facecolor=SCENARIO_COLORS['stress'], edgecolor='black', linewidth=2, 
          label='Stress', alpha=0.7),
    Patch(facecolor=SCENARIO_COLORS['hard'], edgecolor='black', linewidth=2, 
          label='Hard', alpha=0.7),
    # Train/Test distinction
    Patch(facecolor='gray', edgecolor='black', linewidth=2, 
          label='Train (solid)', alpha=0.7),
    Patch(facecolor='gray', edgecolor='black', linewidth=2, hatch='//',
          label='Test (hatched)', alpha=0.9),
]

ax.legend(handles=legend_elements, fontsize=LEGEND_FS, loc='lower right', 
          frameon=True, fancybox=True, shadow=True, framealpha=0.95,
          ncol=2, title='Scenario & Data Split', title_fontsize=LEGEND_FS)

# Professional spine styling: only left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_edgecolor('black')
    ax.spines[spine].set_linewidth(1.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_mnist_matched_runids_train_test_bars.png", dpi=300, bbox_inches="tight", facecolor='white')
fig.savefig(OUT_DIR / "fig_mnist_matched_runids_train_test_bars.pdf", bbox_inches="tight", facecolor='white')
plt.close(fig)
print("✅ Saved: fig_mnist_matched_runids_train_test_bars.png/pdf")

# =========================
# (3) MNIST architecture summary CSV
# =========================
print("\n" + "="*80)
print("📋 GENERATING MNIST ARCHITECTURE SUMMARY")
print("="*80)

summary = pd.DataFrame([
    {"architecture": "QNN",  **summarize(mnist["QNN"])},
    {"architecture": "QCNN", **summarize(mnist["QCNN"])},
    {"architecture": "HQNN", **summarize(mnist["HQNN"])},
])
summary.to_csv(OUT_DIR / "mnist_arch_summary.csv", index=False)
print("✅ Saved: mnist_arch_summary.csv")

# =========================
# (4) Synthetic targets: baseline + stress + hard (AUTO)
# =========================
print("\n" + "="*80)
print("🎯 AUTO-SELECTING SYNTHETIC DATASET TARGETS")
print("="*80)

# Optional: per-dataset minimum test threshold override
SYNTH_MIN_TEST = {
    # "Moons": 0.80,
    # "Blobs": 0.80,
    # "Circles": 0.75,
}

synth_rows = []

for ds_name, df in synthetic.items():
    print(f"\n🔍 Processing {ds_name}...")
    picks = pick_synth_targets(df, min_test=SYNTH_MIN_TEST.get(ds_name))
    thr_used = picks["threshold_used"]

    for role in ["baseline", "stress", "hard"]:
        r = picks[role]
        print(f"  ✓ {role.capitalize()}: run_id={int(r['run_id'])}, "
              f"test_acc={r['acc_test']:.4f}, gap={r['gap_acc']:.4f}")
        
        synth_rows.append({
            "dataset": ds_name,
            "role": role,
            "run_id": int(r["run_id"]),
            "acc_train": float(r["acc_train"]),
            "acc_test": float(r["acc_test"]),
            "gap_acc": float(r["gap_acc"]),
            "selection_rule": (
                "baseline=max acc_test" if role == "baseline" else
                "stress=max gap_acc (acc_test≥{:.3f})".format(thr_used) if role == "stress" else
                "hard=min |gap_acc| (top-k acc_test, acc_test≥{:.3f})".format(thr_used)
            ),
            "config": config_string(r, is_mnist=False),
        })

synthetic_targets = pd.DataFrame(synth_rows)
synthetic_targets.to_csv(OUT_DIR / "synthetic_qnn_targets_table.csv", index=False)
synthetic_targets.to_latex(OUT_DIR / "synthetic_qnn_targets_table.tex", index=False, escape=False)
print("\n✅ Saved: synthetic_qnn_targets_table.csv/tex")

# =========================
# (5) Synthetic target bar plots - CLEAN VERSION (MNIST-CONSISTENT)
# =========================
print("\n" + "="*80)
print("📊 GENERATING SYNTHETIC DATASET TARGET PLOTS (CLEAN VERSION)")
print("="*80)

from matplotlib.patches import Patch

for ds_name in synthetic_targets["dataset"].unique():
    print(f"  Plotting {ds_name}...")
    sub = synthetic_targets[synthetic_targets["dataset"] == ds_name].copy()
    sub = sub.set_index("role").loc[["baseline", "stress", "hard"]].reset_index()

    x = np.arange(len(sub))
    width = 0.35

    fig = plt.figure(figsize=(11, 6.5), facecolor="white")
    ax = fig.add_subplot(111)

    # Scenario colors (same as MNIST)
    bar_colors = [SCENARIO_COLORS[role] for role in sub["role"]]

    train_bars = ax.bar(
        x - width/2,
        sub["acc_train"].values,
        width,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    test_bars = ax.bar(
        x + width/2,
        sub["acc_test"].values,
        width,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=2,
        hatch="//",
    )

    # CLEAN x-axis: scenario names only
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["role"].capitalize() for _, r in sub.iterrows()],
        fontsize=TICK_FS+8,
        # fontweight='bold'
    )

    ax.set_ylim(0.0, 1.08)
    # ax.set_xlabel("Scenario", fontsize=LABEL_FS+4)
    ax.set_ylabel("Accuracy", fontsize=LABEL_FS+6)
    # Title removed for cleaner side-by-side layout

    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(True, axis="y", alpha=0.15, linestyle='-', linewidth=0.8)

    # Legend: SAME as MNIST (semantic, clean)
    legend_elements = [
        Patch(facecolor=SCENARIO_COLORS["baseline"], edgecolor="black", linewidth=2,
              label="Baseline"),
        Patch(facecolor=SCENARIO_COLORS["stress"], edgecolor="black", linewidth=2,
              label="Stress"),
        Patch(facecolor=SCENARIO_COLORS["hard"], edgecolor="black", linewidth=2,
              label="Hard"),
        Patch(facecolor="gray", edgecolor="black", linewidth=2,
              label="Train (solid)", alpha=0.7),
        Patch(facecolor="gray", edgecolor="black", linewidth=2, hatch="//",
              label="Test (hatched)", alpha=0.9),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_FS+6,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        ncol=2,
        title="Scenario & Data Split",
        title_fontsize=LEGEND_FS+6,
    )

    # Professional spine styling: only left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_edgecolor('black')
        ax.spines[spine].set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"fig_{ds_name.lower()}_targets_train_test_bars.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT_DIR / f"fig_{ds_name.lower()}_targets_train_test_bars.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

print("✅ Saved all synthetic target plots (MNIST-consistent style)")


# =========================
# (6) BONUS: Combined synthetic overview plot
# =========================
print("\n" + "="*80)
print("🎨 GENERATING COMBINED SYNTHETIC OVERVIEW")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')

for idx, ds_name in enumerate(['Moons', 'Blobs', 'Circles']):
    ax = axes[idx]
    sub = synthetic_targets[synthetic_targets["dataset"] == ds_name].copy()
    sub = sub.set_index("role").loc[["baseline", "stress", "hard"]].reset_index()

    x = np.arange(len(sub))
    width = 0.35

    train_colors = [SCENARIO_COLORS[role] for role in sub['role']]
    
    ax.bar(x - width/2, sub["acc_train"].values, width, 
           label="Train", color=train_colors, alpha=0.7,
           edgecolor='black', linewidth=2)
    ax.bar(x + width/2, sub["acc_test"].values, width, 
           label="Test", color=train_colors, alpha=0.9,
           edgecolor='black', linewidth=2, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels([r['role'].capitalize() for _, r in sub.iterrows()],
                       fontsize=TICK_FS-1, fontweight='bold')
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Accuracy" if idx == 0 else "", fontsize=LABEL_FS-2)
    ax.set_title(ds_name, fontsize=TITLE_FS-2, fontweight='bold', 
                color=DATASET_COLORS[ds_name], pad=15)
    ax.tick_params(axis='y', labelsize=TICK_FS-2)
    ax.grid(True, axis="y", alpha=0.15, linestyle='-', linewidth=0.8)
    
    if idx == 2:
        ax.legend(fontsize=LEGEND_FS-2, loc='lower right', frameon=True,
                 fancybox=True, shadow=True)
    
    # Professional spine styling: only left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_edgecolor('black')
        ax.spines[spine].set_linewidth(1.5)

plt.suptitle('Synthetic Datasets: Auto-Selected Configurations Overview',
            fontsize=TITLE_FS+2, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_synthetic_combined_overview.png", dpi=300, bbox_inches="tight", facecolor='white')
fig.savefig(OUT_DIR / "fig_synthetic_combined_overview.pdf", bbox_inches="tight", facecolor='white')
plt.close(fig)
print("✅ Saved: fig_synthetic_combined_overview.png/pdf")

# =========================
# (7) LaTeX include snippet
# =========================
latex_snippet = r"""
% Auto-generated figures/tables saved to:
% {out}

% ---- MNIST Figures ----
\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{{paper_arch_compare/fig_mnist_gap_violin.pdf}}
  \caption{{MNIST train--test accuracy gap distribution across QNN, QCNN, and HQNN architectures. 
           Violin plots show the full distribution with means (μ) and medians (M) annotated.}}
  \label{{fig:mnist_gap_violin}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{{paper_arch_compare/fig_mnist_matched_runids_train_test_bars.pdf}}
  \caption{{MNIST matched configurations grouped by architecture. Each group shows baseline, stress, 
           and hard scenarios with train vs. test accuracy comparison. Colors indicate scenario type: 
           green (baseline), orange (stress), purple (hard). Hatched bars represent test accuracy.}}
  \label{{fig:mnist_matched_bars}}
\end{{figure}}

% ---- Tables ----
% \input{{paper_arch_compare/mnist_matched_runids_table.tex}}
% \input{{paper_arch_compare/synthetic_qnn_targets_table.tex}}

% ---- Synthetic Figures ----
\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{{paper_arch_compare/fig_synthetic_combined_overview.pdf}}
  \caption{{Synthetic datasets (Moons, Blobs, Circles): Auto-selected baseline, stress, and hard 
           configurations showing train vs. test accuracy trade-offs.}}
  \label{{fig:synthetic_overview}}
\end{{figure}}

% Individual synthetic plots (optional):
% \includegraphics{{paper_arch_compare/fig_moons_targets_train_test_bars.pdf}}
% \includegraphics{{paper_arch_compare/fig_blobs_targets_train_test_bars.pdf}}
% \includegraphics{{paper_arch_compare/fig_circles_targets_train_test_bars.pdf}}
""".format(
    out=OUT_DIR.as_posix(),
)

(OUT_DIR / "latex_include_snippet.tex").write_text(latex_snippet.strip() + "\n", encoding="utf-8")
print("\n✅ Saved: latex_include_snippet.tex")

# =========================
# PRINT COMPREHENSIVE SUMMARY
# =========================
print("\n" + "="*80)
print("📊 COMPREHENSIVE SUMMARY")
print("="*80)

print("\n" + "─"*80)
print("MNIST ARCHITECTURE SUMMARY")
print("─"*80)
print(summary.to_string(index=False))

print("\n" + "─"*80)
print("MNIST MATCHED RUN_IDS (Grouped by Architecture)")
print("─"*80)
print(mnist_matched[["architecture", "scenario", "run_id", "acc_train", "acc_test", "gap_acc"]].to_string(index=False))

print("\n" + "─"*80)
print("SYNTHETIC AUTO-SELECTED TARGETS")
print("─"*80)
print(synthetic_targets[["dataset", "role", "run_id", "acc_train", "acc_test", "gap_acc"]].to_string(index=False))

print("\n" + "="*80)
print("📁 FILES CREATED")
print("="*80)
files_created = sorted(OUT_DIR.glob("*"))
for p in files_created:
    size_kb = p.stat().st_size / 1024
    print(f"  ✓ {p.name:<50} ({size_kb:>6.1f} KB)")

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📂 Output directory: {OUT_DIR.absolute()}")
print("\n💡 Key changes in CLEAN VERSION:")
print("   ✓ X-axis: Simple scenario names only (no run_id, no Δ)")
print("   ✓ Architecture labels: Plain black text (no colored boxes)")
print("   ✓ Separators: Clean lines only (no transition annotations)")
print("   ✓ Legend: Scenario colors + Train/Test patterns")
print("   ✓ Accuracy annotations: Still on top of bars")
print("\n" + "="*80 + "\n")