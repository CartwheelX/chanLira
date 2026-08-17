# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D
# from matplotlib.patches import Rectangle, Wedge
# from pathlib import Path
# import numpy as np
# from scipy import stats
# import warnings

# warnings.filterwarnings('ignore')

# # ═══════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ═══════════════════════════════════════════════════════════════

# SAVE_DIR = Path(r".\examples\mnist\gen_results")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# FILE_PATHS_WITH_LABELS = [
#     (r".\examples\mnist\gen_results\master_results_full_pipeline_moon.csv",    "Moons"),
#     (r".\examples\mnist\gen_results\master_results_full_pipeline_blobs.csv",   "Blobs"),
#     (r".\examples\mnist\gen_results\master_results_full_pipeline_circles.csv", "Circles"),
#     (r".\examples\mnist\gen_results\mnist_extensive_results_updated.csv",      "MNIST-QNN"),
#     (r".\examples\mnist\gen_results\hqnn_extensive_results.csv",              "MNIST-HQNN"),
#     (r".\examples\mnist\gen_results\qcnn_extensive_results.csv",              "MNIST-QCNN"),
# ]

# DATASET_ORDER = ['Moons', 'Blobs', 'Circles', 'MNIST-QNN', 'MNIST-HQNN', 'MNIST-QCNN']

# # Font sizes
# LABEL_FS = 22
# TICK_FS = 18
# TITLE_FS = 24
# LEGEND_FS = 16

# # Optional component color map (not used everywhere, but kept for reference)
# COMPONENT_COLORS = {
#     'n_wires': '#E74C3C',      # Red - High impact
#     'depth': '#3498DB',        # Blue - Medium impact
#     'reps': '#2ECC71',         # Green - Variable impact
#     'fm_kind': '#F39C12',      # Orange - Feature encoding
#     'interaction': '#9B59B6',  # Purple - Interactions
# }

# # ═══════════════════════════════════════════════════════════════
# # LOAD DATA
# # ═══════════════════════════════════════════════════════════════

# def load_all_data():
#     """Load and combine all datasets."""
#     all_data = []
#     for file_path, label in FILE_PATHS_WITH_LABELS:
#         try:
#             df = pd.read_csv(file_path)
#             df['dataset'] = label
#             all_data.append(df)
#             print(f"✓ Loaded: {label} ({len(df)} rows)")
#         except Exception as e:
#             print(f"✗ Error loading {label}: {e}")

#     if not all_data:
#         raise RuntimeError("No datasets could be loaded. Check file paths and CSVs.")

#     df = pd.concat(all_data, ignore_index=True)

#     # Generalization gap
#     df['generalization_gap'] = df['acc_train'] - df['acc_test']

#     # Handle column names for depth
#     if 'depth' not in df.columns:
#         if 'q_layers' in df.columns:
#             df['depth'] = df['q_layers']
#         elif 'layers' in df.columns:
#             df['depth'] = df['layers']

#     # Handle column names for n_wires
#     if 'n_wires' not in df.columns:
#         if 'n_qubits' in df.columns:
#             df['n_wires'] = df['n_qubits']
#         elif 'qubits' in df.columns:
#             df['n_wires'] = df['qubits']

#     print(f"\n📊 Total data: {len(df)} rows")
#     print(f"📋 Columns: {df.columns.tolist()}")
#     return df

# # ═══════════════════════════════════════════════════════════════
# # 1. COMPONENT CONTRIBUTION ANALYSIS (ANOVA/CORRELATION)
# # ═══════════════════════════════════════════════════════════════

# def analyze_component_contributions(df):
#     """Analyze contribution of each component to generalization gap."""
#     components = ['n_wires', 'depth', 'reps', 'fm_kind']
#     results = []

#     datasets = [ds for ds in DATASET_ORDER if ds in df['dataset'].unique()]

#     for dataset in datasets:
#         subset = df[df['dataset'] == dataset].copy()
#         if subset.empty:
#             continue

#         component_stats = {}

#         for comp in components:
#             if comp not in subset.columns:
#                 continue

#             if comp == 'fm_kind':
#                 # Categorical variable - ANOVA on groups
#                 groups = [group['generalization_gap'].values
#                           for _, group in subset.groupby(comp)]
#                 if len(groups) > 1:
#                     f_stat, p_value = stats.f_oneway(*groups)
#                     # Simple eta-squared style effect size
#                     effect_size = f_stat / (f_stat + len(subset) - len(groups))
#                 else:
#                     effect_size = 0.0
#             else:
#                 # Numerical variable - correlation magnitude
#                 corr = subset[[comp, 'generalization_gap']].corr().iloc[0, 1]
#                 effect_size = 0.0 if pd.isna(corr) else float(abs(corr))

#             component_stats[comp] = effect_size

#         # Interaction terms
#         if 'n_wires' in subset.columns and 'depth' in subset.columns:
#             subset['nwires_depth'] = subset['n_wires'] * subset['depth']
#             corr = subset[['nwires_depth', 'generalization_gap']].corr().iloc[0, 1]
#             component_stats['n_wires×depth'] = 0.0 if pd.isna(corr) else float(abs(corr))

#         if 'n_wires' in subset.columns and 'reps' in subset.columns:
#             subset['nwires_reps'] = subset['n_wires'] * subset['reps']
#             corr = subset[['nwires_reps', 'generalization_gap']].corr().iloc[0, 1]
#             component_stats['n_wires×reps'] = 0.0 if pd.isna(corr) else float(abs(corr))

#         if 'depth' in subset.columns and 'reps' in subset.columns:
#             subset['depth_reps'] = subset['depth'] * subset['reps']
#             corr = subset[['depth_reps', 'generalization_gap']].corr().iloc[0, 1]
#             component_stats['depth×reps'] = 0.0 if pd.isna(corr) else float(abs(corr))

#         results.append({
#             'dataset': dataset,
#             **component_stats
#         })

#     result_df = pd.DataFrame(results)
#     if not result_df.empty:
#         result_df = result_df.fillna(0.0)
#     return result_df

# # ═══════════════════════════════════════════════════════════════
# # 2. STACKED BAR CHART - COMPONENT CONTRIBUTIONS
# # ═══════════════════════════════════════════════════════════════

# def plot_component_contributions_stacked(df, save_name="component_contributions_stacked"):
#     """Create stacked bar chart showing component contributions."""
#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)

#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None, None

#     # Normalize numeric columns to percentages per dataset
#     numeric_cols = contrib_df.select_dtypes(include=[np.number]).columns
#     contrib_df[numeric_cols] = contrib_df[numeric_cols].div(
#         contrib_df[numeric_cols].sum(axis=1), axis=0
#     ).fillna(0.0) * 100.0

#     fig, ax = plt.subplots(figsize=(16, 10), facecolor='#F8F9FA')

#     datasets = contrib_df['dataset'].values
#     x = np.arange(len(datasets))
#     width = 0.6

#     components = [col for col in contrib_df.columns if col != 'dataset']
#     colors = plt.cm.Set3(np.linspace(0, 1, len(components)))

#     bottom = np.zeros(len(datasets))

#     for idx, comp in enumerate(components):
#         if comp in contrib_df.columns:
#             values = contrib_df[comp].values
#             bars = ax.bar(
#                 x, values, width, bottom=bottom,
#                 label=comp, color=colors[idx],
#                 edgecolor='white', linewidth=2
#             )

#             # Add percentage labels
#             for i, (bar, val) in enumerate(zip(bars, values)):
#                 if val > 5:  # Only show if > 5%
#                     height = bar.get_height()
#                     ax.text(
#                         bar.get_x() + bar.get_width() / 2.,
#                         bottom[i] + height / 2.,
#                         f'{val:.1f}%',
#                         ha='center', va='center',
#                         fontsize=12, fontweight='bold',
#                         color='black'
#                     )

#             bottom += values

#     ax.set_xlabel('Dataset', fontsize=LABEL_FS, fontweight='bold')
#     ax.set_ylabel('Contribution to Generalization Gap (%)',
#                   fontsize=LABEL_FS, fontweight='bold')
#     ax.set_title('Component Contribution Analysis\n(Privacy Vulnerability Indicators)',
#                  fontsize=TITLE_FS, fontweight='bold', pad=20)
#     ax.set_xticks(x)
#     ax.set_xticklabels(datasets, fontsize=TICK_FS, rotation=45, ha='right')
#     ax.tick_params(axis='y', labelsize=TICK_FS)

#     ax.legend(
#         loc='upper left', bbox_to_anchor=(1, 1),
#         fontsize=LEGEND_FS, frameon=True,
#         facecolor='white', edgecolor='black', framealpha=0.9
#     )

#     ax.set_facecolor('#F8F9FA')
#     ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

#     for spine in ax.spines.values():
#         spine.set_edgecolor('#2C3E50')
#         spine.set_linewidth(2)

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='#F8F9FA')
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='#F8F9FA')

#     print(f"✅ Saved: {png_path}")
#     plt.show()
#     return fig, contrib_df

# # ═══════════════════════════════════════════════════════════════
# # 3. RADAR CHART - COMPONENT VULNERABILITY PROFILE
# # ═══════════════════════════════════════════════════════════════

# def plot_vulnerability_radar(df, save_name="vulnerability_radar"):
#     """Create radar chart showing vulnerability profile per dataset."""
#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)
#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None

#     datasets = contrib_df['dataset'].values
#     components = [col for col in contrib_df.columns if col != 'dataset']

#     num_vars = len(components)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]  # complete circle

#     fig, axes = plt.subplots(
#         2, 3, figsize=(20, 14),
#         subplot_kw=dict(projection='polar'),
#         facecolor='#1A1A2E'
#     )
#     axes = axes.flatten()

#     colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(datasets)))

#     for idx, (dataset, ax) in enumerate(zip(datasets, axes)):
#         subset = contrib_df[contrib_df['dataset'] == dataset]
#         if subset.empty:
#             ax.axis('off')
#             continue

#         row = subset.iloc[0]
#         values = [float(row[comp]) for comp in components]
#         values += values[:1]

#         max_val = max(values) if max(values) > 0 else 1.0
#         values_norm = [v / max_val for v in values]

#         ax.plot(angles, values_norm, 'o-', linewidth=3,
#                 color=colors[idx], label=dataset)
#         ax.fill(angles, values_norm, alpha=0.25, color=colors[idx])

#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(components, fontsize=TICK_FS - 2, color='white')
#         ax.set_ylim(0, 1)
#         ax.set_yticks([0.25, 0.5, 0.75, 1.0])
#         ax.set_yticklabels(['25%', '50%', '75%', '100%'],
#                            fontsize=TICK_FS - 4, color='white')
#         ax.grid(True, color='white', alpha=0.3, linewidth=1)
#         ax.set_facecolor('#1A1A2E')
#         ax.set_title(dataset, fontsize=TITLE_FS, fontweight='bold',
#                      color='white', pad=20)

#         # Risk indicator
#         avg_contribution = float(np.mean(values_norm))
#         if avg_contribution > 0.7:
#             risk_text = "⚠️ HIGH RISK"
#             risk_color = '#E74C3C'
#         elif avg_contribution > 0.4:
#             risk_text = "⚡ MEDIUM RISK"
#             risk_color = '#F39C12'
#         else:
#             risk_text = "✓ LOW RISK"
#             risk_color = '#2ECC71'

#         ax.text(
#             0.5, 1.15, risk_text, transform=ax.transAxes,
#             fontsize=LEGEND_FS, fontweight='bold',
#             color=risk_color, ha='center',
#             bbox=dict(boxstyle='round,pad=0.5',
#                       facecolor='white', alpha=0.9)
#         )

#     # Hide extra subplots
#     for idx in range(len(datasets), len(axes)):
#         axes[idx].axis('off')

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='#1A1A2E')
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='#1A1A2E')

#     print(f"✅ Saved: {png_path}")
#     plt.show()
#     return fig

# # ═══════════════════════════════════════════════════════════════
# # 4. TREEMAP - HIERARCHICAL CONTRIBUTION VIEW
# # ═══════════════════════════════════════════════════════════════

# def plot_contribution_treemap(df, save_name="contribution_treemap"):
#     """Create treemap showing hierarchical contributions."""
#     import squarify

#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)
#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None

#     datasets = contrib_df['dataset'].values

#     fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='white')
#     axes = axes.flatten()

#     for idx, dataset in enumerate(datasets):
#         ax = axes[idx]
#         subset = contrib_df[contrib_df['dataset'] == dataset]
#         if subset.empty:
#             ax.axis('off')
#             continue

#         row = subset.iloc[0]
#         components = [col for col in contrib_df.columns if col != 'dataset']
#         values = [float(row[comp]) for comp in components]

#         total = sum(values)
#         if total > 0:
#             values = [v / total * 100.0 for v in values]

#         colors = []
#         for v in values:
#             if v > 30:
#                 colors.append('#E74C3C')  # High - Red
#             elif v > 20:
#                 colors.append('#F39C12')  # Medium - Orange
#             elif v > 10:
#                 colors.append('#F1C40F')  # Low-Medium - Yellow
#             else:
#                 colors.append('#2ECC71')  # Low - Green

#         squarify.plot(
#             sizes=values, label=components,
#             color=colors, alpha=0.8,
#             text_kwargs={'fontsize': 14, 'fontweight': 'bold'},
#             edgecolor='white', linewidth=3, ax=ax
#         )

#         ax.set_title(
#             f'{dataset}\nPrivacy Vulnerability Map',
#             fontsize=TITLE_FS, fontweight='bold', pad=15
#         )
#         ax.axis('off')

#     for idx in range(len(datasets), len(axes)):
#         axes[idx].axis('off')

#     legend_elements = [
#         Rectangle((0, 0), 1, 1, fc='#E74C3C', label='High Risk (>30%)'),
#         Rectangle((0, 0), 1, 1, fc='#F39C12', label='Medium Risk (20-30%)'),
#         Rectangle((0, 0), 1, 1, fc='#F1C40F', label='Low-Medium Risk (10-20%)'),
#         Rectangle((0, 0), 1, 1, fc='#2ECC71', label='Low Risk (<10%)'),
#     ]
#     fig.legend(
#         handles=legend_elements, loc='lower center',
#         ncol=4, fontsize=LEGEND_FS, frameon=True,
#         bbox_to_anchor=(0.5, -0.02)
#     )

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight")
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

#     print(f"✅ Saved: {png_path}")
#     plt.show()
#     return fig

# # ═══════════════════════════════════════════════════════════════
# # 5. WATERFALL CHART - CUMULATIVE CONTRIBUTION
# # ═══════════════════════════════════════════════════════════════

# def plot_contribution_waterfall(df, save_name="contribution_waterfall"):
#     """Create waterfall chart showing cumulative contributions."""
#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)
#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None

#     datasets = contrib_df['dataset'].values

#     fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='#ECEFF1')
#     axes = axes.flatten()

#     for idx, dataset in enumerate(datasets):
#         ax = axes[idx]
#         subset = contrib_df[contrib_df['dataset'] == dataset]
#         if subset.empty:
#             ax.axis('off')
#             continue

#         row = subset.iloc[0]
#         components = [col for col in contrib_df.columns if col != 'dataset']
#         values = [float(row[comp]) for comp in components]

#         # Sort by contribution value (descending)
#         sorted_pairs = sorted(zip(components, values), key=lambda x: x[1], reverse=True)
#         components_sorted = [p[0] for p in sorted_pairs]
#         values_sorted = [p[1] for p in sorted_pairs]

#         if len(values_sorted) == 0:
#             ax.axis('off')
#             continue

#         # Cumulative totals and bottoms
#         cumulative = np.cumsum(values_sorted)
#         bottoms = np.concatenate(([0.0], cumulative[:-1]))
#         total = cumulative[-1]

#         colors = ['#E74C3C' if i == 0 else '#3498DB' for i in range(len(values_sorted))]
#         colors.append('#2ECC71')  # Final total bar

#         # Plot contribution bars
#         for i, (val, bottom) in enumerate(zip(values_sorted, bottoms)):
#             ax.bar(
#                 i, val, bottom=bottom,
#                 color=colors[i], edgecolor='white',
#                 linewidth=2, width=0.6, alpha=0.8
#             )

#             # Numeric label
#             ax.text(
#                 i, bottom + val / 2.0,
#                 f'{val:.3f}',
#                 ha='center', va='center',
#                 fontsize=12, fontweight='bold', color='white'
#             )

#             # Connector to the next cumulative level
#             if i < len(values_sorted) - 1:
#                 next_total = cumulative[i]
#                 ax.plot(
#                     [i + 0.3, i + 0.7],
#                     [next_total, next_total],
#                     'k--', linewidth=1.5, alpha=0.5
#                 )

#         # Total bar
#         total_idx = len(values_sorted)
#         ax.bar(
#             total_idx, total, bottom=0.0,
#             color=colors[-1], edgecolor='white',
#             linewidth=2, width=0.6, alpha=0.8
#         )
#         ax.text(
#             total_idx, total / 2.0,
#             f'Total\n{total:.3f}',
#             ha='center', va='center',
#             fontsize=14, fontweight='bold', color='white'
#         )

#         ax.set_xticks(range(len(components_sorted) + 1))
#         ax.set_xticklabels(
#             components_sorted + ['TOTAL'],
#             fontsize=TICK_FS - 2, rotation=45, ha='right'
#         )
#         ax.set_ylabel('Contribution', fontsize=LABEL_FS, fontweight='bold')
#         ax.set_title(
#             f'{dataset}\nCumulative Contribution',
#             fontsize=TITLE_FS, fontweight='bold', pad=15
#         )

#         ax.set_facecolor('#ECEFF1')
#         ax.grid(axis='y', alpha=0.3, linestyle='--')

#         for spine in ax.spines.values():
#             spine.set_edgecolor('#2C3E50')
#             spine.set_linewidth(2)

#     for idx in range(len(datasets), len(axes)):
#         axes[idx].axis('off')

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='#ECEFF1')
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='#ECEFF1')

#     print(f"✅ Saved: {png_path}")
#     plt.show()
#     return fig

# # ═══════════════════════════════════════════════════════════════
# # 6. SUNBURST CHART - HIERARCHICAL PRIVACY RISK
# # ═══════════════════════════════════════════════════════════════

# def plot_privacy_risk_sunburst(df, save_name="privacy_risk_sunburst"):
#     """Create sunburst-like ring chart showing hierarchical privacy risks."""
#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)
#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None

#     components = [col for col in contrib_df.columns if col != 'dataset']
#     avg_contributions = contrib_df[components].mean()

#     sorted_components = avg_contributions.sort_values(ascending=False)
#     total = sorted_components.sum()

#     if total <= 0:
#         print("⚠️ All contributions are zero; cannot build sunburst.")
#         return None

#     fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')

#     outer_radius = 1.0
#     inner_radius = 0.4

#     colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_components)))

#     # Angles in degrees
#     angles = [0.0]
#     for val in sorted_components:
#         angles.append(angles[-1] + (val / total) * 360.0)

#     # Draw wedges
#     for i, (comp, val) in enumerate(sorted_components.items()):
#         wedge = Wedge(
#             (0.5, 0.5), outer_radius,
#             angles[i], angles[i + 1],
#             width=outer_radius - inner_radius,
#             facecolor=colors[i], edgecolor='white', linewidth=3
#         )
#         ax.add_patch(wedge)

#         mid_angle = (angles[i] + angles[i + 1]) / 2.0
#         label_radius = (outer_radius + inner_radius) / 2.0
#         x = 0.5 + label_radius * np.cos(np.radians(mid_angle))
#         y = 0.5 + label_radius * np.sin(np.radians(mid_angle))

#         ax.text(
#             x, y, f'{comp}\n{val:.3f}',
#             ha='center', va='center', fontsize=14,
#             fontweight='bold', color='white',
#             bbox=dict(
#                 boxstyle='round,pad=0.3',
#                 facecolor='black', alpha=0.6
#             )
#         )

#     # Center circle with title
#     center_circle = plt.Circle((0.5, 0.5), inner_radius,
#                                color='#2C3E50', zorder=10)
#     ax.add_patch(center_circle)
#     ax.text(
#         0.5, 0.5, 'Privacy\nVulnerability\nProfile',
#         ha='center', va='center',
#         fontsize=TITLE_FS, fontweight='bold', color='white'
#     )

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect('equal')
#     ax.axis('off')
#     ax.set_title(
#         'QML Component Contribution to Privacy Risk\n(Average Across All Datasets)',
#         fontsize=TITLE_FS + 2, fontweight='bold', pad=30
#     )

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight")
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

#     print(f"✅ Saved: {png_path}")
#     plt.show()
#     return fig

# # ═══════════════════════════════════════════════════════════════
# # 7. RANKING TABLE WITH HEATMAP
# # ═══════════════════════════════════════════════════════════════

# def plot_component_ranking_table(df, save_name="component_ranking_table"):
#     """Create ranking table with color-coded rows for component contributions."""
#     png_path = SAVE_DIR / f"{save_name}.png"
#     pdf_path = SAVE_DIR / f"{save_name}.pdf"

#     contrib_df = analyze_component_contributions(df)
#     if contrib_df.empty:
#         print("⚠️ No contribution data available")
#         return None, None

#     components = [col for col in contrib_df.columns if col != 'dataset']

#     ranking_data = []
#     for comp in components:
#         avg_contrib = contrib_df[comp].mean()
#         max_contrib = contrib_df[comp].max()
#         min_contrib = contrib_df[comp].min()
#         std_contrib = contrib_df[comp].std()

#         if avg_contrib > 0.3:
#             risk = 'HIGH'
#         elif avg_contrib > 0.15:
#             risk = 'MEDIUM'
#         else:
#             risk = 'LOW'

#         ranking_data.append({
#             'Component': comp,
#             'Avg Contribution': avg_contrib,
#             'Max': max_contrib,
#             'Min': min_contrib,
#             'Std Dev': std_contrib,
#             'Risk Level': risk,
#         })

#     ranking_df = pd.DataFrame(ranking_data).sort_values(
#         'Avg Contribution', ascending=False
#     )

#     fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
#     ax.axis('tight')
#     ax.axis('off')

#     table_data = []
#     table_data.append(['Rank', 'Component', 'Avg Contrib', 'Max', 'Min', 'Std Dev', 'Risk Level'])

#     for _, row in ranking_df.iterrows():
#         table_data.append([
#             f'{len(table_data)}',
#             row['Component'],
#             f'{row["Avg Contribution"]:.4f}',
#             f'{row["Max"]:.4f}',
#             f'{row["Min"]:.4f}',
#             f'{row["Std Dev"]:.4f}',
#             row['Risk Level'],
#         ])

#     table = ax.table(
#         cellText=table_data, cellLoc='center',
#         loc='center', bbox=[0, 0, 1, 1]
#     )

#     table.auto_set_font_size(False)
#     table.set_fontsize(14)
#     table.scale(1, 3)

#     # Header styling
#     for i in range(7):
#         cell = table[(0, i)]
#         cell.set_facecolor('#2C3E50')
#         cell.set_text_props(weight='bold', color='white', fontsize=16)

#     # Data row styling based on risk
#     for i in range(1, len(table_data)):
#         risk_level = table_data[i][6]

#         if risk_level == 'HIGH':
#             row_color = '#FADBD8'
#         elif risk_level == 'MEDIUM':
#             row_color = '#FCF3CF'
#         else:
#             row_color = '#D5F4E6'

#         for j in range(7):
#             cell = table[(i, j)]
#             cell.set_facecolor(row_color)
#             cell.set_edgecolor('white')
#             cell.set_linewidth(2)

#             if j == 6:  # Risk column bold
#                 cell.set_text_props(weight='bold', fontsize=15)

#     ax.set_title(
#         'QML Component Contribution Rankings\nPrivacy Vulnerability Indicators',
#         fontsize=TITLE_FS + 2, fontweight='bold', pad=30
#     )

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=300, bbox_inches="tight")
#     plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

#     print(f"✅ Saved: {png_path}")
#     plt.show()

#     return fig, ranking_df

# # ═══════════════════════════════════════════════════════════════
# # MAIN EXECUTION
# # ═══════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     print("\n" + "=" * 80)
#     print("🔐 QML COMPONENT PRIVACY VULNERABILITY ANALYSIS")
#     print("Identifying Top Contributors to Generalization Gap")
#     print("=" * 80 + "\n")

#     df = load_all_data()

#     print("\n" + "=" * 80)
#     print("GENERATING PRIVACY VULNERABILITY VISUALIZATIONS")
#     print("=" * 80 + "\n")

#     print("1️⃣  Creating stacked bar chart (component contributions)...")
#     fig1, contrib_df = plot_component_contributions_stacked(df)

#     print("\n2️⃣  Creating radar chart (vulnerability profiles)...")
#     plot_vulnerability_radar(df)

#     print("\n3️⃣  Creating treemap (hierarchical contributions)...")
#     plot_contribution_treemap(df)

#     print("\n4️⃣  Creating waterfall chart (cumulative contributions)...")
#     plot_contribution_waterfall(df)

#     print("\n5️⃣  Creating sunburst chart (privacy risk hierarchy)...")
#     plot_privacy_risk_sunburst(df)

#     print("\n6️⃣  Creating ranking table (component rankings)...")
#     fig6, ranking_df = plot_component_ranking_table(df)

#     print("\n" + "=" * 80)
#     print("📊 COMPONENT CONTRIBUTION SUMMARY")
#     print("=" * 80)

#     if contrib_df is not None:
#         print("\n🎯 Average Contributions Across All Datasets:")
#         components = [col for col in contrib_df.columns if col != 'dataset']
#         avg_contribs = contrib_df[components].mean().sort_values(ascending=False)
#         for comp, val in avg_contribs.items():
#             print(f"   {comp:20s}: {val:.4f}")

#     if ranking_df is not None:
#         print("\n🏆 TOP 3 PRIVACY VULNERABILITY INDICATORS:")
#         for _, row in ranking_df.head(3).iterrows():
#             print(f"   {row['Component']:20s} - Risk: {row['Risk Level']:6s} - Contrib: {row['Avg Contribution']:.4f}")

#     print("\n" + "=" * 80)
#     print("✅ ALL PRIVACY VULNERABILITY VISUALIZATIONS GENERATED!")
#     print("=" * 80)
#     print("\n📁 Saved in:", SAVE_DIR)
#     print("  • component_contributions_stacked.png/pdf")
#     print("  • vulnerability_radar.png/pdf")
#     print("  • contribution_treemap.png/pdf")
#     print("  • contribution_waterfall.png/pdf")
#     print("  • privacy_risk_sunburst.png/pdf")
#     print("  • component_ranking_table.png/pdf")
#     print("=" * 80 + "\n")


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Wedge
from pathlib import Path
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SAVE_DIR = Path(r".\examples\mnist\gen_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATHS_WITH_LABELS = [
    (r".\examples\mnist\gen_results\master_results_full_pipeline_moon.csv",    "Moons"),
    (r".\examples\mnist\gen_results\master_results_full_pipeline_blobs.csv",   "Blobs"),
    (r".\examples\mnist\gen_results\master_results_full_pipeline_circles.csv", "Circles"),
    (r".\examples\mnist\gen_results\mnist_extensive_results_updated.csv",      "MNIST-QNN"),
    (r".\examples\mnist\gen_results\hqnn_extensive_results.csv",              "MNIST-HQNN"),
    (r".\examples\mnist\gen_results\qcnn_extensive_results.csv",              "MNIST-QCNN"),
]

DATASET_ORDER = ['Moons', 'Blobs', 'Circles', 'MNIST-QNN', 'MNIST-HQNN', 'MNIST-QCNN']

# Font sizes
LABEL_FS = 22
TICK_FS = 18
TITLE_FS = 24
LEGEND_FS = 16

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_all_data():
    """Load and combine all datasets."""
    all_data = []
    for file_path, label in FILE_PATHS_WITH_LABELS:
        try:
            df = pd.read_csv(file_path)
            df['dataset'] = label
            all_data.append(df)
            print(f"✓ Loaded: {label} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error loading {label}: {e}")

    if not all_data:
        raise RuntimeError("No datasets could be loaded. Check file paths and CSVs.")

    df = pd.concat(all_data, ignore_index=True)

    # Generalization gap
    df['generalization_gap'] = df['acc_train'] - df['acc_test']

    # Handle column names for depth
    if 'depth' not in df.columns:
        if 'q_layers' in df.columns:
            df['depth'] = df['q_layers']
        elif 'layers' in df.columns:
            df['depth'] = df['layers']

    # Handle column names for n_wires
    if 'n_wires' not in df.columns:
        if 'n_qubits' in df.columns:
            df['n_wires'] = df['n_qubits']
        elif 'qubits' in df.columns:
            df['n_wires'] = df['qubits']

    print(f"\n📊 Total data: {len(df)} rows")
    print(f"📋 Columns: {df.columns.tolist()}")
    return df

# ═══════════════════════════════════════════════════════════════
# COMPONENT CONTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_component_contributions(df):
    """
    Analyze contribution of each component to generalization gap.

    Components:
      - Numeric: n_wires, depth, reps
      - Categorical: fm_kind, fm_op, ql_ent, ql_op
      - Interactions (numeric × numeric): n_wires×depth, n_wires×reps, depth×reps
    """
    # Base components we want to analyze
    components = ['n_wires', 'depth', 'reps',
                  'fm_kind', 'fm_op', 'ql_ent', 'ql_op']

    # Which ones are categorical (ANOVA)
    categorical_comps = {'fm_kind', 'fm_op', 'ql_ent', 'ql_op'}

    results = []
    datasets = [ds for ds in DATASET_ORDER if ds in df['dataset'].unique()]

    for dataset in datasets:
        subset = df[df['dataset'] == dataset].copy()
        if subset.empty:
            continue

        component_stats = {}

        for comp in components:
            if comp not in subset.columns:
                continue

            if comp in categorical_comps:
                # Categorical variable – ANOVA on groups
                groups = [
                    group['generalization_gap'].values
                    for _, group in subset.groupby(comp)
                ]
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    # Simple eta-squared–style effect size
                    effect_size = f_stat / (f_stat + len(subset) - len(groups))
                else:
                    effect_size = 0.0
            else:
                # Numerical variable – correlation magnitude
                corr = subset[[comp, 'generalization_gap']].corr().iloc[0, 1]
                effect_size = 0.0 if pd.isna(corr) else float(abs(corr))

            component_stats[comp] = effect_size

        # Interaction terms (numeric × numeric)
        if 'n_wires' in subset.columns and 'depth' in subset.columns:
            subset['nwires_depth'] = subset['n_wires'] * subset['depth']
            corr = subset[['nwires_depth', 'generalization_gap']].corr().iloc[0, 1]
            component_stats['n_wires×depth'] = 0.0 if pd.isna(corr) else float(abs(corr))

        if 'n_wires' in subset.columns and 'reps' in subset.columns:
            subset['nwires_reps'] = subset['n_wires'] * subset['reps']
            corr = subset[['nwires_reps', 'generalization_gap']].corr().iloc[0, 1]
            component_stats['n_wires×reps'] = 0.0 if pd.isna(corr) else float(abs(corr))

        if 'depth' in subset.columns and 'reps' in subset.columns:
            subset['depth_reps'] = subset['depth'] * subset['reps']
            corr = subset[['depth_reps', 'generalization_gap']].corr().iloc[0, 1]
            component_stats['depth×reps'] = 0.0 if pd.isna(corr) else float(abs(corr))

        results.append({
            'dataset': dataset,
            **component_stats
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.fillna(0.0)
    return result_df

# ═══════════════════════════════════════════════════════════════
# TREEMAP - HIERARCHICAL CONTRIBUTION VIEW
# ═══════════════════════════════════════════════════════════════

def plot_contribution_treemap(df, save_name="contribution_treemap"):
    """Create treemap showing hierarchical contributions."""
    import squarify

    png_path = SAVE_DIR / f"{save_name}.png"
    pdf_path = SAVE_DIR / f"{save_name}.pdf"

    contrib_df = analyze_component_contributions(df)
    if contrib_df.empty:
        print("⚠️ No contribution data available")
        return None

    datasets = contrib_df['dataset'].values

    # White figure background
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='white')
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = contrib_df[contrib_df['dataset'] == dataset]
        if subset.empty:
            ax.axis('off')
            continue

        row = subset.iloc[0]
        components = [col for col in contrib_df.columns if col != 'dataset']
        values = [float(row[comp]) for comp in components]

        total = sum(values)
        if total > 0:
            values = [v / total * 100.0 for v in values]

        colors = []
        for v in values:
            if v > 30:
                colors.append('#E74C3C')  # High - Red
            elif v > 20:
                colors.append('#F39C12')  # Medium - Orange
            elif v > 10:
                colors.append('#F1C40F')  # Low-Medium - Yellow
            else:
                colors.append('#2ECC71')  # Low - Green

        squarify.plot(
            sizes=values, label=components,
            color=colors, alpha=0.8,
            text_kwargs={'fontsize': 14, 'fontweight': 'bold'},
            edgecolor='white', linewidth=3, ax=ax
        )

        ax.set_title(
            f'{dataset}\nPrivacy Vulnerability Map',
            fontsize=TITLE_FS, fontweight='bold', pad=15
        )
        ax.axis('off')

    for idx in range(len(datasets), len(axes)):
        axes[idx].axis('off')

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#E74C3C', label='High Risk (>30%)'),
        Rectangle((0, 0), 1, 1, fc='#F39C12', label='Medium Risk (20-30%)'),
        Rectangle((0, 0), 1, 1, fc='#F1C40F', label='Low-Medium Risk (10-20%)'),
        Rectangle((0, 0), 1, 1, fc='#2ECC71', label='Low Risk (<10%)'),
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=4, fontsize=LEGEND_FS, frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    # Explicit white background in save
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='white')

    print(f"✅ Saved: {png_path}")
    plt.show()
    return fig

# ═══════════════════════════════════════════════════════════════
# WATERFALL CHART - CUMULATIVE CONTRIBUTION
# ═══════════════════════════════════════════════════════════════

def plot_contribution_waterfall(df, save_name="contribution_waterfall"):
    """Create waterfall chart showing cumulative contributions."""
    png_path = SAVE_DIR / f"{save_name}.png"
    pdf_path = SAVE_DIR / f"{save_name}.pdf"

    plt.rcParams['font.family'] = 'serif'
    
    contrib_df = analyze_component_contributions(df)
    if contrib_df.empty:
        print("⚠️ No contribution data available")
        return None

    datasets = contrib_df['dataset'].values

    # White figure background
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='white')
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = contrib_df[contrib_df['dataset'] == dataset]
        if subset.empty:
            ax.axis('off')
            continue

        row = subset.iloc[0]
        components = [col for col in contrib_df.columns if col != 'dataset']
        values = [float(row[comp]) for comp in components]

        # Sort by contribution value (descending)
        sorted_pairs = sorted(zip(components, values), key=lambda x: x[1], reverse=True)
        components_sorted = [p[0] for p in sorted_pairs]
        values_sorted = [p[1] for p in sorted_pairs]

        if len(values_sorted) == 0:
            ax.axis('off')
            continue

        # Cumulative totals and bottoms
        cumulative = np.cumsum(values_sorted)
        bottoms = np.concatenate(([0.0], cumulative[:-1]))
        total = cumulative[-1]

        # Y-limit with headroom for labels
        y_max = total * 1.25 if total > 0 else 1.0
        ax.set_ylim(0, y_max)

        colors = ["#E73C3C" if i == 0 else "#3434DB" for i in range(len(values_sorted))]
        colors.append("#00CA54")  # Final total bar

        label_offset = 0.02 * y_max  # vertical gap above bars

        # Plot contribution bars
        for i, (val, bottom) in enumerate(zip(values_sorted, bottoms)):
            ax.bar(
                i, val, bottom=bottom,
                color=colors[i], edgecolor='white',
                linewidth=2, width=0.6, alpha=0.8
            )

            # Percentage value (without % sign)
            if total > 0:
                pct = (val / total) * 100.0
            else:
                pct = 0.0
            label_text = f'{pct:.1f}'

            # Position label just above the bar
            y_label = bottom + val + label_offset
            if y_label > y_max:
                y_label = bottom + val * 0.5  # fallback inside bar if needed

            ax.text(
                i, y_label,
                label_text,
                ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='black'
            )

            # Connector line to next cumulative level
            if i < len(values_sorted) - 1:
                next_total = cumulative[i]
                ax.plot(
                    [i + 0.3, i + 0.7],
                    [next_total, next_total],
                    'k--', linewidth=1.5, alpha=0.5
                )

        # Total bar
        total_idx = len(values_sorted)
        ax.bar(
            total_idx, total, bottom=0.0,
            color=colors[-1], edgecolor='white',
            linewidth=2, width=0.6, alpha=0.8
        )

        # Total label (100, without %)
        total_label_y = total + label_offset
        if total_label_y > y_max:
            total_label_y = total * 0.5

        total_label_text = "Total\n100" if total > 0 else "Total\n0.0"

        ax.text(
            total_idx, total_label_y,
            total_label_text,
            ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='black'
        )

        ax.set_xticks(range(len(components_sorted) + 1))
        ax.set_xticklabels(
            components_sorted + ['TOTAL'],
            fontsize=TICK_FS - 2, rotation=45, ha='right'
        )
        ax.set_ylabel('Contribution', fontsize=LABEL_FS)
        ax.set_title(
            f'{dataset}',
            fontsize=TITLE_FS
        )

        ax.set_facecolor('white')

        # put grid behind bars and only vertical lines
        ax.set_axisbelow(True)
        # ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # make y-tick labels larger
        ax.tick_params(axis='y', labelsize=TICK_FS)

        for spine in ax.spines.values():
            spine.set_edgecolor('#2C3E50')
            spine.set_linewidth(2)

    # Hide unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor='white')

    print(f"✅ Saved: {png_path}")
    plt.show()
    return fig


def plot_contribution_bars(df, save_name="contribution_bars"):
    """Create small-multiple horizontal bar charts of component contributions per dataset."""
    png_path = SAVE_DIR / f"{save_name}.png"
    pdf_path = SAVE_DIR / f"{save_name}.pdf"

    contrib_df = analyze_component_contributions(df)
    if contrib_df.empty:
        print("⚠️ No contribution data available")
        return None

    datasets = contrib_df["dataset"].values
    components = [col for col in contrib_df.columns if col != "dataset"]

    # Convert to percentage contributions per dataset (like the treemap)
    contrib_perc = contrib_df.copy()
    contrib_perc[components] = (
        contrib_perc[components]
        .div(contrib_perc[components].sum(axis=1), axis=0)
        .fillna(0.0)
        * 100.0
    )

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor="white")
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = contrib_perc[contrib_perc["dataset"] == dataset]
        if subset.empty:
            ax.axis("off")
            continue

        row = subset.iloc[0]
        vals = [float(row[c]) for c in components]

        # Sort components by contribution
        sorted_pairs = sorted(zip(components, vals), key=lambda x: x[1], reverse=True)
        comps_sorted = [p[0] for p in sorted_pairs]
        vals_sorted = [p[1] for p in sorted_pairs]

        if len(vals_sorted) == 0:
            ax.axis("off")
            continue

        y = np.arange(len(comps_sorted))

        # Risk-level colors (same logic as treemap)
        colors = []
        for v in vals_sorted:
            if v > 30:
                colors.append("#E74C3C")   # High - Red
            elif v > 20:
                colors.append("#F39C12")   # Medium - Orange
            elif v > 10:
                colors.append("#F1C40F")   # Low-Medium - Yellow
            else:
                colors.append("#2ECC71")   # Low - Green

        ax.barh(y, vals_sorted, color=colors, edgecolor="white", linewidth=2)

        ax.set_yticks(y)
        ax.set_yticklabels(comps_sorted, fontsize=TICK_FS - 2)
        ax.invert_yaxis()  # Top = highest contribution

        x_max = max(vals_sorted) if max(vals_sorted) > 0 else 1.0
        ax.set_xlim(0, x_max * 1.2)

        # Add % labels at end of bars
        for i, v in enumerate(vals_sorted):
            ax.text(
                v + x_max * 0.02,
                i,
                f"{v:.1f}%",
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

        ax.set_xlabel("Contribution (%)", fontsize=LABEL_FS, fontweight="bold")
        ax.set_title(
            f"{dataset}\nPrivacy Vulnerability Contributions",
            fontsize=TITLE_FS,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")
            spine.set_linewidth(1.5)

    # Hide any unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].axis("off")

    # Shared legend for risk levels
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="#E74C3C", label="High Risk (>30%)"),
        Rectangle((0, 0), 1, 1, fc="#F39C12", label="Medium Risk (20–30%)"),
        Rectangle((0, 0), 1, 1, fc="#F1C40F", label="Low-Medium Risk (10–20%)"),
        Rectangle((0, 0), 1, 1, fc="#2ECC71", label="Low Risk (<10%)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=LEGEND_FS,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

    print(f"✅ Saved: {png_path}")
    plt.show()
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔐 QML COMPONENT PRIVACY VULNERABILITY ANALYSIS")
    print("Treemap + Waterfall Only")
    print("=" * 80 + "\n")

    df = load_all_data()

    print("\n" + "=" * 80)
    print("GENERATING PRIVACY VULNERABILITY VISUALIZATIONS (TREEMAP & WATERFALL)")
    print("=" * 80 + "\n")

    print("1️⃣  Creating treemap (hierarchical contributions)...")
    plot_contribution_treemap(df)

    print("\n2️⃣  Creating waterfall chart (cumulative contributions)...")
    plot_contribution_waterfall(df)

    print("\n" + "=" * 80)
    print("✅ VISUALIZATIONS GENERATED!")
    print("=" * 80)
    print("\n📁 Saved in:", SAVE_DIR)
    print("  • contribution_treemap.png/pdf")
    print("  • contribution_waterfall.png/pdf")
    print("=" * 80 + "\n")
