"""
Internal vs External Spillover Decomposition
=============================================

This script decomposes the spatial spillover for target stations into:
1. INTERNAL Spillover: Pollution from OTHER target stations (self-reinforcing cluster)
2. EXTERNAL Spillover: Pollution from stations OUTSIDE the target network

Key Question: Are target stations a self-reinforcing pollution cluster 
              or victims of external pollution pressure?

Target Stations: Borgo Valsugana, Monte Gaza, Parco S. Chiara, Piana Rotaliana
"""

import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from libpysal.weights import KNN
from spreg import ML_Lag
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_STATIONS = [
    'Borgo Valsugana',
    'Monte Gaza', 
    'Parco S. Chiara',
    'Piana Rotaliana'
]

K_NEIGHBORS = 6

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_DIR / 'assets'
RESULTS_DIR = PROJECT_DIR / 'results'
DATA_DIR = PROJECT_DIR / 'data'

ASSETS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / 'internal_external_spillover.txt'


class TeeOutput:
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def load_and_prepare_data():
    """Load data and prepare spatial weights"""
    print("=" * 75)
    print("INTERNAL vs EXTERNAL SPILLOVER DECOMPOSITION")
    print("=" * 75)
    print("\n[1] Loading Data...")
    
    spatial_df = pl.read_parquet(DATA_DIR / 'spatial_pollution_matrix.parquet')
    
    with open(DATA_DIR / 'data_stations_metadata.json', 'r') as f:
        stations_meta = json.load(f)
    
    meta_df = pd.DataFrame(stations_meta).set_index('Station_ID')
    data_stations = [c for c in spatial_df.columns if c != 'Data']
    valid_stations = [s for s in data_stations if s in meta_df.index]
    
    print(f"    ✓ Loaded {len(valid_stations)} stations")
    
    # Verify targets
    available_targets = [s for s in TARGET_STATIONS if s in valid_stations]
    external_stations = [s for s in valid_stations if s not in TARGET_STATIONS]
    
    print(f"    ✓ Target stations: {len(available_targets)}")
    print(f"    ✓ External stations: {len(external_stations)}")
    
    # Prepare data
    pm10_means = spatial_df.select(valid_stations).mean().to_pandas().iloc[0]
    y = pm10_means.values.reshape(-1, 1)
    coords = meta_df.loc[valid_stations, ['Longitude', 'Latitude']].values
    
    # Create spatial weights
    w = KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = 'r'
    
    # Create weight matrix
    n = len(valid_stations)
    W = np.zeros((n, n))
    for i in range(n):
        for j, wt in zip(w.neighbors[i], w.weights[i]):
            W[i, j] = wt
    
    # Fit model to get rho
    X = np.column_stack([
        meta_df.loc[valid_stations, 'Latitude'].values,
        meta_df.loc[valid_stations, 'Longitude'].values
    ])
    model = ML_Lag(y, X, w=w)
    rho = model.rho
    
    print(f"\n    Spatial Lag (ρ): {rho:.4f}")
    
    return valid_stations, available_targets, external_stations, y.flatten(), W, rho, coords, meta_df


def decompose_internal_external(valid_stations, targets, external, y, W, rho):
    """
    Decompose spillover into internal (from targets) and external (from others)
    
    For each target station i:
    - Internal Spillover: ρ * Σ(w_ij * y_j) where j ∈ targets
    - External Spillover: ρ * Σ(w_ij * y_j) where j ∉ targets
    """
    print("\n[2] Decomposing Spillover: Internal vs External...")
    
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    target_indices = set(station_to_idx[t] for t in targets)
    
    results = []
    
    print(f"\n    {'='*75}")
    print(f"    {'Station':<25} {'Total Spill':<12} {'Internal':<12} {'External':<12} {'Int %':<8} {'Ext %':<8}")
    print(f"    {'='*75}")
    
    for target in targets:
        idx = station_to_idx[target]
        
        internal_spillover = 0.0
        external_spillover = 0.0
        internal_sources = []
        external_sources = []
        
        for j in range(len(valid_stations)):
            if W[idx, j] > 0:
                contribution = W[idx, j] * y[j]
                neighbor_name = valid_stations[j]
                
                if j in target_indices and j != idx:
                    internal_spillover += contribution
                    internal_sources.append({
                        'Source': neighbor_name,
                        'Weight': W[idx, j],
                        'PM10': y[j],
                        'Contribution': contribution
                    })
                elif j != idx:
                    external_spillover += contribution
                    external_sources.append({
                        'Source': neighbor_name,
                        'Weight': W[idx, j],
                        'PM10': y[j],
                        'Contribution': contribution
                    })
        
        # Apply rho
        internal_spillover *= rho
        external_spillover *= rho
        total_spillover = internal_spillover + external_spillover
        
        internal_pct = (internal_spillover / total_spillover * 100) if total_spillover > 0 else 0
        external_pct = (external_spillover / total_spillover * 100) if total_spillover > 0 else 0
        
        results.append({
            'Station': target,
            'Observed_PM10': y[idx],
            'Total_Spillover': total_spillover,
            'Internal_Spillover': internal_spillover,
            'External_Spillover': external_spillover,
            'Internal_Pct': internal_pct,
            'External_Pct': external_pct,
            'Internal_Sources': pd.DataFrame(internal_sources).sort_values('Contribution', ascending=False) if internal_sources else pd.DataFrame(),
            'External_Sources': pd.DataFrame(external_sources).sort_values('Contribution', ascending=False) if external_sources else pd.DataFrame()
        })
        
        print(f"    {target:<25} {total_spillover:<12.2f} {internal_spillover:<12.2f} {external_spillover:<12.2f} {internal_pct:<8.1f} {external_pct:<8.1f}")
    
    return results


def analyze_spillover_sources(results, rho):
    """Detailed analysis of internal vs external sources"""
    print("\n[3] Detailed Source Analysis...")
    
    for r in results:
        target = r['Station']
        
        print(f"\n    {'='*70}")
        print(f"    📍 {target}")
        print(f"    {'='*70}")
        print(f"    Observed PM10: {r['Observed_PM10']:.2f} μg/m³")
        print(f"    Total Spillover: {r['Total_Spillover']:.2f} μg/m³")
        
        # Internal analysis
        print(f"\n    🔵 INTERNAL SPILLOVER (from other target stations):")
        print(f"       Amount: {r['Internal_Spillover']:.2f} μg/m³ ({r['Internal_Pct']:.1f}%)")
        
        if not r['Internal_Sources'].empty:
            print(f"\n       Sources (within target network):")
            for _, row in r['Internal_Sources'].iterrows():
                contrib_scaled = row['Contribution'] * rho
                print(f"         → {row['Source']:<25} {contrib_scaled:.2f} μg/m³")
        else:
            print(f"       No direct connections to other target stations!")
        
        # External analysis
        print(f"\n    🔴 EXTERNAL SPILLOVER (from outside network):")
        print(f"       Amount: {r['External_Spillover']:.2f} μg/m³ ({r['External_Pct']:.1f}%)")
        
        if not r['External_Sources'].empty:
            print(f"\n       Top External Polluters:")
            for _, row in r['External_Sources'].head(5).iterrows():
                contrib_scaled = row['Contribution'] * rho
                print(f"         → {row['Source']:<35} {contrib_scaled:.2f} μg/m³")
        
        # Save detailed sources to CSV
        if not r['Internal_Sources'].empty:
            r['Internal_Sources']['Scaled_Contribution'] = r['Internal_Sources']['Contribution'] * rho
            r['Internal_Sources'].to_csv(
                RESULTS_DIR / f"internal_sources_{target.replace(' ', '_')}.csv", 
                index=False
            )
        
        if not r['External_Sources'].empty:
            r['External_Sources']['Scaled_Contribution'] = r['External_Sources']['Contribution'] * rho
            r['External_Sources'].to_csv(
                RESULTS_DIR / f"external_sources_{target.replace(' ', '_')}.csv",
                index=False
            )


def identify_top_external_polluters(results, valid_stations, targets, y, W, rho):
    """Identify the top external stations affecting the target network"""
    print("\n[4] Identifying Top External Polluters Affecting Target Network...")
    
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    target_indices = [station_to_idx[t] for t in targets]
    
    # Calculate total contribution of each external station to ALL targets
    external_impact = {}
    
    for ext_station in valid_stations:
        if ext_station in targets:
            continue
            
        ext_idx = station_to_idx[ext_station]
        total_impact = 0.0
        impacted_targets = []
        
        for target in targets:
            target_idx = station_to_idx[target]
            
            if W[target_idx, ext_idx] > 0:
                impact = rho * W[target_idx, ext_idx] * y[ext_idx]
                total_impact += impact
                impacted_targets.append(target)
        
        if total_impact > 0:
            external_impact[ext_station] = {
                'Total_Impact': total_impact,
                'PM10': y[ext_idx],
                'Targets_Affected': impacted_targets,
                'N_Targets': len(impacted_targets)
            }
    
    # Sort by total impact
    sorted_polluters = sorted(external_impact.items(), key=lambda x: x[1]['Total_Impact'], reverse=True)
    
    print(f"\n    Top External Polluters (bombarding target stations):")
    print(f"    {'Rank':<6} {'Station':<40} {'Impact (μg/m³)':<15} {'Targets Hit':<12}")
    print(f"    {'-'*73}")
    
    top_polluters = []
    for rank, (station, data) in enumerate(sorted_polluters[:10], 1):
        print(f"    {rank:<6} {station:<40} {data['Total_Impact']:<15.2f} {data['N_Targets']:<12}")
        top_polluters.append({
            'Rank': rank,
            'Station': station,
            'PM10_Level': data['PM10'],
            'Total_Impact_on_Targets': data['Total_Impact'],
            'N_Targets_Affected': data['N_Targets'],
            'Targets_Affected': ', '.join(data['Targets_Affected'])
        })
    
    # Save to CSV
    pd.DataFrame(top_polluters).to_csv(RESULTS_DIR / 'top_external_polluters_detailed.csv', index=False)
    
    return sorted_polluters


def calculate_network_summary(results):
    """Calculate summary statistics for the target network"""
    print("\n[5] Network Summary Statistics...")
    
    total_internal = sum(r['Internal_Spillover'] for r in results)
    total_external = sum(r['External_Spillover'] for r in results)
    total_spillover = total_internal + total_external
    
    avg_internal_pct = np.mean([r['Internal_Pct'] for r in results])
    avg_external_pct = np.mean([r['External_Pct'] for r in results])
    
    print(f"\n    {'='*60}")
    print(f"    NETWORK-WIDE SPILLOVER SUMMARY")
    print(f"    {'='*60}")
    print(f"\n    Total Spillover to Target Network: {total_spillover:.2f} μg/m³")
    print(f"\n    Internal (self-reinforcing): {total_internal:.2f} μg/m³ ({total_internal/total_spillover*100:.1f}%)")
    print(f"    External (external pressure): {total_external:.2f} μg/m³ ({total_external/total_spillover*100:.1f}%)")
    
    print(f"\n    Average per Target Station:")
    print(f"      - Internal Spillover: {avg_internal_pct:.1f}%")
    print(f"      - External Spillover: {avg_external_pct:.1f}%")
    
    # Verdict
    print(f"\n    {'='*60}")
    print(f"    VERDICT")
    print(f"    {'='*60}")
    
    if avg_external_pct > 70:
        verdict = "EXTERNAL VICTIMS"
        explanation = "Target stations are primarily VICTIMS of external pollution pressure."
    elif avg_internal_pct > 50:
        verdict = "SELF-REINFORCING CLUSTER"
        explanation = "Target stations form a SELF-REINFORCING pollution cluster."
    else:
        verdict = "MIXED INFLUENCE"
        explanation = "Target stations receive roughly equal internal and external pollution."
    
    print(f"\n    🎯 {verdict}")
    print(f"    → {explanation}")
    
    return {
        'total_internal': total_internal,
        'total_external': total_external,
        'avg_internal_pct': avg_internal_pct,
        'avg_external_pct': avg_external_pct,
        'verdict': verdict
    }


def create_visualizations(results, top_polluters, summary, valid_stations, targets, 
                          coords, meta_df, W, y):
    """Create comprehensive visualizations"""
    print("\n[6] Creating Visualizations...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    # =========================================================================
    # FIGURE 1: Stacked Bar - Internal vs External Spillover
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    stations = [r['Station'] for r in results]
    internal = [r['Internal_Spillover'] for r in results]
    external = [r['External_Spillover'] for r in results]
    
    x = np.arange(len(stations))
    width = 0.6
    
    bars1 = ax1.bar(x, internal, width, label='Internal (from target network)', 
                    color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x, external, width, bottom=internal, 
                    label='External (from outside)', color='#e74c3c', edgecolor='black')
    
    # Add percentage labels
    for i, r in enumerate(results):
        total = r['Total_Spillover']
        int_mid = r['Internal_Spillover'] / 2
        ext_mid = r['Internal_Spillover'] + r['External_Spillover'] / 2
        
        if r['Internal_Pct'] > 10:
            ax1.text(i, int_mid, f"{r['Internal_Pct']:.0f}%", ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='white')
        ax1.text(i, ext_mid, f"{r['External_Pct']:.0f}%", ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white')
    
    ax1.set_xlabel('Target Station', fontsize=12)
    ax1.set_ylabel('Spillover (μg/m³)', fontsize=12)
    ax1.set_title('Internal vs External Spillover Decomposition\n(Where does the pollution come from?)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in stations], fontsize=10)
    ax1.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    fig1.savefig(ASSETS_DIR / 'internal_vs_external_spillover.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: internal_vs_external_spillover.png")
    plt.close(fig1)
    
    # =========================================================================
    # FIGURE 2: Percentage Breakdown Pie Charts
    # =========================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c']
    
    for i, r in enumerate(results):
        ax = axes[i]
        sizes = [r['Internal_Pct'], r['External_Pct']]
        labels = ['Internal\n(Network)', 'External\n(Outside)']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, explode=(0.02, 0.02),
                                           textprops={'fontsize': 11})
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax.set_title(f"{r['Station']}\n(Total: {r['Total_Spillover']:.1f} μg/m³)", 
                     fontsize=12, fontweight='bold')
    
    plt.suptitle('Spillover Source Breakdown by Target Station', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig(ASSETS_DIR / 'spillover_percentage_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spillover_percentage_breakdown.png")
    plt.close(fig2)
    
    # =========================================================================
    # FIGURE 3: Network Map - Targets and External Polluters
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 11))
    
    # Get top external polluters
    top_ext_names = [p[0] for p in top_polluters[:6]]
    
    # Plot all stations
    for i, station in enumerate(valid_stations):
        if station in targets:
            color = '#e74c3c'
            size = 300
            marker = 's'  # Square for targets
            zorder = 10
        elif station in top_ext_names:
            color = '#f39c12'  # Orange for top external
            size = 200
            marker = '^'  # Triangle for external polluters
            zorder = 8
        else:
            color = '#bdc3c7'
            size = 50
            marker = 'o'
            zorder = 2
        
        ax3.scatter(coords[i, 0], coords[i, 1], c=color, s=size, marker=marker,
                    edgecolor='black', linewidth=1, zorder=zorder)
    
    # Draw connections from external polluters to targets
    for ext_station, data in top_polluters[:6]:
        if ext_station in station_to_idx:
            ext_idx = station_to_idx[ext_station]
            for target in data['Targets_Affected']:
                if target in station_to_idx:
                    target_idx = station_to_idx[target]
                    ax3.annotate('', 
                                 xy=(coords[target_idx, 0], coords[target_idx, 1]),
                                 xytext=(coords[ext_idx, 0], coords[ext_idx, 1]),
                                 arrowprops=dict(arrowstyle='->', color='#e74c3c', 
                                                 lw=2, alpha=0.6))
    
    # Draw internal connections (between targets)
    for i, t1 in enumerate(targets):
        if t1 in station_to_idx:
            idx1 = station_to_idx[t1]
            for t2 in targets[i+1:]:
                if t2 in station_to_idx:
                    idx2 = station_to_idx[t2]
                    if W[idx1, idx2] > 0 or W[idx2, idx1] > 0:
                        ax3.plot([coords[idx1, 0], coords[idx2, 0]],
                                 [coords[idx1, 1], coords[idx2, 1]],
                                 'b-', lw=3, alpha=0.7, zorder=5)
    
    # Add labels for targets and top external
    for station in targets + top_ext_names:
        if station in station_to_idx:
            idx = station_to_idx[station]
            offset = (8, 8) if station in targets else (5, -10)
            fontweight = 'bold' if station in targets else 'normal'
            ax3.annotate(station, (coords[idx, 0], coords[idx, 1]),
                         xytext=offset, textcoords='offset points',
                         fontsize=9, fontweight=fontweight,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#e74c3c', s=150, marker='s', edgecolor='black', label='Target Stations'),
        plt.scatter([], [], c='#f39c12', s=100, marker='^', edgecolor='black', label='Top External Polluters'),
        plt.scatter([], [], c='#bdc3c7', s=50, marker='o', edgecolor='black', label='Other Stations'),
        plt.Line2D([0], [0], color='#3498db', lw=3, label='Internal Connections'),
        plt.Line2D([0], [0], color='#e74c3c', lw=2, label='External → Target Flow')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.set_title('Spatial Network: Target Stations and External Polluters\n(Arrows show pollution flow direction)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig3.savefig(ASSETS_DIR / 'network_targets_external.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: network_targets_external.png")
    plt.close(fig3)
    
    # =========================================================================
    # FIGURE 4: Top External Polluters Ranking
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    top_10 = top_polluters[:10]
    names = [p[0] for p in top_10]
    impacts = [p[1]['Total_Impact'] for p in top_10]
    n_targets = [p[1]['N_Targets'] for p in top_10]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_10)))
    
    bars = ax4.barh(range(len(top_10)), impacts, color=colors, edgecolor='black')
    
    # Add target count annotations
    for i, (impact, n) in enumerate(zip(impacts, n_targets)):
        ax4.text(impact + 0.1, i, f"{n} targets", va='center', fontsize=10, fontweight='bold')
    
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels(names, fontsize=10)
    ax4.invert_yaxis()
    ax4.set_xlabel('Total Impact on Target Network (μg/m³)', fontsize=12)
    ax4.set_title('Top External Polluters Affecting Target Stations\n(Ranked by total spillover contribution)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig4.savefig(ASSETS_DIR / 'top_external_polluters_ranking.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: top_external_polluters_ranking.png")
    plt.close(fig4)
    
    # =========================================================================
    # FIGURE 5: Summary Dashboard
    # =========================================================================
    fig5 = plt.figure(figsize=(14, 10))
    
    # Main pie chart for overall network
    ax_main = fig5.add_subplot(2, 2, 1)
    overall_sizes = [summary['total_internal'], summary['total_external']]
    ax_main.pie(overall_sizes, labels=['Internal', 'External'], colors=['#3498db', '#e74c3c'],
                autopct='%1.1f%%', startangle=90, explode=(0.03, 0.03),
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax_main.set_title('Overall Network Spillover\nInternal vs External', fontsize=13, fontweight='bold')
    
    # Verdict box
    ax_verdict = fig5.add_subplot(2, 2, 2)
    ax_verdict.axis('off')
    verdict_color = '#e74c3c' if summary['verdict'] == 'EXTERNAL VICTIMS' else '#3498db'
    ax_verdict.text(0.5, 0.6, f"🎯 {summary['verdict']}", ha='center', va='center',
                    fontsize=24, fontweight='bold', color=verdict_color,
                    transform=ax_verdict.transAxes)
    
    if summary['verdict'] == 'EXTERNAL VICTIMS':
        explanation = "Target stations are primarily\nVICTIMS of external pollution"
    elif summary['verdict'] == 'SELF-REINFORCING CLUSTER':
        explanation = "Target stations form a\nSELF-REINFORCING pollution cluster"
    else:
        explanation = "Target stations receive\nMIXED pollution sources"
    
    ax_verdict.text(0.5, 0.35, explanation, ha='center', va='center',
                    fontsize=14, transform=ax_verdict.transAxes)
    
    # Stats table
    ax_stats = fig5.add_subplot(2, 2, 3)
    ax_stats.axis('off')
    
    stats_text = f"""
    Network Statistics
    ──────────────────────────
    Total Internal Spillover: {summary['total_internal']:.2f} μg/m³
    Total External Spillover: {summary['total_external']:.2f} μg/m³
    
    Average Internal %: {summary['avg_internal_pct']:.1f}%
    Average External %: {summary['avg_external_pct']:.1f}%
    
    Target Stations: {len(targets)}
    """
    ax_stats.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=12,
                  family='monospace', transform=ax_stats.transAxes)
    
    # Per-station bars
    ax_bars = fig5.add_subplot(2, 2, 4)
    x = np.arange(len(results))
    width = 0.35
    
    int_pcts = [r['Internal_Pct'] for r in results]
    ext_pcts = [r['External_Pct'] for r in results]
    
    ax_bars.bar(x - width/2, int_pcts, width, label='Internal %', color='#3498db', edgecolor='black')
    ax_bars.bar(x + width/2, ext_pcts, width, label='External %', color='#e74c3c', edgecolor='black')
    
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([r['Station'] for r in results], rotation=45, ha='right', fontsize=9)
    ax_bars.set_ylabel('Percentage (%)', fontsize=11)
    ax_bars.set_title('Internal vs External % by Station', fontsize=12, fontweight='bold')
    ax_bars.legend(fontsize=10)
    ax_bars.set_ylim(0, 100)
    
    plt.suptitle('Internal vs External Spillover Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig5.savefig(ASSETS_DIR / 'spillover_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spillover_summary_dashboard.png")
    plt.close(fig5)


def save_summary_results(results, summary):
    """Save summary results to CSV"""
    print("\n[7] Saving Results...")
    
    summary_df = pd.DataFrame([{
        'Station': r['Station'],
        'Observed_PM10': r['Observed_PM10'],
        'Total_Spillover': r['Total_Spillover'],
        'Internal_Spillover': r['Internal_Spillover'],
        'External_Spillover': r['External_Spillover'],
        'Internal_Pct': r['Internal_Pct'],
        'External_Pct': r['External_Pct']
    } for r in results])
    
    summary_df.to_csv(RESULTS_DIR / 'internal_external_spillover.csv', index=False)
    print(f"    ✓ internal_external_spillover.csv")


def main():
    with open(OUTPUT_FILE, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        
        try:
            # Load and prepare data
            valid_stations, targets, external, y, W, rho, coords, meta_df = load_and_prepare_data()
            
            # Decompose spillover
            results = decompose_internal_external(valid_stations, targets, external, y, W, rho)
            
            # Analyze sources
            analyze_spillover_sources(results, rho)
            
            # Identify top external polluters
            top_polluters = identify_top_external_polluters(results, valid_stations, targets, y, W, rho)
            
            # Network summary
            summary = calculate_network_summary(results)
            
            # Visualizations
            create_visualizations(results, top_polluters, summary, valid_stations, targets, 
                                  coords, meta_df, W, y)
            
            # Save results
            save_summary_results(results, summary)
            
            print("\n" + "=" * 75)
            print("ANALYSIS COMPLETE")
            print("=" * 75)
            print(f"\n    📁 Text Output: {OUTPUT_FILE}")
            print(f"    📊 CSV Results: {RESULTS_DIR}")
            print(f"    🖼️  Visualizations: {ASSETS_DIR}")
            print("\n")
            
        finally:
            sys.stdout = original_stdout
    
    print("Script completed successfully!")


if __name__ == '__main__':
    main()
