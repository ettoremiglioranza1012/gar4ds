"""
Spatial Impact Matrix Analysis
==============================

Computes the Impact Matrix (I - ρW)^(-1) which captures:
- DIRECT effects: Station A → Station B
- INDIRECT effects: Station A → Station C → Station B (and all longer chains)

The Impact Matrix is the spatial multiplier that transforms local shocks 
into total network-wide effects, accounting for all feedback loops.

Key Outputs:
1. Full Heatmap: Total influence between ALL station pairs
2. Target Sub-Matrix: Internal feedback among target stations
3. Network Centrality: Most influential and most vulnerable stations
4. Influence Profiles: Who affects vs who is affected most
"""

import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
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
PROJECT_DIR = SCRIPT_DIR.parent.parent
ASSETS_DIR = PROJECT_DIR / 'assets'
RESULTS_DIR = PROJECT_DIR / 'results'
DATA_DIR = PROJECT_DIR / 'data'

ASSETS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / 'impact_matrix_analysis.txt'


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


def load_data():
    """Load data and prepare spatial model"""
    print("=" * 75)
    print("SPATIAL IMPACT MATRIX ANALYSIS")
    print("Capturing Direct + Indirect Spillover Pathways")
    print("=" * 75)
    print("\n[1] Loading Data...")
    
    spatial_df = pl.read_parquet(DATA_DIR / 'spatial_pollution_matrix.parquet')
    
    with open(DATA_DIR / 'data_stations_metadata.json', 'r') as f:
        stations_meta = json.load(f)
    
    meta_df = pd.DataFrame(stations_meta).set_index('Station_ID')
    data_stations = [c for c in spatial_df.columns if c != 'Data']
    valid_stations = [s for s in data_stations if s in meta_df.index]
    
    print(f"    ✓ Loaded {len(valid_stations)} stations")
    
    available_targets = [s for s in TARGET_STATIONS if s in valid_stations]
    print(f"    ✓ Target stations available: {len(available_targets)}")
    
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
    
    return valid_stations, available_targets, y.flatten(), W, rho, coords, meta_df


def compute_impact_matrix(W, rho, valid_stations):
    """
    Compute the Impact Matrix: (I - ρW)^(-1)
    
    This matrix captures ALL spillover pathways:
    - Entry M[i,j] = total effect of a unit shock at station j on station i
    - Includes direct (j→i), indirect (j→k→i), and higher-order effects
    
    Mathematical derivation:
    y = ρWy + Xβ + ε
    (I - ρW)y = Xβ + ε
    y = (I - ρW)^(-1) (Xβ + ε)
    
    The multiplier (I - ρW)^(-1) = I + ρW + ρ²W² + ρ³W³ + ...
    captures all feedback loops in the spatial network.
    """
    print("\n[2] Computing Impact Matrix (I - ρW)^(-1)...")
    
    n = len(valid_stations)
    I = np.eye(n)
    
    # Impact Matrix: (I - ρW)^(-1)
    impact_matrix = np.linalg.inv(I - rho * W)
    
    print(f"    ✓ Computed {n}×{n} impact matrix")
    print(f"\n    Mathematical Interpretation:")
    print(f"    M = (I - ρW)^(-1) = I + ρW + ρ²W² + ρ³W³ + ...")
    print(f"    Each entry M[i,j] = total effect of station j on station i")
    
    # Basic statistics
    off_diag = impact_matrix[~np.eye(n, dtype=bool)]
    print(f"\n    Impact Matrix Statistics:")
    print(f"    - Diagonal (self-feedback): mean = {np.diag(impact_matrix).mean():.4f}")
    print(f"    - Off-diagonal (cross-effects): mean = {off_diag.mean():.4f}, max = {off_diag.max():.4f}")
    
    return impact_matrix


def analyze_target_submatrix(impact_matrix, valid_stations, targets):
    """Extract and analyze the sub-matrix for target stations"""
    print("\n[3] Analyzing Target Station Sub-Matrix...")
    
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    target_indices = [station_to_idx[t] for t in targets]
    
    # Extract sub-matrix
    target_matrix = impact_matrix[np.ix_(target_indices, target_indices)]
    
    print(f"\n    Target Influence Matrix (rows=affected, cols=source):")
    print(f"    {'':>20}", end='')
    for t in targets:
        print(f"{t[:15]:>16}", end='')
    print()
    print(f"    {'-'*85}")
    
    for i, t_row in enumerate(targets):
        print(f"    {t_row[:20]:<20}", end='')
        for j in range(len(targets)):
            val = target_matrix[i, j]
            if i == j:
                print(f"[{val:>12.4f}]", end='')  # Highlight diagonal
            else:
                print(f"{val:>16.4f}", end='')
        print()
    
    # Internal feedback analysis
    print(f"\n    Internal Feedback Analysis:")
    for i, t in enumerate(targets):
        self_effect = target_matrix[i, i]
        total_from_others = target_matrix[i, :].sum() - self_effect
        print(f"    - {t}: Self-feedback = {self_effect:.4f}, From other targets = {total_from_others:.4f}")
    
    # Total influence within target network
    total_internal_influence = target_matrix.sum()
    print(f"\n    Total influence within target network: {total_internal_influence:.4f}")
    
    return target_matrix, target_indices


def compute_network_centrality(impact_matrix, valid_stations, targets):
    """
    Compute network centrality metrics from the impact matrix
    
    - OUT-Influence (column sum): How much a station affects the ENTIRE network
    - IN-Influence (row sum): How much a station is affected BY the entire network
    """
    print("\n[4] Computing Network Centrality Metrics...")
    
    n = len(valid_stations)
    
    # Exclude self-effects for pure cross-influence
    cross_effects = impact_matrix.copy()
    np.fill_diagonal(cross_effects, 0)
    
    # OUT-Influence: column sums (excluding self)
    out_influence = cross_effects.sum(axis=0)
    
    # IN-Influence: row sums (excluding self)
    in_influence = cross_effects.sum(axis=1)
    
    # Self-feedback: diagonal
    self_feedback = np.diag(impact_matrix)
    
    # Net influence: OUT - IN (positive = net exporter, negative = net importer)
    net_influence = out_influence - in_influence
    
    centrality_df = pd.DataFrame({
        'Station': valid_stations,
        'Out_Influence': out_influence,
        'In_Influence': in_influence,
        'Self_Feedback': self_feedback,
        'Net_Influence': net_influence,
        'Is_Target': [s in targets for s in valid_stations]
    })
    
    # Rankings
    print("\n    TOP 10 MOST INFLUENTIAL STATIONS (affect others the most):")
    print(f"    {'Rank':<6} {'Station':<40} {'Out-Influence':<15}")
    print(f"    {'-'*61}")
    top_influential = centrality_df.nlargest(10, 'Out_Influence')
    for rank, (_, row) in enumerate(top_influential.iterrows(), 1):
        marker = "⭐" if row['Is_Target'] else "  "
        print(f"    {rank:<6} {marker}{row['Station']:<38} {row['Out_Influence']:<15.4f}")
    
    print("\n    TOP 10 MOST VULNERABLE STATIONS (affected by others the most):")
    print(f"    {'Rank':<6} {'Station':<40} {'In-Influence':<15}")
    print(f"    {'-'*61}")
    top_vulnerable = centrality_df.nlargest(10, 'In_Influence')
    for rank, (_, row) in enumerate(top_vulnerable.iterrows(), 1):
        marker = "⭐" if row['Is_Target'] else "  "
        print(f"    {rank:<6} {marker}{row['Station']:<38} {row['In_Influence']:<15.4f}")
    
    print("\n    NET INFLUENCE (Positive = Net Exporter, Negative = Net Importer):")
    print(f"    Target Stations:")
    target_df = centrality_df[centrality_df['Is_Target']]
    for _, row in target_df.iterrows():
        status = "EXPORTER ↑" if row['Net_Influence'] > 0 else "IMPORTER ↓"
        print(f"    - {row['Station']:<25} Net: {row['Net_Influence']:>8.4f} ({status})")
    
    return centrality_df


def compute_influence_profiles(impact_matrix, valid_stations, targets, y):
    """
    Compute PM10-weighted influence profiles
    
    The actual pollution transmitted = Impact × PM10 level
    """
    print("\n[5] Computing PM10-Weighted Influence Profiles...")
    
    n = len(valid_stations)
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    # Pollution transmitted by each station = sum of (impact[i,j] × PM10[j])
    # This tells us how much actual pollution each station sends to the network
    cross_effects = impact_matrix.copy()
    np.fill_diagonal(cross_effects, 0)
    
    # Pollution sent = column of impact matrix × own PM10
    pollution_sent = cross_effects.sum(axis=0) * y
    
    # Pollution received = row of impact matrix × source PM10 (dot product)
    pollution_received = cross_effects @ y
    
    profiles_df = pd.DataFrame({
        'Station': valid_stations,
        'PM10_Level': y,
        'Pollution_Sent': pollution_sent,
        'Pollution_Received': pollution_received,
        'Net_Position': pollution_sent - pollution_received,
        'Is_Target': [s in targets for s in valid_stations]
    })
    
    print("\n    PM10-WEIGHTED POLLUTION EXCHANGE:")
    print(f"    {'Station':<35} {'PM10':<10} {'Sent':<12} {'Received':<12} {'Net':<12}")
    print(f"    {'-'*81}")
    
    # Show targets first
    target_profiles = profiles_df[profiles_df['Is_Target']].sort_values('Net_Position')
    for _, row in target_profiles.iterrows():
        net_status = "SINK" if row['Net_Position'] < 0 else "SOURCE"
        print(f"    📍 {row['Station']:<33} {row['PM10_Level']:<10.2f} {row['Pollution_Sent']:<12.2f} "
              f"{row['Pollution_Received']:<12.2f} {row['Net_Position']:<12.2f} ({net_status})")
    
    print("\n    Top 5 Pollution SOURCES (net exporters):")
    top_sources = profiles_df.nlargest(5, 'Net_Position')
    for _, row in top_sources.iterrows():
        marker = "⭐" if row['Is_Target'] else "  "
        print(f"    {marker}{row['Station']:<33} Net: +{row['Net_Position']:.2f} μg/m³")
    
    print("\n    Top 5 Pollution SINKS (net importers):")
    top_sinks = profiles_df.nsmallest(5, 'Net_Position')
    for _, row in top_sinks.iterrows():
        marker = "⭐" if row['Is_Target'] else "  "
        print(f"    {marker}{row['Station']:<33} Net: {row['Net_Position']:.2f} μg/m³")
    
    return profiles_df


def trace_indirect_pathways(W, rho, valid_stations, targets):
    """
    Trace specific indirect pathways between stations
    
    Example: Rovereto → Piana Rotaliana → Monte Gaza → Parco S. Chiara
    """
    print("\n[6] Tracing Indirect Spillover Pathways...")
    
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    # Compute power matrices
    W1 = rho * W  # Direct effect
    W2 = rho**2 * (W @ W)  # 2-hop indirect
    W3 = rho**3 * (W @ W @ W)  # 3-hop indirect
    
    print("\n    Pathway Decomposition for Target Stations:")
    print(f"    (Direct = ρW, 2-hop = ρ²W², 3-hop = ρ³W³)\n")
    
    # For each target, show top influences decomposed by path length
    for target in targets:
        if target not in station_to_idx:
            continue
        
        target_idx = station_to_idx[target]
        
        print(f"    📍 {target}:")
        print(f"    {'Source':<35} {'Direct':<10} {'2-hop':<10} {'3-hop':<10} {'Total':<10}")
        print(f"    {'-'*75}")
        
        # Find top sources
        total_effect = W1[target_idx, :] + W2[target_idx, :] + W3[target_idx, :]
        top_sources_idx = np.argsort(total_effect)[::-1][:5]
        
        for src_idx in top_sources_idx:
            if src_idx == target_idx:
                continue
            source = valid_stations[src_idx]
            direct = W1[target_idx, src_idx]
            hop2 = W2[target_idx, src_idx]
            hop3 = W3[target_idx, src_idx]
            total = direct + hop2 + hop3
            
            if total > 0.001:
                print(f"    {source:<35} {direct:<10.4f} {hop2:<10.4f} {hop3:<10.4f} {total:<10.4f}")
        print()


def create_visualizations(impact_matrix, target_matrix, centrality_df, profiles_df,
                          valid_stations, targets, target_indices, coords, meta_df, y):
    """Create comprehensive visualizations"""
    print("\n[7] Creating Visualizations...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    # =========================================================================
    # FIGURE 1: Full Impact Matrix Heatmap
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(16, 14))
    
    # Remove diagonal for better color scaling
    display_matrix = impact_matrix.copy()
    np.fill_diagonal(display_matrix, np.nan)
    
    # Create heatmap
    sns.heatmap(display_matrix, cmap='YlOrRd', ax=ax1,
                xticklabels=valid_stations, yticklabels=valid_stations,
                cbar_kws={'label': 'Total Impact (Direct + Indirect)'})
    
    # Highlight target stations
    for i, station in enumerate(valid_stations):
        if station in targets:
            ax1.get_xticklabels()[i].set_color('red')
            ax1.get_xticklabels()[i].set_fontweight('bold')
            ax1.get_yticklabels()[i].set_color('red')
            ax1.get_yticklabels()[i].set_fontweight('bold')
    
    ax1.set_xlabel('Source Station (causes pollution)', fontsize=12)
    ax1.set_ylabel('Affected Station (receives pollution)', fontsize=12)
    ax1.set_title('Full Impact Matrix: (I - ρW)⁻¹\n(Red labels = Target Stations)', 
                  fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig1.savefig(ASSETS_DIR / 'impact_matrix_full_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: impact_matrix_full_heatmap.png")
    plt.close(fig1)
    
    # =========================================================================
    # FIGURE 2: Target Sub-Matrix (Zoomed)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(target_matrix, annot=True, fmt='.4f', cmap='Blues', ax=ax2,
                xticklabels=targets, yticklabels=targets,
                cbar_kws={'label': 'Total Impact'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax2.set_xlabel('Source Station', fontsize=12)
    ax2.set_ylabel('Affected Station', fontsize=12)
    ax2.set_title('Target Stations Impact Sub-Matrix\n(Internal Feedback Structure)', 
                  fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    fig2.savefig(ASSETS_DIR / 'impact_matrix_targets.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: impact_matrix_targets.png")
    plt.close(fig2)
    
    # =========================================================================
    # FIGURE 3: Network Centrality - Out vs In Influence
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    colors = ['#e74c3c' if t else '#3498db' for t in centrality_df['Is_Target']]
    sizes = [250 if t else 80 for t in centrality_df['Is_Target']]
    
    scatter = ax3.scatter(centrality_df['Out_Influence'], centrality_df['In_Influence'],
                          c=colors, s=sizes, edgecolor='black', alpha=0.7)
    
    # Add labels for targets and extremes
    for _, row in centrality_df.iterrows():
        if row['Is_Target'] or row['Out_Influence'] > centrality_df['Out_Influence'].quantile(0.9) \
           or row['In_Influence'] > centrality_df['In_Influence'].quantile(0.9):
            ax3.annotate(row['Station'], (row['Out_Influence'], row['In_Influence']),
                         fontsize=8, fontweight='bold' if row['Is_Target'] else 'normal',
                         xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal (net zero)
    max_val = max(centrality_df['Out_Influence'].max(), centrality_df['In_Influence'].max())
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Net Zero Line')
    
    # Quadrant labels
    ax3.text(max_val*0.95, max_val*0.05, 'NET EXPORTERS\n(Sources)', ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax3.text(max_val*0.05, max_val*0.95, 'NET IMPORTERS\n(Sinks)', ha='left', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#ccccff', alpha=0.8))
    
    ax3.set_xlabel('Out-Influence (affects others)', fontsize=12)
    ax3.set_ylabel('In-Influence (affected by others)', fontsize=12)
    ax3.set_title('Network Centrality: Influence vs Vulnerability\n(Red = Target Stations)', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    fig3.savefig(ASSETS_DIR / 'network_centrality_scatter.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: network_centrality_scatter.png")
    plt.close(fig3)
    
    # =========================================================================
    # FIGURE 4: Influence Profiles - Sent vs Received
    # =========================================================================
    fig4, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Top 15 by pollution sent
    ax4a = axes[0]
    top_sent = profiles_df.nlargest(15, 'Pollution_Sent')
    colors = ['#e74c3c' if t else '#3498db' for t in top_sent['Is_Target']]
    bars = ax4a.barh(range(len(top_sent)), top_sent['Pollution_Sent'], color=colors, edgecolor='black')
    ax4a.set_yticks(range(len(top_sent)))
    ax4a.set_yticklabels(top_sent['Station'], fontsize=9)
    ax4a.invert_yaxis()
    ax4a.set_xlabel('Pollution Sent (μg/m³)', fontsize=11)
    ax4a.set_title('Top Pollution EXPORTERS\n(Affect the network most)', fontsize=12, fontweight='bold')
    
    # Top 15 by pollution received
    ax4b = axes[1]
    top_received = profiles_df.nlargest(15, 'Pollution_Received')
    colors = ['#e74c3c' if t else '#3498db' for t in top_received['Is_Target']]
    bars = ax4b.barh(range(len(top_received)), top_received['Pollution_Received'], color=colors, edgecolor='black')
    ax4b.set_yticks(range(len(top_received)))
    ax4b.set_yticklabels(top_received['Station'], fontsize=9)
    ax4b.invert_yaxis()
    ax4b.set_xlabel('Pollution Received (μg/m³)', fontsize=11)
    ax4b.set_title('Top Pollution IMPORTERS\n(Affected by network most)', fontsize=12, fontweight='bold')
    
    plt.suptitle('PM10-Weighted Influence Profiles (Red = Target Stations)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig(ASSETS_DIR / 'influence_profiles.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: influence_profiles.png")
    plt.close(fig4)
    
    # =========================================================================
    # FIGURE 5: Spatial Map with Influence Arrows
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(14, 11))
    
    # Size by out-influence
    out_inf_norm = (centrality_df['Out_Influence'] - centrality_df['Out_Influence'].min()) / \
                   (centrality_df['Out_Influence'].max() - centrality_df['Out_Influence'].min())
    sizes = 50 + 300 * out_inf_norm.values
    
    # Color by net position
    colors = profiles_df['Net_Position'].values
    
    scatter = ax5.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes,
                          cmap='RdBu_r', edgecolor='black', alpha=0.8,
                          vmin=-np.abs(colors).max(), vmax=np.abs(colors).max())
    
    # Highlight targets
    for target in targets:
        if target in station_to_idx:
            idx = station_to_idx[target]
            ax5.scatter(coords[idx, 0], coords[idx, 1], s=400, facecolors='none',
                        edgecolors='black', linewidth=3)
            ax5.annotate(target, (coords[idx, 0], coords[idx, 1]),
                         fontsize=10, fontweight='bold',
                         xytext=(8, 8), textcoords='offset points',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
    cbar.set_label('Net Position (Red=Exporter, Blue=Importer)', fontsize=11)
    
    ax5.set_xlabel('Longitude', fontsize=12)
    ax5.set_ylabel('Latitude', fontsize=12)
    ax5.set_title('Spatial Network: Influence Map\n(Size=Out-Influence, Color=Net Position, Circles=Targets)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig5.savefig(ASSETS_DIR / 'spatial_influence_map.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spatial_influence_map.png")
    plt.close(fig5)


def save_results(impact_matrix, target_matrix, centrality_df, profiles_df, valid_stations, targets):
    """Save all results to CSV"""
    print("\n[8] Saving Results...")
    
    # Full impact matrix
    impact_df = pd.DataFrame(impact_matrix, index=valid_stations, columns=valid_stations)
    impact_df.to_csv(RESULTS_DIR / 'impact_matrix_full.csv')
    print(f"    ✓ impact_matrix_full.csv")
    
    # Target sub-matrix
    target_impact_df = pd.DataFrame(target_matrix, index=targets, columns=targets)
    target_impact_df.to_csv(RESULTS_DIR / 'impact_matrix_targets.csv')
    print(f"    ✓ impact_matrix_targets.csv")
    
    # Centrality metrics
    centrality_df.to_csv(RESULTS_DIR / 'network_centrality.csv', index=False)
    print(f"    ✓ network_centrality.csv")
    
    # Influence profiles
    profiles_df.to_csv(RESULTS_DIR / 'influence_profiles.csv', index=False)
    print(f"    ✓ influence_profiles.csv")


def main():
    with open(OUTPUT_FILE, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        
        try:
            # Load data
            valid_stations, targets, y, W, rho, coords, meta_df = load_data()
            
            # Compute impact matrix
            impact_matrix = compute_impact_matrix(W, rho, valid_stations)
            
            # Analyze target sub-matrix
            target_matrix, target_indices = analyze_target_submatrix(impact_matrix, valid_stations, targets)
            
            # Network centrality
            centrality_df = compute_network_centrality(impact_matrix, valid_stations, targets)
            
            # Influence profiles
            profiles_df = compute_influence_profiles(impact_matrix, valid_stations, targets, y)
            
            # Trace pathways
            trace_indirect_pathways(W, rho, valid_stations, targets)
            
            # Visualizations
            create_visualizations(impact_matrix, target_matrix, centrality_df, profiles_df,
                                  valid_stations, targets, target_indices, coords, meta_df, y)
            
            # Save results
            save_results(impact_matrix, target_matrix, centrality_df, profiles_df, valid_stations, targets)
            
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
