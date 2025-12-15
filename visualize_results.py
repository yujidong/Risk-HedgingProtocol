"""
Visualization Script for Noise Robustness Experiment Results
=============================================================
This script generates three publication-quality figures from the experimental results:
1. Random Drop Impact Analysis
2. Entropy Comparison (Low vs High Entropy Profiles)
3. Data Scarcity Impact Analysis

Author: Yuji Dong
Date: December 2025
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# Figure 1: Random Drop Impact Analysis
# ============================================================================

def plot_random_drop_analysis(results_file='output/data/random_drop_results.json', 
                               output_file='random_drop_enhanced.png',
                               add_timestamp=True):
    """Generate Random Drop impact visualization."""
    
    print("Generating Figure 1: Random Drop Analysis...")
    
    # Add timestamp to output file if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(output_file).stem
        extension = Path(output_file).suffix
        output_file = f"output/figures/{base_name}_{timestamp}{extension}"
    else:
        output_file = f"output/figures/{output_file}"
    
    # Ensure output directory exists
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    mean_r2 = []
    std_r2 = []
    ci_lower = []
    ci_upper = []
    
    for rate in drop_rates:
        key = f'random_drop_{rate:.2f}_data1.00'
        if key in results:
            mean = results[key]['r2_score']['mean']
            std = results[key]['r2_score']['std']
            mean_r2.append(mean)
            std_r2.append(std)
            ci_lower.append(mean - 1.96 * std)
            ci_upper.append(mean + 1.96 * std)
    
    # Calculate metrics relative to baseline
    baseline_r2 = mean_r2[0]
    baseline_std = std_r2[0]
    uncertainty_mults_full = [std/baseline_std for std in std_r2]
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ===== TOP PLOT: Dual Y-axis =====
    ax_top = fig.add_subplot(gs[0, :])
    
    color_perf = '#2E86AB'
    ax_top.set_xlabel('Random Drop Rate', fontsize=13, fontweight='bold')
    ax_top.set_ylabel('Mean R² Score', color=color_perf, fontsize=13, fontweight='bold')
    line1 = ax_top.plot(drop_rates, mean_r2, 'o-', color=color_perf, linewidth=3, 
                        markersize=10, label='Mean R²', alpha=0.8)
    ax_top.tick_params(axis='y', labelcolor=color_perf)
    ax_top.set_ylim([0.3, 0.85])
    
    # Uncertainty Multiplier (right y-axis)
    ax_top2 = ax_top.twinx()
    color_unc = '#E63946'
    ax_top2.set_ylabel('Uncertainty Multiplier (×)', color=color_unc, fontsize=13, fontweight='bold')
    line2 = ax_top2.plot(drop_rates, uncertainty_mults_full, 's-', color=color_unc, linewidth=3,
                         markersize=10, label='Uncertainty Multiplier', alpha=0.8)
    ax_top2.tick_params(axis='y', labelcolor=color_unc)
    ax_top2.set_ylim([0, 20])
    
    # Reference lines with minimal labels
    ax_top2.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_top2.axhline(y=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.3)
    ax_top2.axhline(y=10, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
    
    # Add minimal reference labels
    ax_top2.text(0.82, 1, '1×', fontsize=9, color='gray', alpha=0.8, va='bottom')
    ax_top2.text(0.82, 5, '5×', fontsize=9, color='orange', alpha=0.8, va='bottom')
    ax_top2.text(0.82, 10, '10×', fontsize=9, color='red', alpha=0.8, va='bottom')
    
    ax_top.set_title('Performance Degradation and Uncertainty Growth',
                    fontsize=14, fontweight='bold', pad=15)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_top.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax_top.set_xlim([-0.05, 0.85])
    
    # ===== BOTTOM LEFT: Performance with Multiple CIs =====
    ax_bl = fig.add_subplot(gs[1, 0])
    
    ci_68_lower = [m - 1.0 * s for m, s in zip(mean_r2, std_r2)]
    ci_68_upper = [m + 1.0 * s for m, s in zip(mean_r2, std_r2)]
    ci_99_lower = [m - 2.576 * s for m, s in zip(mean_r2, std_r2)]
    ci_99_upper = [m + 2.576 * s for m, s in zip(mean_r2, std_r2)]
    
    ax_bl.fill_between(drop_rates, ci_99_lower, ci_99_upper, alpha=0.15, 
                       color='#2E86AB', label='99% CI (2.58σ)')
    ax_bl.fill_between(drop_rates, ci_lower, ci_upper, alpha=0.2, 
                       color='#2E86AB', label='95% CI (1.96σ)')
    ax_bl.fill_between(drop_rates, ci_68_lower, ci_68_upper, alpha=0.3, 
                       color='#2E86AB', label='68% CI (1σ)')
    
    ax_bl.plot(drop_rates, mean_r2, 'o-', color='#2E86AB', linewidth=2.5, 
              markersize=8, label='Mean R²', alpha=0.8, zorder=10)
    
    ax_bl.axhline(y=baseline_r2, color='gray', linestyle='--', linewidth=1.5, 
                 alpha=0.5, label=f'Baseline: {baseline_r2:.3f}')
    
    ax_bl.set_xlabel('Random Drop Rate', fontsize=12, fontweight='bold')
    ax_bl.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax_bl.set_title('Confidence Intervals', 
                   fontsize=13, fontweight='bold')
    ax_bl.grid(True, alpha=0.3, linestyle='--')
    ax_bl.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
    ax_bl.set_ylim([0.2, 0.88])
    
    # ===== BOTTOM RIGHT: Uncertainty Growth =====
    ax_br = fig.add_subplot(gs[1, 1])
    
    ax_br.plot(drop_rates, std_r2, 's-', color='#E63946', linewidth=2.5, 
              markersize=8, label='Std Dev', alpha=0.8)
    
    ax_br.axhline(y=baseline_std, color='gray', linestyle='--', linewidth=1.5,
                 alpha=0.5, label=f'Baseline: {baseline_std:.4f}')
    
    # Shade regions
    ax_br.axhspan(0, baseline_std * 2, alpha=0.1, color='green', zorder=0)
    ax_br.axhspan(baseline_std * 2, baseline_std * 5, alpha=0.1, color='orange', zorder=0)
    ax_br.axhspan(baseline_std * 5, ax_br.get_ylim()[1], alpha=0.1, color='red', zorder=0)
    
    # Add minimal zone labels
    ax_br.text(0.05, baseline_std * 1.2, 'Low', fontsize=8, style='italic', 
              color='green', alpha=0.6)
    ax_br.text(0.05, baseline_std * 3, 'Moderate', fontsize=8, style='italic',
              color='orange', alpha=0.6)
    ax_br.text(0.05, baseline_std * 6, 'High', fontsize=8, style='italic',
              color='red', alpha=0.6)
    
    ax_br.set_xlabel('Random Drop Rate', fontsize=12, fontweight='bold')
    ax_br.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax_br.set_title('Uncertainty Growth', fontsize=13, fontweight='bold')
    ax_br.grid(True, alpha=0.3, linestyle='--')
    ax_br.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    
    fig.suptitle('Impact of Random Drop Noise on Model Performance and Uncertainty',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ============================================================================
# Figure 2: Entropy Comparison
# ============================================================================

def plot_entropy_comparison(results_file='output/data/random_drop_results.json',
                            output_file='entropy_comparison_enhanced.png',
                            add_timestamp=True):
    """Generate entropy comparison visualization."""
    
    print("Generating Figure 2: Entropy Comparison...")
    
    # Add timestamp to output file if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(output_file).stem
        extension = Path(output_file).suffix
        output_file = f"output/figures/{base_name}_{timestamp}{extension}"
    else:
        output_file = f"output/figures/{output_file}"
    
    # Ensure output directory exists
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get data for two extreme configurations
    baseline_config = 'random_drop_0.00_data1.00'
    high_entropy_config = 'random_drop_0.70_data1.00'
    
    baseline_values = results[baseline_config]['r2_score']['all_values']
    high_entropy_values = results[high_entropy_config]['r2_score']['all_values']
    
    baseline_mean = results[baseline_config]['r2_score']['mean']
    baseline_std = results[baseline_config]['r2_score']['std']
    high_entropy_mean = results[high_entropy_config]['r2_score']['mean']
    high_entropy_std = results[high_entropy_config]['r2_score']['std']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create smooth distributions
    x_range = np.linspace(0.4, 0.85, 1000)
    baseline_kde = stats.norm.pdf(x_range, baseline_mean, baseline_std)
    high_entropy_kde = stats.norm.pdf(x_range, high_entropy_mean, high_entropy_std)
    
    # Normalize
    baseline_kde_norm = baseline_kde / baseline_kde.max()
    high_entropy_kde_norm = high_entropy_kde / high_entropy_kde.max()
    
    # Plot distributions
    color_baseline = '#2E86AB'
    color_high = '#E63946'
    
    ax.fill_between(x_range, baseline_kde_norm, alpha=0.3, color=color_baseline, 
                    label=f'Low-Entropy Profile (Baseline)')
    ax.plot(x_range, baseline_kde_norm, linewidth=3, color=color_baseline)
    
    ax.fill_between(x_range, high_entropy_kde_norm, alpha=0.3, color=color_high,
                    label=f'High-Entropy Profile (Random Drop 0.7)')
    ax.plot(x_range, high_entropy_kde_norm, linewidth=3, color=color_high)
    
    # Add mean lines
    ax.axvline(baseline_mean, color=color_baseline, linestyle='--', 
              linewidth=2, alpha=0.7)
    ax.axvline(high_entropy_mean, color=color_high, linestyle='--', 
              linewidth=2, alpha=0.7)
    
    # Add confidence interval shading
    ci_baseline = 1.96 * baseline_std
    ci_high = 1.96 * high_entropy_std
    ax.axvspan(baseline_mean - ci_baseline, baseline_mean + ci_baseline, 
              alpha=0.1, color=color_baseline, zorder=0)
    ax.axvspan(high_entropy_mean - ci_high, high_entropy_mean + ci_high,
              alpha=0.1, color=color_high, zorder=0)
    
    # Add text annotations with key metrics
    stats_baseline = (f'Baseline\nμ={baseline_mean:.3f}, σ={baseline_std:.4f}')
    ax.text(0.80, 0.92, stats_baseline, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor=color_baseline, linewidth=2, alpha=0.9))
    
    stats_high = (f'Drop 0.7\nμ={high_entropy_mean:.3f}, σ={high_entropy_std:.4f}')
    ax.text(0.05, 0.92, stats_high, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=color_high, linewidth=2, alpha=0.9))
    
    # Add comparison at bottom
    uncertainty_mult = high_entropy_std / baseline_std
    perf_drop = (1 - high_entropy_mean/baseline_mean) * 100
    comparison_text = f'Uncertainty: {uncertainty_mult:.1f}×  |  Performance Drop: {perf_drop:.1f}%'
    ax.text(0.5, -0.10, comparison_text, transform=ax.transAxes,
           fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # Add comparison metrics
    uncertainty_mult = high_entropy_std / baseline_std
    perf_drop = (1 - high_entropy_mean/baseline_mean) * 100
    range_baseline = max(baseline_values) - min(baseline_values)
    range_high = max(high_entropy_values) - min(high_entropy_values)
    range_mult = range_high / range_baseline
    
    ax.set_xlabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Probability Density', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Comparison: Baseline vs High Noise',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0.45, 0.85])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper center', fontsize=11, frameon=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ============================================================================
# Figure 3: Data Scarcity Impact Analysis
# ============================================================================

def plot_data_scarcity_analysis(results_file='output/data/data_scarcity_results.json',
                                output_file='data_scarcity_enhanced.png',
                                add_timestamp=True):
    """Generate data scarcity impact visualization."""
    
    print("Generating Figure 3: Data Scarcity Analysis...")
    
    # Add timestamp to output file if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(output_file).stem
        extension = Path(output_file).suffix
        output_file = f"output/figures/{base_name}_{timestamp}{extension}"
    else:
        output_file = f"output/figures/{output_file}"
    
    # Ensure output directory exists
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    drift_intensities = [0.30, 0.40, 0.50, 0.60]
    data_ratios = [1.0, 0.7, 0.5, 0.3]
    
    colors = {0.30: '#2E86AB', 0.40: '#A23B72', 0.50: '#F18F01', 0.60: '#C73E1D'}
    markers = {0.30: 'o', 0.40: 's', 0.50: '^', 0.60: 'D'}
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ===== TOP: Trade-off Analysis =====
    ax_top = fig.add_subplot(gs[0, :])
    
    max_uncertainty = 0  # Track maximum uncertainty for y-axis
    
    for drift in drift_intensities:
        baseline_key = f'drift_{drift:.2f}_data1.00'
        if baseline_key not in results:
            continue
        
        baseline_r2 = results[baseline_key]['r2_score']['mean']
        baseline_std = results[baseline_key]['r2_score']['std']
        
        perf_drops = []
        uncertainty_multipliers = []
        ratio_labels = []
        
        for ratio in [0.7, 0.5, 0.3]:
            key = f'drift_{drift:.2f}_data{ratio:.2f}'
            if key in results:
                mean_r2 = results[key]['r2_score']['mean']
                std_dev = results[key]['r2_score']['std']
                
                perf_drop = (1 - mean_r2/baseline_r2) * 100
                uncertainty_mult = std_dev / baseline_std
                
                perf_drops.append(perf_drop)
                uncertainty_multipliers.append(uncertainty_mult)
                ratio_labels.append(ratio)
                
                max_uncertainty = max(max_uncertainty, uncertainty_mult)
        
        if perf_drops:
            ax_top.plot(perf_drops, uncertainty_multipliers, 
                       marker=markers[drift], color=colors[drift], 
                       linewidth=3, markersize=12, label=f'Drift {drift:.2f}',
                       alpha=0.8)
            
            # Annotate key points for clarity
            # First point (0.7 ratio) and last point (0.3 ratio)
            if len(perf_drops) >= 1:
                # First point
                ax_top.annotate(f'0.7',
                              xy=(perf_drops[0], uncertainty_multipliers[0]),
                              xytext=(-8, 8), textcoords='offset points',
                              fontsize=8, color=colors[drift], alpha=0.8)
                # Last point with drift label
                ax_top.annotate(f'{drift:.2f}\n0.3',
                              xy=(perf_drops[-1], uncertainty_multipliers[-1]),
                              xytext=(8, 0), textcoords='offset points',
                              fontsize=8, fontweight='bold',
                              color=colors[drift], alpha=0.9)
    
    # Reference lines - simplified
    ax_top.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax_top.axhline(y=2, color='orange', linestyle=':', linewidth=1.5, alpha=0.3)
    ax_top.axhline(y=4, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
    
    # Add subtle note about data ratios
    ax_top.text(0.98, 0.03, 'Data ratios: 0.7 → 0.5 → 0.3', 
               transform=ax_top.transAxes, fontsize=9, ha='right', 
               style='italic', alpha=0.6)
    
    ax_top.set_xlabel('Performance Drop (%)', fontsize=13, fontweight='bold')
    ax_top.set_ylabel('Uncertainty Multiplier (×)', fontsize=13, fontweight='bold')
    ax_top.set_title('Performance vs Uncertainty Trade-off',
                    fontsize=14, fontweight='bold', pad=15)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax_top.set_xlim([0, 20])
    # Dynamic y-axis based on actual data, with 10% padding
    y_max = max(9, max_uncertainty * 1.1)
    ax_top.set_ylim([0, y_max])
    
    # ===== BOTTOM LEFT: Bar Chart with CIs =====
    ax_bl = fig.add_subplot(gs[1, 0])
    
    x_positions = {0.30: 0, 0.40: 1, 0.50: 2, 0.60: 3}
    bar_width = 0.18
    ratio_offsets = {1.0: -1.5*bar_width, 0.7: -0.5*bar_width, 
                    0.5: 0.5*bar_width, 0.3: 1.5*bar_width}
    
    for ratio in data_ratios:
        means = []
        stds = []
        positions = []
        
        for drift in drift_intensities:
            key = f'drift_{drift:.2f}_data{ratio:.2f}'
            if key in results:
                means.append(results[key]['r2_score']['mean'])
                stds.append(results[key]['r2_score']['std'])
                positions.append(x_positions[drift] + ratio_offsets[ratio])
        
        alpha = ratio
        ax_bl.bar(positions, means, bar_width, yerr=[1.96*s for s in stds],
                 label=f'Data {ratio:.1f}', alpha=0.5 + alpha*0.3,
                 capsize=3, error_kw={'linewidth': 1.5})
    
    ax_bl.set_ylabel('Mean R² Score', fontsize=12, fontweight='bold')
    ax_bl.set_xlabel('Drift Intensity', fontsize=12, fontweight='bold')
    ax_bl.set_title('Performance with 95% Confidence Intervals', fontsize=13, fontweight='bold')
    ax_bl.set_xticks([0, 1, 2, 3])
    ax_bl.set_xticklabels(['0.30', '0.40', '0.50', '0.60'])
    ax_bl.legend(fontsize=9, ncol=2)
    ax_bl.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_bl.set_ylim([0.55, 0.85])
    
    # ===== BOTTOM RIGHT: Heatmap =====
    ax_br = fig.add_subplot(gs[1, 1])
    
    matrix = np.zeros((len(data_ratios), len(drift_intensities)))
    for i, ratio in enumerate(data_ratios):
        for j, drift in enumerate(drift_intensities):
            key = f'drift_{drift:.2f}_data{ratio:.2f}'
            if key in results:
                matrix[i, j] = results[key]['r2_score']['std']
    
    im = ax_br.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.055)
    
    for i in range(len(data_ratios)):
        for j in range(len(drift_intensities)):
            text = ax_br.text(j, i, f'{matrix[i, j]:.3f}',
                             ha='center', va='center', color='black',
                             fontsize=10, fontweight='bold')
    
    ax_br.set_xticks(np.arange(len(drift_intensities)))
    ax_br.set_yticks(np.arange(len(data_ratios)))
    ax_br.set_xticklabels([f'{d:.2f}' for d in drift_intensities])
    ax_br.set_yticklabels([f'{r:.1f}' for r in data_ratios])
    ax_br.set_xlabel('Drift Intensity', fontsize=12, fontweight='bold')
    ax_br.set_ylabel('Data Availability', fontsize=12, fontweight='bold')
    ax_br.set_title('Uncertainty Heatmap (Std Dev)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax_br)
    cbar.set_label('Std Dev', rotation=270, labelpad=20, fontweight='bold')
    
    fig.suptitle('Impact of Training Data Availability on Model Performance and Uncertainty',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import sys
    import glob
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Check for result files
    random_drop_files = sorted(glob.glob('output/data/random_drop_results*.json'), reverse=True)
    data_scarcity_files = sorted(glob.glob('output/data/data_scarcity_results*.json'), reverse=True)
    
    # Use latest files by default, or allow user to specify
    random_drop_file = 'output/data/random_drop_results.json'
    data_scarcity_file = 'output/data/data_scarcity_results.json'
    
    if len(sys.argv) > 1:
        # User specified a timestamp
        timestamp = sys.argv[1]
        random_drop_file = f'output/data/random_drop_results_{timestamp}.json'
        data_scarcity_file = f'output/data/data_scarcity_results_{timestamp}.json'
        print(f"Using results from session: {timestamp}")
    elif random_drop_files:
        # Use latest timestamped files
        random_drop_file = random_drop_files[0]
        if data_scarcity_files:
            data_scarcity_file = data_scarcity_files[0]
        print(f"Using latest result files:")
        print(f"  Random Drop: {random_drop_file}")
        print(f"  Data Scarcity: {data_scarcity_file}")
    else:
        print("Using default result files (no timestamp)")
    
    print("="*80)
    
    # Extract timestamps from data file names
    import re
    
    # Extract timestamp from random_drop_file
    random_drop_timestamp = None
    match = re.search(r'_(\d{8}_\d{6})\.json$', random_drop_file)
    if match:
        random_drop_timestamp = match.group(1)
    
    # Extract timestamp from data_scarcity_file
    data_scarcity_timestamp = None
    match = re.search(r'_(\d{8}_\d{6})\.json$', data_scarcity_file)
    if match:
        data_scarcity_timestamp = match.group(1)
    
    # Determine output timestamp strategy
    if random_drop_timestamp and data_scarcity_timestamp:
        if random_drop_timestamp == data_scarcity_timestamp:
            # Both from same session - perfect!
            output_timestamp = random_drop_timestamp
            print(f"Using timestamp from data files: {output_timestamp}")
        else:
            # Different timestamps - warn user and use both
            print(f"WARNING: Data files are from different sessions!")
            print(f"  Random Drop: {random_drop_timestamp}")
            print(f"  Data Scarcity: {data_scarcity_timestamp}")
            output_timestamp = f"{random_drop_timestamp}_and_{data_scarcity_timestamp}"
            print(f"Output files will use combined timestamp: {output_timestamp}")
    elif random_drop_timestamp:
        output_timestamp = random_drop_timestamp
        print(f"Using timestamp from random_drop file: {output_timestamp}")
    elif data_scarcity_timestamp:
        output_timestamp = data_scarcity_timestamp
        print(f"Using timestamp from data_scarcity file: {output_timestamp}")
    else:
        # No timestamp found, use current time
        print("No timestamp found in data files, using current time")
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate all three figures with the same timestamp as data
    random_drop_png = f'random_drop_enhanced_{output_timestamp}.png'
    entropy_png = f'entropy_comparison_enhanced_{output_timestamp}.png'
    data_scarcity_png = f'data_scarcity_enhanced_{output_timestamp}.png'
    
    plot_random_drop_analysis(results_file=random_drop_file, output_file=random_drop_png, add_timestamp=False)
    plot_entropy_comparison(results_file=random_drop_file, output_file=entropy_png, add_timestamp=False)
    plot_data_scarcity_analysis(results_file=data_scarcity_file, output_file=data_scarcity_png, add_timestamp=False)
    
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {random_drop_png}")
    print(f"  2. {entropy_png}")
    print(f"  3. {data_scarcity_png}")
    print("="*80)
