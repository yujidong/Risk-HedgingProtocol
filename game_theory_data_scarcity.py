"""
Game Theory Simulation for Data Scarcity Scenario
Comparative Analysis: Large Buyer vs. SME Buyer
Based on trained model results from data scarcity experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
import glob

# ==========================================
# Step 0: Load Data Scarcity Results
# ==========================================

def load_scarcity_data(json_file):
    """
    Load and extract data from data scarcity JSON result file
    Extract variance functions for rich (100% data) and poor (30% data) buyers
    They face the SAME seller noise (drift), but react differently due to their data foundation
    Returns: drift_levels, rich_buyer_data, poor_buyer_data
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Focus on drift variation (seller's noise level)
    # Different buyers react differently to the same noise
    drift_levels = [0.3, 0.4, 0.5, 0.6]
    
    # Storage for two buyer types facing the same seller
    rich_buyer = {'mu': [], 'var': [], 'drift': []}   # 100% data foundation
    poor_buyer = {'mu': [], 'var': [], 'drift': []}   # 30% data foundation
    
    for drift in drift_levels:
        # Construct keys for different data availability levels
        key_rich = f"drift_{drift:.2f}_data1.00"
        key_poor = f"drift_{drift:.2f}_data0.30"
        
        # Check if keys exist
        if key_rich in data and key_poor in data:
            # Data-Rich Incumbent (e.g., Google Maps with massive historical data)
            # Can absorb noise better, lower variance
            rich_buyer['drift'].append(drift)
            rich_buyer['mu'].append(data[key_rich]['r2_score']['mean'])
            rich_buyer['var'].append(data[key_rich]['r2_score']['variance'])
            
            # Data-Scarce Entrant (e.g., startup with limited data)
            # Same seller noise hits them harder, High Variance Exposure
            poor_buyer['drift'].append(drift)
            poor_buyer['mu'].append(data[key_poor]['r2_score']['mean'])
            poor_buyer['var'].append(data[key_poor]['r2_score']['variance'])
    
    return np.array(drift_levels), rich_buyer, poor_buyer


# ==========================================
# Step 1: Load JSON Data
# ==========================================

# Get filename from command line or use default
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    # Automatically find the latest data_scarcity result file
    files = sorted(glob.glob('output/data/data_scarcity_results*.json'), reverse=True)
    if files:
        json_file = files[0]
        print(f"Using latest result file: {json_file}")
    else:
        json_file = 'output/data/data_scarcity_results.json'
        print(f"Using default file: {json_file}")

# Load data
try:
    drift_levels, rich_data, poor_data = load_scarcity_data(json_file)
    print(f"\nLoaded data from: {json_file}")
    print(f"Number of drift levels (seller noise): {len(drift_levels)}")
    print(f"Drift range: {drift_levels.min():.1f} - {drift_levels.max():.1f}")
    print(f"\nData-Rich Incumbent (100% data foundation, e.g., Google Maps):")
    print(f"  Mean R² range: {min(rich_data['mu']):.4f} - {max(rich_data['mu']):.4f}")
    print(f"  Variance range: {min(rich_data['var']):.2e} - {max(rich_data['var']):.2e}")
    print(f"\nData-Scarce Entrant (30% data foundation, e.g., Startup):")
    print(f"  Mean R² range: {min(poor_data['mu']):.4f} - {max(poor_data['mu']):.4f}")
    print(f"  Variance range: {min(poor_data['var']):.2e} - {max(poor_data['var']):.2e}")
    
    # Calculate variance multiplier (how much worse Data-Scarce Entrants suffer)
    var_multiplier = np.array(poor_data['var']) / np.array(rich_data['var'])
    print(f"\nHigh Variance Exposure Factor (Poor/Rich): {var_multiplier.min():.1f}x - {var_multiplier.max():.1f}x")
except Exception as e:
    print(f"Error loading data: {e}")
    print("\nUsage: python game_theory_data_scarcity.py [json_file]")
    print("Example: python game_theory_data_scarcity.py data_scarcity_results_20251210_181251.json")
    sys.exit(1)


# ==========================================
# Step 2: Curve Fitting
# ==========================================

def func_mu_decay(x, a, b, c):
    """Utility typically decreases with drift"""
    return a * x**2 + b * x + c

def func_sigma_growth(x, a, b):
    """Risk typically grows with drift"""
    return a * np.exp(b * x)

# Note: We assume both buyers see similar utility degradation (same seller data quality)
# But variance (risk sensitivity) differs dramatically due to their data foundation

# Use Data-Rich Incumbent's utility as baseline (similar for both in our model)
try:
    popt_mu_baseline, _ = curve_fit(func_mu_decay, drift_levels, rich_data['mu'])
except:
    popt_mu_baseline, _ = curve_fit(func_mu_decay, drift_levels, rich_data['mu'], 
                                     p0=[-0.1, -0.1, rich_data['mu'][0]])

# Fit variance for Data-Rich Incumbent (low variance baseline)
try:
    popt_var_rich, _ = curve_fit(func_sigma_growth, drift_levels, rich_data['var'])
except:
    popt_var_rich, _ = curve_fit(func_sigma_growth, drift_levels, rich_data['var'],
                                  p0=[rich_data['var'][0], 3])

# Fit variance for Data-Scarce Entrant (High Variance Exposure)
try:
    popt_var_poor, _ = curve_fit(func_sigma_growth, drift_levels, poor_data['var'])
except:
    popt_var_poor, _ = curve_fit(func_sigma_growth, drift_levels, poor_data['var'],
                                  p0=[poor_data['var'][0], 3])

print(f"\n{'='*60}")
print("FITTED FUNCTIONS")
print(f"{'='*60}")
print(f"Baseline Utility (same for both): {popt_mu_baseline[0]:.4f}x² + {popt_mu_baseline[1]:.4f}x + {popt_mu_baseline[2]:.4f}")
print(f"Data-Rich Incumbent Variance: {popt_var_rich[0]:.6f} * e^({popt_var_rich[1]:.2f}x)")
print(f"Data-Scarce Entrant Variance: {popt_var_poor[0]:.6f} * e^({popt_var_poor[1]:.2f}x)")
print(f"Variance ratio at δ=0.6: {func_sigma_growth(0.6, *popt_var_poor) / func_sigma_growth(0.6, *popt_var_rich):.1f}x")


# ==========================================
# Step 3: Game Theoretic Simulation
# ==========================================

def simulate_welfare(drift_x, mu_func, var_func, lambda_B, lambda_S):
    """
    Simulate market welfare for given buyer parameters
    Returns: welfare_spot, welfare_equity, collapse_idx
    """
    welfare_spot = []
    welfare_equity = []
    
    for d in drift_x:
        mu = mu_func(d)
        var = var_func(d)
        
        # Scenario A: Spot Trading
        risk_spot = lambda_B * var
        wf_spot = mu - risk_spot
        welfare_spot.append(max(0, wf_spot))
        
        # Scenario B: Risk-Hedging Protocol
        risk_equity = var * (lambda_B * lambda_S) / (lambda_B + lambda_S)
        wf_equity = mu - risk_equity
        welfare_equity.append(max(0, wf_equity))
    
    welfare_spot = np.array(welfare_spot)
    welfare_equity = np.array(welfare_equity)
    
    # Find collapse point
    collapse_indices = np.where(welfare_spot <= 0)[0]
    collapse_idx = collapse_indices[0] if len(collapse_indices) > 0 else None
    
    return welfare_spot, welfare_equity, collapse_idx


# Market parameters
lambda_B = 800  # Buyer risk aversion
lambda_S = 150  # Seller risk aversion

print(f"\n{'='*60}")
print("GAME THEORY PARAMETERS")
print(f"{'='*60}")
print(f"Buyer risk aversion (λ_B): {lambda_B}")
print(f"Seller risk aversion (λ_S): {lambda_S}")

# Create continuous drift range for smooth curves - START FROM ZERO!
drift_continuous = np.linspace(0, 0.9, 150)  # Full lifecycle view

# Generate continuous functions
def get_mu_baseline(d):
    """Baseline utility function (same seller data quality for both buyers)"""
    return func_mu_decay(d, *popt_mu_baseline)

def get_var_rich(d):
    """Data-Rich Incumbent's low variance (strong data foundation)"""
    return func_sigma_growth(d, *popt_var_rich)

def get_var_poor(d):
    """Data-Scarce Entrant's high variance (weak data foundation -> High Variance Exposure)"""
    return func_sigma_growth(d, *popt_var_poor)

# Simulate for Data-Rich Incumbent (100% data foundation)
# Strong foundation -> can absorb seller's noise
wf_spot_rich, wf_equity_rich, collapse_rich = simulate_welfare(
    drift_continuous, get_mu_baseline, get_var_rich, lambda_B, lambda_S
)

# Simulate for Data-Scarce Entrant (30% data foundation) in Spot Market
# Weak foundation -> same noise causes High Variance Exposure -> early collapse
wf_spot_poor, wf_equity_poor, collapse_poor = simulate_welfare(
    drift_continuous, get_mu_baseline, get_var_poor, lambda_B, lambda_S
)

print(f"\n{'='*60}")
print("SIMULATION RESULTS")
print(f"{'='*60}")
print(f"\nData-Rich Incumbent (e.g., Google Maps with massive data):")
print(f"  Max welfare (Spot): {wf_spot_rich.max():.4f}")
print(f"  Max welfare (Hedging): {wf_equity_rich.max():.4f}")
if collapse_rich is not None:
    print(f"  Collapse point: drift = {drift_continuous[collapse_rich]:.2f}")
else:
    print(f"  No collapse in baseline (resilient to noise)")

print(f"\nData-Scarce Entrant (e.g., Startup with limited data):")
print(f"  Max welfare (Spot): {wf_spot_poor.max():.4f}")
print(f"  Max welfare (Hedging): {wf_equity_poor.max():.4f}")
if collapse_poor is not None:
    print(f"  Collapse point (Spot): drift = {drift_continuous[collapse_poor]:.2f}")
    print(f"  -> Dies much earlier due to High Variance Exposure!")
else:
    print(f"  No collapse in baseline")

# Calculate welfare recovery
welfare_recovery = wf_equity_poor - wf_spot_poor
total_recovery = np.sum(welfare_recovery[welfare_recovery > 0])
print(f"\nTotal welfare recovered for Data-Scarce Entrant: {total_recovery:.4f}")
print(f"Protocol enables Viability from δ={drift_continuous[collapse_poor]:.2f} to δ={drift_continuous.max():.2f}")


# ==========================================
# Step 4: Visualization (Publication Quality)
# ==========================================

import matplotlib.patches as mpatches

# Set global style for academic publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ========== Figure 1: Curve Fitting Validation ==========
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Mean R² (Utility) - similar for both
ax1.scatter(drift_levels, rich_data['mu'], color='blue', s=100, alpha=0.7,
            label='Data-Rich Incumbent (100% data)', zorder=5, edgecolors='darkblue', linewidths=1.5)
ax1.scatter(drift_levels, poor_data['mu'], color='red', s=100, alpha=0.7,
            marker='s', label='Data-Scarce Entrant (30% data)', zorder=5, edgecolors='darkred', linewidths=1.5)

# Fit linear trends for both
from numpy.polynomial import Polynomial
p_rich = Polynomial.fit(drift_levels, rich_data['mu'], 1)
p_poor = Polynomial.fit(drift_levels, poor_data['mu'], 1)

# Get slopes
slope_rich = p_rich.convert().coef[1]
slope_poor = p_poor.convert().coef[1]

# Plot linear trends
ax1.plot(drift_continuous, p_rich(drift_continuous), 
         'b--', linewidth=1.5, alpha=0.6, label=f'Rich Trend (slope={slope_rich:.3f})', zorder=2)
ax1.plot(drift_continuous, p_poor(drift_continuous), 
         'r--', linewidth=1.5, alpha=0.6, label=f'Poor Trend (slope={slope_poor:.3f})', zorder=2)

# Add annotation comparing slopes
slope_diff_pct = abs((slope_rich - slope_poor) / slope_rich * 100)
ax1.text(0.55, 0.82, f'Slope Difference: {slope_diff_pct:.1f}%\n(Nearly Parallel)', 
         fontsize=10, color='black', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1.5))

ax1.set_xlabel('Noise Intensity δ (Seller Drift)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Mean R² Score (Utility μ)', fontsize=13, fontweight='bold')
ax1.set_title('Utility: Parallel Decline Under Same Seller Noise', fontsize=14, fontweight='bold')
ax1.set_ylim(0.60, 0.85)  # Extended y-axis for better visibility
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')

# Right panel: Variance (Risk) - HUGE difference!
ax2.scatter(drift_levels, rich_data['var'], color='blue', s=100, alpha=0.7,
            label='Data-Rich Incumbent (Low Variance)', zorder=5, edgecolors='darkblue', linewidths=1.5)
ax2.scatter(drift_levels, poor_data['var'], color='red', s=100, alpha=0.7,
            marker='s', label='Data-Scarce Entrant (High Variance Exposure!)', zorder=5, edgecolors='darkred', linewidths=1.5)
ax2.plot(drift_continuous, get_var_rich(drift_continuous), 
         'b-', linewidth=2, alpha=0.8)
ax2.plot(drift_continuous, get_var_poor(drift_continuous), 
         'r--', linewidth=2, alpha=0.8)

# Add multiplier annotation
max_idx = len(drift_levels) - 1
multiplier_at_max = poor_data['var'][max_idx] / rich_data['var'][max_idx]
ax2.annotate(f'{multiplier_at_max:.0f}× variance\nratio at δ=0.6',
            xy=(drift_levels[max_idx], poor_data['var'][max_idx]),
            xytext=(drift_levels[max_idx] - 0.08, poor_data['var'][max_idx] * 1.8),
            fontsize=10, color='black', fontweight='normal',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax2.set_xlabel('Noise Intensity δ (Seller Drift)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Variance σ² (Risk)', fontsize=13, fontweight='bold')
ax2.set_title('Risk: Data Foundation Determines Resilience', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_yscale('log')  # Log scale to show the explosion

plt.tight_layout()

from pathlib import Path
Path('output/figures').mkdir(parents=True, exist_ok=True)
base_name = Path(json_file).stem
output_file1 = f'output/figures/{base_name}_fitting_comparison.png'
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print(f"Figure 1 saved to: {output_file1}")
print(f"{'='*60}")


# ========== Figure 2: The "Market Fairness" Plot (Publication Version) ==========
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# 1. Data-Rich Incumbent (The "Privileged" Reference) - Gray background
ax.plot(drift_continuous, wf_spot_rich, color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.5,
        label='Data-Rich Incumbent (Spot Baseline)', zorder=1)

# 2. Data-Scarce Entrant - Spot Market (The "Victim") - Red dashed showing failure
ax.plot(drift_continuous, wf_spot_poor, color='#d62728', linestyle='--', linewidth=2, alpha=0.8,
        label='Data-Scarce Entrant (Spot - Premature Market Exit)', zorder=2)

# 3. Data-Scarce Entrant - Risk-Hedging Protocol (The "Solution") - Bold blue solid
ax.plot(drift_continuous, wf_equity_poor, color='#1f77b4', linestyle='-', linewidth=2.5,
        label='Data-Scarce Entrant (Risk-Hedging Protocol)', zorder=4)

# 4. Highlight "Structural Discrimination Gap" (Between Rich and Poor in Spot Market)
ax.fill_between(drift_continuous, wf_spot_poor, wf_spot_rich, 
                where=(wf_spot_rich > wf_spot_poor),
                color='#7f7f7f', alpha=0.15, hatch='//', 
                label='Data Scarcity Gap\n(Structural Discrimination)', zorder=0)

# 5. Highlight "Inclusion Zone" (Where Protocol Saves Data-Scarce Entrant)
ax.fill_between(drift_continuous, 0, wf_equity_poor, 
                where=(wf_equity_poor > wf_spot_poor) & (wf_equity_poor > 0),
                color='#2ca02c', alpha=0.25, 
                label='Welfare Recovered\n(Market Inclusion)', zorder=0)

# 6. Critical Annotations (Minimal and clean)

# Mark Early Collapse of Data-Scarce Entrant
if collapse_poor is not None:
    x_collapse = drift_continuous[collapse_poor]
    ax.axvline(x=x_collapse, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    ax.scatter([x_collapse], [0], color='#d62728', s=100, zorder=10, marker='X', edgecolors='darkred', linewidths=1.5)
    
    ax.annotate(f'Early Exit\n(Discriminatory\nCollapse)\nδ≈{x_collapse:.2f}', 
                xy=(x_collapse, 0), xytext=(x_collapse + 0.08, 0.18),
                color='#d62728', fontweight='bold', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5, connectionstyle="arc3,rad=.2"))

# Mark Extended Viability under Protocol
Viability_indices = np.where(wf_equity_poor > 0.01)[0]  # Non-trivial welfare
if len(Viability_indices) > 0 and collapse_poor is not None:
    Viability_end = drift_continuous[Viability_indices[-1]]
    if Viability_end > x_collapse:
        ax.annotate(f'Viability Extended\nto δ≈{Viability_end:.2f}', 
                    xy=(Viability_end, wf_equity_poor[Viability_indices[-1]]), 
                    xytext=(Viability_end - 0.18, 0.35),
                    color='#1f77b4', fontweight='bold', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5, connectionstyle="arc3,rad=-.2"))

# 7. Final Polish
ax.set_xlabel('Noise Intensity δ (Seller Drift)', fontweight='bold', fontsize=13)
ax.set_ylabel('Social Welfare', fontweight='bold', fontsize=13)
ax.set_title('Mitigating the Scarcity Multiplier:\nEnhancing Market Fairness for Data-Scarce Buyers', 
             fontweight='bold', fontsize=15, pad=15)

# Custom Legend - reorder to tell the story
handles, labels = ax.get_legend_handles_labels()
# Order: Rich baseline -> Poor spot -> Poor protocol -> Discrimination -> Inclusion
order = [0, 1, 2, 3, 4]
ax.legend([handles[i] for i in order], [labels[i] for i in order], 
          loc='upper right', frameon=True, framealpha=0.95, edgecolor='gray', fontsize=10)

ax.grid(True, linestyle=':', alpha=0.5)
ax.set_xlim(0, 0.8)  # Full lifecycle view
ax.set_ylim(0, max(wf_spot_rich.max(), wf_equity_poor.max()) * 1.08)

plt.tight_layout()

output_file2 = f'output/figures/{base_name}_market_fairness.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Figure 2 saved to: {output_file2}")
print(f"{'='*60}")

plt.show()
