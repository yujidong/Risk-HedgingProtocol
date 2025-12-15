"""
Game Theory Simulation for Risk-Hedging Protocol
Based on trained model results from noise robustness experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

# ==========================================
# Step 0: Load JSON Data
# ==========================================

def load_experiment_results(json_file):
    """
    Load and extract data from JSON result file
    Returns: drop_rates, mean_r2, variance_r2
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract drop rates and corresponding metrics
    drop_rates = []
    mean_r2 = []
    variance_r2 = []
    
    # Parse keys like 'random_drop_0.00_data1.00'
    for key in sorted(data.keys()):
        if 'random_drop' in key:
            # Extract drop rate from key
            parts = key.split('_')
            drop_rate = float(parts[2])
            
            # Extract metrics
            r2_score = data[key]['r2_score']
            drop_rates.append(drop_rate)
            mean_r2.append(r2_score['mean'])
            variance_r2.append(r2_score['variance'])
    
    return np.array(drop_rates), np.array(mean_r2), np.array(variance_r2)


# ==========================================
# Step 1: Data Extraction
# ==========================================

# Get filename from command line or use default
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    # Automatically find the latest random_drop result file
    import glob
    files = sorted(glob.glob('output/data/random_drop_results*.json'), reverse=True)
    if files:
        json_file = files[0]
        print(f"Using latest result file: {json_file}")
    else:
        json_file = 'output/data/random_drop_results.json'
        print(f"Using default file: {json_file}")

# Load data
try:
    drop_rates, raw_mu, raw_sigma_sq = load_experiment_results(json_file)
    print(f"\nLoaded data from: {json_file}")
    print(f"Number of configurations: {len(drop_rates)}")
    print(f"Drop rate range: {drop_rates.min():.1f} - {drop_rates.max():.1f}")
    print(f"Mean R² range: {raw_mu.min():.4f} - {raw_mu.max():.4f}")
    print(f"Variance range: {raw_sigma_sq.min():.2e} - {raw_sigma_sq.max():.2e}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("\nUsage: python game_theory_simulation.py [json_file]")
    print("Example: python game_theory_simulation.py random_drop_results_20251210_150050.json")
    sys.exit(1)

# ==========================================
# Step 2: Curve Fitting
# ==========================================

# Define fitting function forms
def func_mu_decay(x, a, b, c):
    """Expected utility typically decreases linearly or quadratically with noise"""
    return a * x**2 + b * x + c

def func_sigma_growth(x, a, b):
    """Risk typically grows exponentially with noise (Variance Explosion)"""
    return a * np.exp(b * x)

# Perform curve fitting
try:
    popt_mu, _ = curve_fit(func_mu_decay, drop_rates, raw_mu)
    popt_sigma, _ = curve_fit(func_sigma_growth, drop_rates, raw_sigma_sq)
except Exception as e:
    print(f"Error during curve fitting: {e}")
    print("Trying with different initial parameters...")
    # Use better initial parameters
    popt_mu, _ = curve_fit(func_mu_decay, drop_rates, raw_mu, 
                           p0=[-0.5, -0.3, raw_mu[0]])
    popt_sigma, _ = curve_fit(func_sigma_growth, drop_rates, raw_sigma_sq, 
                              p0=[raw_sigma_sq[0], 5])

# Generate Continuous Functions for Simulation
def get_simulated_mu(delta):
    return func_mu_decay(delta, *popt_mu)

def get_simulated_var(delta):
    return func_sigma_growth(delta, *popt_sigma)

print(f"\n{'='*60}")
print("FITTED FUNCTIONS")
print(f"{'='*60}")
print(f"Mu Function: {popt_mu[0]:.4f}x² + {popt_mu[1]:.4f}x + {popt_mu[2]:.4f}")
print(f"Variance Function: {popt_sigma[0]:.6f} * e^({popt_sigma[1]:.2f}x)")

# ==========================================
# Step 3: Game Theoretic Simulation
# ==========================================

def simulate_market(lambda_B, lambda_S, delta_range):
    """
    Simulate market welfare under different risk aversion parameters
    Returns: welfare_spot, welfare_equity, collapse_idx
    """
    welfare_spot = []
    welfare_equity = []
    
    for d in delta_range:
        mu = get_simulated_mu(d)
        var = get_simulated_var(d)
        
        # --- Scenario A: Traditional Spot Trading ---
        # Alpha = 0, buyer bears all risk
        buyer_utility_spot = mu - lambda_B * var
        
        if buyer_utility_spot < 0:
            welfare_spot.append(0)
        else:
            welfare_spot.append(mu - lambda_B * var)

        # --- Scenario B: Risk-Hedging Protocol ---
        alpha_star = lambda_B / (lambda_B + lambda_S)
        reduced_risk_cost = var * (lambda_B * lambda_S) / (lambda_B + lambda_S)
        buyer_utility_equity = mu - reduced_risk_cost
        
        if mu - reduced_risk_cost < 0:
            welfare_equity.append(0)
        else:
            welfare_equity.append(mu - reduced_risk_cost)
    
    # Convert to numpy arrays
    welfare_spot = np.array(welfare_spot)
    welfare_equity = np.array(welfare_equity)
    
    # Find collapse point
    collapse_indices = np.where(welfare_spot <= 0)[0]
    collapse_idx = collapse_indices[0] if len(collapse_indices) > 0 else None
    
    return welfare_spot, welfare_equity, collapse_idx


# Market parameter settings - Four combinations
risk_scenarios = [
    {'lambda_B': 100, 'lambda_S': 40, 'label': 'Low Risk Aversion (λ_B=100, λ_S=40)'},
    {'lambda_B': 100, 'lambda_S': 300, 'label': 'Low Buyer, High Seller (λ_B=100, λ_S=300)'},
    {'lambda_B': 800, 'lambda_S': 40, 'label': 'High Buyer, Low Seller (λ_B=800, λ_S=40)'},
    {'lambda_B': 800, 'lambda_S': 300, 'label': 'High Risk Aversion (λ_B=800, λ_S=300)'},
]

print(f"\n{'='*60}")
print("GAME THEORY SIMULATIONS")
print(f"{'='*60}")

# Simulation range
delta_range = np.linspace(0, 0.9, 100)

# Run simulations for all scenarios
simulation_results = []
for scenario in risk_scenarios:
    lambda_B = scenario['lambda_B']
    lambda_S = scenario['lambda_S']
    print(f"\nScenario: {scenario['label']}")
    
    welfare_spot, welfare_equity, collapse_idx = simulate_market(lambda_B, lambda_S, delta_range)
    
    welfare_improvement = welfare_equity - welfare_spot
    total_recovered_loss = np.sum(welfare_improvement[welfare_improvement > 0])
    
    print(f"  Max welfare (Spot): {welfare_spot.max():.4f}")
    print(f"  Max welfare (Hedging): {welfare_equity.max():.4f}")
    print(f"  Recovered loss: {total_recovered_loss:.4f}")
    if collapse_idx is not None:
        print(f"  Collapse point: δ = {delta_range[collapse_idx]:.2f}")
    else:
        print(f"  No collapse in baseline")
    
    simulation_results.append({
        'scenario': scenario,
        'welfare_spot': welfare_spot,
        'welfare_equity': welfare_equity,
        'collapse_idx': collapse_idx
    })

# ==========================================
# Step 4: Visualization
# ==========================================

# ========== Figure 1: Curve Fitting Validation ==========
fig1, ax = plt.subplots(1, 1, figsize=(10, 6))
ax_twin = ax.twinx()

# Plot mean fitting (left axis)
ax.scatter(drop_rates, raw_mu, color='blue', s=100, alpha=0.7, 
           label='Observed Mean R²', zorder=5, edgecolors='darkblue', linewidths=1.5)
fitted_mu = get_simulated_mu(delta_range)
ax.plot(delta_range, fitted_mu, 'b-', linewidth=3, label='Fitted Mean Function', alpha=0.9)

# Plot variance fitting (right axis)
ax_twin.scatter(drop_rates, raw_sigma_sq, color='red', s=100, alpha=0.7, 
                marker='s', label='Observed Variance', zorder=5, edgecolors='darkred', linewidths=1.5)
fitted_var = get_simulated_var(delta_range)
ax_twin.plot(delta_range, fitted_var, 'r--', linewidth=3, label='Fitted Variance Function', alpha=0.9)

ax.set_xlabel('Noise Intensity δ (Drop Rate)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean R² Score (Utility μ)', fontsize=14, fontweight='bold', color='blue')
ax_twin.set_ylabel('Variance σ² (Volatility)', fontsize=14, fontweight='bold', color='red')
ax.set_title('Utility and Volatility Functions from Experimental Data', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='y', labelcolor='blue', labelsize=12)
ax_twin.tick_params(axis='y', labelcolor='red', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()

# Save Figure 1
from pathlib import Path
Path('output/figures').mkdir(parents=True, exist_ok=True)
base_name = Path(json_file).stem
output_file1 = f'output/figures/{base_name}_curve_fitting.png'
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print(f"Figure 1 saved to: {output_file1}")
print(f"{'='*60}")


# ========== Figure 2: Market Welfare Comparison (4 Scenarios) ==========
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors_spot = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']  # Blue
colors_hedge = ['#9467bd', '#9467bd', '#9467bd', '#9467bd']  # Purple

for idx, result in enumerate(simulation_results):
    ax = axes[idx]
    scenario = result['scenario']
    welfare_spot = result['welfare_spot']
    welfare_equity = result['welfare_equity']
    collapse_idx = result['collapse_idx']
    
    # Plot welfare curves
    ax.plot(delta_range, welfare_spot, '--', linewidth=2.5, 
            color=colors_spot[idx], label='Spot Trading (Baseline)', alpha=0.8)
    ax.plot(delta_range, welfare_equity, '-', linewidth=3, 
            color=colors_hedge[idx], label='Risk-Hedging Protocol', alpha=0.9)
    
    # Fill shaded area
    ax.fill_between(delta_range, welfare_spot, welfare_equity, 
                    where=(welfare_equity > welfare_spot),
                    color='green', alpha=0.15, label='Welfare Gain')
    
    # Mark collapse point
    if collapse_idx is not None:
        ax.axvline(x=delta_range[collapse_idx], color='gray', linestyle=':', 
                   linewidth=2, alpha=0.6)
        y_pos = max(welfare_equity) * 0.65
        # Place label on the left side of the collapse line
        ax.text(delta_range[collapse_idx] - 0.08, y_pos, 
                f'Collapse\nδ={delta_range[collapse_idx]:.2f}', 
                fontsize=9, color='darkred', ha='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Labels and title
    ax.set_xlabel('Noise Intensity δ (Drop Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Social Welfare', fontsize=12, fontweight='bold')
    ax.set_title(scenario['label'], fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 0.9])

fig2.suptitle('Market Robustness under Different Risk Preferences', 
              fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save Figure 2
output_file2 = f'output/figures/{base_name}_market_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Figure 2 saved to: {output_file2}")
print(f"{'='*60}")

plt.show()
