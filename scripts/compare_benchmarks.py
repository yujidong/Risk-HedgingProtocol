"""
Benchmark Comparison Script
Analyzes and compares benchmark results across different testnets

Usage: python scripts/compare_benchmarks.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import statistics

def load_benchmark_files(benchmark_dir: str) -> Dict[str, List[dict]]:
    """Load all benchmark JSON files grouped by network"""
    results = {}
    benchmark_path = Path(benchmark_dir)
    
    if not benchmark_path.exists():
        print(f"⚠️  Benchmark directory not found: {benchmark_dir}")
        return results
    
    for file in benchmark_path.glob("benchmark_*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            network = data.get('network', 'unknown')
            
            if network not in results:
                results[network] = []
            results[network].append(data)
    
    return results

def analyze_network(network_name: str, benchmarks: List[dict]) -> dict:
    """Analyze benchmarks for a single network"""
    if not benchmarks:
        return {}
    
    analysis = {
        'network': network_name,
        'num_runs': len(benchmarks),
        'tests': {}
    }
    
    # Group by test type
    test_data = {}
    for benchmark in benchmarks:
        for test in benchmark.get('tests', []):
            test_name = test.get('test', 'unknown')
            if test_name not in test_data:
                test_data[test_name] = {
                    'gas_used': [],
                    'cost_eth': [],
                    'time_ms': []
                }
            
            if 'gasUsed' in test:
                test_data[test_name]['gas_used'].append(int(test['gasUsed']))
            if 'totalCost' in test:
                cost = float(test['totalCost'].split()[0])
                test_data[test_name]['cost_eth'].append(cost)
            if 'timeMs' in test:
                test_data[test_name]['time_ms'].append(float(test['timeMs']))
    
    # Calculate statistics
    for test_name, data in test_data.items():
        analysis['tests'][test_name] = {}
        
        if data['gas_used']:
            analysis['tests'][test_name]['avg_gas'] = int(statistics.mean(data['gas_used']))
            analysis['tests'][test_name]['min_gas'] = min(data['gas_used'])
            analysis['tests'][test_name]['max_gas'] = max(data['gas_used'])
        
        if data['cost_eth']:
            analysis['tests'][test_name]['avg_cost_eth'] = statistics.mean(data['cost_eth'])
            analysis['tests'][test_name]['min_cost_eth'] = min(data['cost_eth'])
            analysis['tests'][test_name]['max_cost_eth'] = max(data['cost_eth'])
        
        if data['time_ms']:
            analysis['tests'][test_name]['avg_time_ms'] = statistics.mean(data['time_ms'])
            analysis['tests'][test_name]['min_time_ms'] = min(data['time_ms'])
            analysis['tests'][test_name]['max_time_ms'] = max(data['time_ms'])
    
    return analysis

def print_comparison(analyses: Dict[str, dict]):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON ACROSS TESTNETS")
    print("="*80 + "\n")
    
    if not analyses:
        print("No benchmark data found.")
        return
    
    # Get all test types
    all_tests = set()
    for analysis in analyses.values():
        all_tests.update(analysis.get('tests', {}).keys())
    
    for test_name in sorted(all_tests):
        print(f"\n📊 {test_name}")
        print("-" * 80)
        print(f"{'Network':<20} {'Avg Gas':<15} {'Avg Cost (ETH)':<15} {'Avg Time (ms)':<15}")
        print("-" * 80)
        
        for network, analysis in sorted(analyses.items()):
            test_data = analysis.get('tests', {}).get(test_name, {})
            
            avg_gas = test_data.get('avg_gas', 'N/A')
            avg_cost = test_data.get('avg_cost_eth', 'N/A')
            avg_time = test_data.get('avg_time_ms', 'N/A')
            
            if avg_gas != 'N/A':
                avg_gas = f"{avg_gas:,}"
            if avg_cost != 'N/A':
                avg_cost = f"{avg_cost:.6f}"
            if avg_time != 'N/A':
                avg_time = f"{avg_time:.2f}"
            
            print(f"{network:<20} {str(avg_gas):<15} {str(avg_cost):<15} {str(avg_time):<15}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for network, analysis in sorted(analyses.items()):
        print(f"\n{network.upper()}:")
        print(f"  Number of runs: {analysis['num_runs']}")
        print(f"  Tests analyzed: {len(analysis['tests'])}")
        
        # Calculate total cost estimate
        total_cost = 0
        for test_data in analysis['tests'].values():
            if 'avg_cost_eth' in test_data:
                total_cost += test_data['avg_cost_eth']
        
        if total_cost > 0:
            print(f"  Estimated total cost: {total_cost:.6f} ETH")

def save_comparison(analyses: Dict[str, dict], output_file: str):
    """Save comparison results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"\n📁 Detailed comparison saved to: {output_file}")

def main():
    benchmark_dir = "output/benchmark"
    
    print("Loading benchmark results...")
    benchmark_data = load_benchmark_files(benchmark_dir)
    
    if not benchmark_data:
        print("No benchmark files found.")
        return
    
    print(f"Found results for networks: {', '.join(benchmark_data.keys())}")
    
    # Analyze each network
    analyses = {}
    for network, benchmarks in benchmark_data.items():
        analyses[network] = analyze_network(network, benchmarks)
    
    # Print comparison
    print_comparison(analyses)
    
    # Save detailed analysis
    output_file = "output/benchmark/comparison_analysis.json"
    save_comparison(analyses, output_file)
    
    print("\n✅ Analysis complete!\n")

if __name__ == "__main__":
    main()
