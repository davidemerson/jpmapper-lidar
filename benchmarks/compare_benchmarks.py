"""
Benchmark comparison tool for JPMapper.

This script compares benchmark results from different runs to track performance over time.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.core import BenchmarkResult, compare_results


def load_benchmark_results(result_files: List[Path]) -> List[List[BenchmarkResult]]:
    """
    Load benchmark results from multiple files.
    
    Args:
        result_files: List of paths to result files
        
    Returns:
        List of lists of BenchmarkResult objects
    """
    all_results = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_data in data.get('results', []):
            result = BenchmarkResult.from_dict(result_data)
            results.append(result)
        
        all_results.append(results)
    
    return all_results


def generate_comparison_table(all_results: List[List[BenchmarkResult]], labels: List[str]) -> pd.DataFrame:
    """
    Generate a comparison table of benchmark results.
    
    Args:
        all_results: List of lists of BenchmarkResult objects
        labels: Labels for each set of results
        
    Returns:
        DataFrame containing the comparison
    """
    # Get unique benchmark names
    all_names = set()
    for results in all_results:
        all_names.update(result.name for result in results)
    all_names = sorted(all_names)
    
    # Create a DataFrame for comparison
    data = {}
    for results, label in zip(all_results, labels):
        # Create a dictionary mapping benchmark names to mean times
        result_dict = {result.name: result.mean_time for result in results}
        # Add to data dictionary with missing benchmarks as NaN
        data[label] = [result_dict.get(name, float('nan')) for name in all_names]
    
    # Create DataFrame
    df = pd.DataFrame(data, index=all_names)
    return df


def plot_performance_trends(all_results: List[List[BenchmarkResult]], labels: List[str], 
                          output_path: Optional[Path] = None, figsize=(12, 8)) -> None:
    """
    Plot performance trends across multiple benchmark runs.
    
    Args:
        all_results: List of lists of BenchmarkResult objects
        labels: Labels for each set of results
        output_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    # Create a DataFrame from the results
    df = generate_comparison_table(all_results, labels)
    
    # Transpose the DataFrame to have benchmark names as columns
    df_t = df.transpose()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lines for each benchmark
    for column in df_t.columns:
        df_t[column].plot(ax=ax, marker='o', label=column)
    
    # Customize plot
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Performance Trends')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Benchmark')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main():
    """Main function to compare benchmark results."""
    parser = argparse.ArgumentParser(description="Compare JPMapper benchmark results")
    parser.add_argument("result_files", nargs='+', type=str,
                      help="Paths to benchmark result JSON files")
    parser.add_argument("--labels", nargs='+', type=str,
                      help="Labels for each result file (default: file names)")
    parser.add_argument("--output", type=str,
                      help="Path to save comparison plot")
    args = parser.parse_args()
    
    result_paths = [Path(file_path) for file_path in args.result_files]
    
    # Use file names as default labels if not provided
    labels = args.labels or [path.stem for path in result_paths]
    
    # Make sure the number of labels matches the number of files
    if len(labels) != len(result_paths):
        print("Warning: Number of labels doesn't match number of files. Using file names as labels.")
        labels = [path.stem for path in result_paths]
    
    # Load results
    all_results = load_benchmark_results(result_paths)
    
    # Generate comparison table
    df = generate_comparison_table(all_results, labels)
    print("\nBenchmark Comparison:")
    print(df)
    
    # Calculate improvements
    if len(all_results) >= 2:
        # Compare first and last result sets
        print("\nPerformance Changes:")
        for name in df.index:
            first_value = df.iloc[df.index.get_loc(name), 0]
            last_value = df.iloc[df.index.get_loc(name), -1]
            
            if pd.isna(first_value) or pd.isna(last_value):
                print(f"{name}: Cannot calculate change (missing data)")
            else:
                change_pct = (first_value - last_value) / first_value * 100
                change_str = f"+{change_pct:.2f}%" if change_pct < 0 else f"{change_pct:.2f}%"
                print(f"{name}: {change_str} ({'slower' if change_pct < 0 else 'faster'})")
    
    # Plot results
    output_path = Path(args.output) if args.output else None
    plot_performance_trends(all_results, labels, output_path)
    
    if output_path:
        print(f"\nComparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
