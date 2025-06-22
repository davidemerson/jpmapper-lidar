"""
Core benchmarking utilities for JPMapper.

This module provides tools for benchmarking the performance of JPMapper components,
measuring execution times, and reporting results.
"""

import time
import datetime
import statistics
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Try to import tabulate, but provide a fallback if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not installed. Benchmark tables will use a simpler format.")
    print("Install with: pip install tabulate")


class BenchmarkResult:
    """Class to store and analyze benchmark results."""
    
    def __init__(self, name: str, times: List[float], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a benchmark result.
        
        Args:
            name: Name of the benchmark
            times: List of execution times in seconds
            metadata: Additional metadata about the benchmark
        """
        self.name = name
        self.times = times
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now().isoformat()
    
    @property
    def min_time(self) -> float:
        """Minimum execution time."""
        return min(self.times)
    
    @property
    def max_time(self) -> float:
        """Maximum execution time."""
        return max(self.times)
    
    @property
    def mean_time(self) -> float:
        """Mean execution time."""
        return statistics.mean(self.times)
    
    @property
    def median_time(self) -> float:
        """Median execution time."""
        return statistics.median(self.times)
    
    @property
    def stdev_time(self) -> float:
        """Standard deviation of execution times."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0
    
    @property
    def iterations(self) -> int:
        """Number of iterations run."""
        return len(self.times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'times': self.times,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'mean_time': self.mean_time,
            'median_time': self.median_time,
            'stdev_time': self.stdev_time,
            'iterations': self.iterations,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a BenchmarkResult from a dictionary."""
        result = cls(data['name'], data['times'], data.get('metadata', {}))
        result.timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
        return result


class Benchmark:
    """Class to run and manage benchmarks."""
    
    def __init__(self, name: str, func: Callable, args: Optional[List] = None, 
                 kwargs: Optional[Dict[str, Any]] = None, iterations: int = 10,
                 warmup: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a benchmark.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            iterations: Number of iterations to run
            warmup: Number of warmup iterations to run
            metadata: Additional metadata about the benchmark
        """
        self.name = name
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.iterations = iterations
        self.warmup = warmup
        self.metadata = metadata or {'function_name': func.__name__}
    
    def run(self) -> BenchmarkResult:
        """
        Run the benchmark and return the results.
        
        Returns:
            BenchmarkResult containing timing information
        """
        # Run warmup iterations
        for _ in range(self.warmup):
            self.func(*self.args, **self.kwargs)
        
        # Run benchmark iterations
        times = []
        for _ in range(self.iterations):
            start_time = time.time()
            self.func(*self.args, **self.kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return BenchmarkResult(self.name, times, self.metadata)


class BenchmarkSuite:
    """Class to manage multiple benchmarks."""
    
    def __init__(self, name: str, benchmarks: Optional[List[Benchmark]] = None):
        """
        Initialize a benchmark suite.
        
        Args:
            name: Name of the benchmark suite
            benchmarks: List of benchmarks to run
        """
        self.name = name
        self.benchmarks = benchmarks or []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """
        Add a benchmark to the suite.
        
        Args:
            benchmark: Benchmark to add
        """
        self.benchmarks.append(benchmark)
    
    def run(self, progress_callback: Optional[Callable[[str, int, int], None]] = None) -> List[BenchmarkResult]:
        """
        Run all benchmarks in the suite.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of BenchmarkResult objects
        """
        self.results = []
        total = len(self.benchmarks)
        
        for i, benchmark in enumerate(self.benchmarks):
            if progress_callback:
                progress_callback(benchmark.name, i, total)
            
            result = benchmark.run()
            self.results.append(result)
        
        return self.results
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_path: Path to save results to
        """
        output_path = Path(output_path)
        results_data = {
            'suite_name': self.name,
            'timestamp': datetime.datetime.now().isoformat(),
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    @classmethod
    def load_results(cls, input_path: Union[str, Path]) -> Tuple[str, List[BenchmarkResult]]:
        """
        Load benchmark results from a JSON file.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Tuple of (suite_name, results)
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        suite_name = data.get('suite_name', 'Unnamed Suite')
        results = [BenchmarkResult.from_dict(result) for result in data['results']]
        
        return suite_name, results


def print_results(results: List[BenchmarkResult]) -> None:
    """
    Print benchmark results in a table format.
    
    Args:
        results: List of BenchmarkResult objects
    """
    table_data = []
    headers = ['Name', 'Mean (s)', 'Median (s)', 'Min (s)', 'Max (s)', 'Stdev (s)', 'Iterations']
    
    for result in results:
        table_data.append([
            result.name,
            f"{result.mean_time:.6f}",
            f"{result.median_time:.6f}",
            f"{result.min_time:.6f}",
            f"{result.max_time:.6f}",
            f"{result.stdev_time:.6f}",
            result.iterations
        ])
    
    if HAS_TABULATE:
        # Use tabulate for nice formatting
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        # Fallback to simple formatting
        print("\n" + " | ".join(headers))
        print("-" * 80)
        for row in table_data:
            print(" | ".join(str(cell) for cell in row))


def plot_results(results: List[BenchmarkResult], output_path: Optional[Union[str, Path]] = None,
                title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot benchmark results as a bar chart.
    
    Args:
        results: List of BenchmarkResult objects
        output_path: Optional path to save the plot to
        title: Optional title for the plot
        figsize: Figure size (width, height) in inches
    """
    # Extract data
    names = [result.name for result in results]
    means = [result.mean_time for result in results]
    stdevs = [result.stdev_time for result in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(names, means, yerr=stdevs, capsize=10)
    
    # Customize plot
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(title or 'Benchmark Results')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.6f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def compare_results(results_list: List[List[BenchmarkResult]], labels: List[str],
                   output_path: Optional[Union[str, Path]] = None,
                   title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Compare multiple sets of benchmark results.
    
    Args:
        results_list: List of lists of BenchmarkResult objects
        labels: Labels for each set of results
        output_path: Optional path to save the plot to
        title: Optional title for the plot
        figsize: Figure size (width, height) in inches
    """
    # Get unique benchmark names across all result sets
    all_names = set()
    for results in results_list:
        all_names.update(result.name for result in results)
    all_names = sorted(all_names)
    
    # Create a DataFrame for comparison
    data = {}
    for i, (results, label) in enumerate(zip(results_list, labels)):
        # Create a dictionary mapping benchmark names to mean times
        result_dict = {result.name: result.mean_time for result in results}
        # Add to data dictionary with missing benchmarks as NaN
        data[label] = [result_dict.get(name, float('nan')) for name in all_names]
    
    # Create DataFrame
    df = pd.DataFrame(data, index=all_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grouped bar chart
    df.plot(kind='bar', ax=ax)
    
    # Customize plot
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(title or 'Benchmark Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
