"""
Benchmarking tools for JPMapper.

This module provides tools for benchmarking the performance of JPMapper components.
This is a backward compatibility module that re-exports functions from benchmarks.core.
For new code, please use benchmarks.core directly.
"""

# Re-export functions from core module to maintain backward compatibility
from benchmarks.core import (
    BenchmarkResult as _BenchmarkResult,
    Benchmark,
    BenchmarkSuite,
    print_results,
    plot_results,
    compare_results
)

# For backward compatibility with existing code
from pathlib import Path
from typing import List, Dict, Any, Callable
import time
import statistics
import json

# Import rich components if available
try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function by running it multiple times and measuring performance.
    This is a backward compatibility function that uses Benchmark from core.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with benchmark results
    """
    # Extract iterations from kwargs
    iterations = kwargs.pop('iterations', 10)
    
    # Create and run a Benchmark
    benchmark = Benchmark(
        name=func.__name__,
        func=func,
        args=args,
        kwargs=kwargs,
        iterations=iterations,
        warmup=1
    )
    
    # Run the benchmark
    result = benchmark.run()
    
    # Convert BenchmarkResult to the old format dictionary
    return {
        'function': func.__name__,
        'min_time': result.min_time,
        'max_time': result.max_time,
        'mean_time': result.mean_time,
        'median_time': result.median_time,
        'stdev_time': result.stdev_time,
        'iterations': result.iterations,
        'result': None  # We don't have the result in the new API
    }


def run_benchmarks(benchmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run a list of benchmark configurations.
    This is a backward compatibility function.
    
    Args:
        benchmarks: List of benchmark configurations
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for benchmark in benchmarks:
        func = benchmark['function']
        args = benchmark.get('args', [])
        kwargs = benchmark.get('kwargs', {})
        
        if HAS_RICH:
            console.print(f"[yellow]Benchmarking {func.__name__}...[/yellow]")
        else:
            print(f"Benchmarking {func.__name__}...")
            
        result = benchmark_function(func, *args, **kwargs)
        results.append(result)
        
        if HAS_RICH:
            console.print(f"[green]Mean time: {result['mean_time']:.6f} seconds[/green]")
        else:
            print(f"Mean time: {result['mean_time']:.6f} seconds")
    
    return results


def print_benchmark_table(results: List[Dict[str, Any]]):
    """
    Print a table of benchmark results.
    This is a backward compatibility function.
    
    Args:
        results: List of benchmark results
    """
    if HAS_RICH:
        table = Table(title="Benchmark Results")
        
        table.add_column("Function", style="cyan")
        table.add_column("Mean (s)", style="green")
        table.add_column("Median (s)", style="green")
        table.add_column("Min (s)", style="blue")
        table.add_column("Max (s)", style="red")
        table.add_column("Stdev (s)", style="yellow")
        
        for result in results:
            table.add_row(
                result['function'],
                f"{result['mean_time']:.6f}",
                f"{result['median_time']:.6f}",
                f"{result['min_time']:.6f}",
                f"{result['max_time']:.6f}",
                f"{result['stdev_time']:.6f}"
            )
        
        console.print(table)
    else:
        # Use the core module's print_results function, converting the format first
        core_results = [
            _BenchmarkResult(
                name=result['function'],
                times=[result['mean_time']] * result['iterations'],
                metadata={}
            )
            for result in results
        ]
        print_results(core_results)


def save_benchmark_results(results: List[Dict[str, Any]], output_path: Path):
    """
    Save benchmark results to a JSON file.
    This is a backward compatibility function.
    
    Args:
        results: List of benchmark results
        output_path: Path to save results to
    """
    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = result.copy()
        clean_result.pop('result', None)
        clean_results.append(clean_result)
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    if HAS_RICH:
        console.print(f"[green]Benchmark results saved to {output_path}[/green]")
    else:
        print(f"Benchmark results saved to {output_path}")
