"""
Memory profiling utilities for JPMapper benchmarks.

This module provides tools for profiling memory usage of JPMapper functions,
which can be useful for identifying memory leaks and optimizing memory usage.
"""

import time
import functools
from typing import Callable, Dict, Any, List, Optional, Union
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import psutil, but don't fail if it's not installed
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory profiling functionality will be limited.")
    print("Install with: pip install psutil")


def get_process_memory() -> float:
    """
    Get the current memory usage of the process in MB.
    
    Returns:
        Memory usage in megabytes
    """
    if not HAS_PSUTIL:
        return 0.0
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_PSUTIL:
            print(f"Warning: Cannot profile memory for {func.__name__} - psutil not installed.")
            return func(*args, **kwargs)
        
        # Record memory before
        mem_before = get_process_memory()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Record memory after
        mem_after = get_process_memory()
        
        # Print memory usage
        print(f"Memory usage for {func.__name__}:")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After:  {mem_after:.2f} MB")
        print(f"  Diff:   {mem_after - mem_before:.2f} MB")
        
        return result
    
    return wrapper


class MemoryProfiler:
    """Class for profiling memory usage over time."""
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize a memory profiler.
        
        Args:
            interval: Sampling interval in seconds
        """
        if not HAS_PSUTIL:
            print("Warning: MemoryProfiler initialized but psutil is not installed.")
            print("Memory profiling will not be accurate.")
        
        self.interval = interval
        self.timestamps: List[float] = []
        self.memory_usage: List[float] = []
        self.running = False
    
    def _profile_thread(self):
        """Background thread for sampling memory usage."""
        import threading
        
        self.running = True
        start_time = time.time()
        
        while self.running:
            # Record timestamp and memory usage
            self.timestamps.append(time.time() - start_time)
            self.memory_usage.append(get_process_memory())
            
            # Sleep for the interval
            time.sleep(self.interval)
    
    def start(self):
        """Start profiling."""
        import threading
        
        self.timestamps = []
        self.memory_usage = []
        
        # Start the profiling thread
        self.thread = threading.Thread(target=self._profile_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop profiling."""
        self.running = False
        
        # Wait for the thread to finish
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=self.interval * 2)
    
    def profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function's memory usage.
        
        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with profiling results
        """
        # Start profiling
        self.start()
        
        # Run the function and time it
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Stop profiling
        self.stop()
        
        # Calculate statistics
        if self.memory_usage:
            min_memory = min(self.memory_usage)
            max_memory = max(self.memory_usage)
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            peak_usage = max_memory - self.memory_usage[0]
        else:
            min_memory = max_memory = avg_memory = peak_usage = 0
        
        return {
            'function': func.__name__,
            'execution_time': execution_time,
            'min_memory_mb': min_memory,
            'max_memory_mb': max_memory,
            'avg_memory_mb': avg_memory,
            'peak_usage_mb': peak_usage,
            'timestamps': self.timestamps,
            'memory_usage': self.memory_usage,
            'result': result
        }
    
    def plot(self, output_path: Optional[Union[str, Path]] = None, 
           title: Optional[str] = None, figsize=(10, 6)):
        """
        Plot memory usage over time.
        
        Args:
            output_path: Optional path to save the plot to
            title: Optional title for the plot
            figsize: Figure size (width, height) in inches
        """
        if not self.timestamps or not self.memory_usage:
            print("No profiling data to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot memory usage
        ax.plot(self.timestamps, self.memory_usage)
        
        # Add annotations for min and max
        min_idx = np.argmin(self.memory_usage)
        max_idx = np.argmax(self.memory_usage)
        
        ax.annotate(f"Min: {self.memory_usage[min_idx]:.2f} MB",
                   xy=(self.timestamps[min_idx], self.memory_usage[min_idx]),
                   xytext=(10, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->"))
        
        ax.annotate(f"Max: {self.memory_usage[max_idx]:.2f} MB",
                   xy=(self.timestamps[max_idx], self.memory_usage[max_idx]),
                   xytext=(10, -10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->"))
        
        # Customize plot
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title(title or "Memory Usage Over Time")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


def profile_memory_usage(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with profiling results
    """
    profiler = MemoryProfiler()
    return profiler.profile(func, *args, **kwargs)


# Example usage:
if __name__ == "__main__":
    # Example function to profile
    def create_large_array(size=1000000):
        """Create a large array to demonstrate memory usage."""
        print(f"Creating array of size {size}")
        arr = np.zeros(size)
        return arr
    
    # Profile with decorator
    @profile_memory
    def test_function():
        arr1 = create_large_array(1000000)
        arr2 = create_large_array(2000000)
        return arr1, arr2
    
    # Run the decorated function
    test_function()
    
    # Profile with MemoryProfiler
    profiler = MemoryProfiler()
    results = profiler.profile(create_large_array, 5000000)
    
    print(f"\nProfiling results:")
    print(f"  Execution time: {results['execution_time']:.4f} seconds")
    print(f"  Min memory: {results['min_memory_mb']:.2f} MB")
    print(f"  Max memory: {results['max_memory_mb']:.2f} MB")
    print(f"  Average memory: {results['avg_memory_mb']:.2f} MB")
    print(f"  Peak usage: {results['peak_usage_mb']:.2f} MB")
    
    # Plot memory usage
    profiler.plot(title=f"Memory Usage for {results['function']}")
