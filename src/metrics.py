"""
Metrics & Benchmarking Utility for Groceries Transaction Mining Pipeline.
Measures execution time, memory usage, and formats baseline results.
"""

import time
import psutil
import os
from typing import Callable, Tuple, Any, Dict


def measure_performance(func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Executes a callable function and measures execution time (seconds)
    and memory usage delta (MB).

    Parameters:
    -----------
    func : Callable
        Function to execute and benchmark.

    Returns:
    --------
    result : Any
        Return value of func.
    execution_time : float
        Elapsed time in seconds.
    memory_used_mb : float
        Peak RSS memory used in megabytes during execution.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 * 1024)

    execution_time = end_time - start_time
    memory_used_mb = max(0.0, mem_after - mem_before)

    return result, execution_time, memory_used_mb


def get_system_resource_info() -> Dict[str, Any]:
    """
    Returns current system resource statistics (CPU count, Total RAM, Current Memory RSS).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm_info = psutil.virtual_memory()

    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm_info.total / (1024**3), 2),
        "available_ram_gb": round(vm_info.available / (1024**3), 2),
        "current_rss_mb": round(mem_info.rss / (1024**2), 2),
    }
