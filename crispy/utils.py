import functools
import tracemalloc

def peak_memory_usage(func):
    """
    Decorator to measure and print the peak memory usage of a function.

    This decorator uses `tracemalloc` to track memory allocation during the
    execution of the decorated function and reports the peak memory usage in
    megabytes (MB).

    Parameters
    ----------
    func : callable
        The function to be wrapped and monitored for peak memory usage.

    Returns
    -------
    callable
        A wrapped function that prints the peak memory usage upon execution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[{func.__name__}] Peak memory usage: {peak / (1024 * 1024):.2f} MB")
        return result

    return wrapper