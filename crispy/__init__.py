from importlib.metadata import version, PackageNotFoundError
import warnings

try:
    # First, try using the distribution package name
    __version__ = version("crispy-learn")  # Replace with your PyPI package name
except PackageNotFoundError:
    try:
        # Fallback: try using the module name (__name__)
        __version__ = version(__name__)
    except PackageNotFoundError:
        # Final fallback if both attempts fail
        __version__ = "0.0.0"
        warnings.warn(
            "Package metadata is unavailable. Using fallback version '0.0.0'."
        )