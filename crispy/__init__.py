try:
    from importlib.metadata import version
    __version__ = version("astro-crispy")
except ImportError:
    __version__ = "unknown"  # Fallback when metadata is unavailable

__version__ = version("astro-crispy")
