from pathlib import Path
import sys
import warnings


def load_tomllib():
    """Load the TOML library based on the Python version."""
    if sys.version_info < (3, 11):
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError("The 'tomli' module is required for Python < 3.11. Install it using 'pip install tomli'.")
    else:
        import tomllib
    return tomllib


def get_metadata():
    """
    Load project metadata from pyproject.toml.

    Returns
    -------
    dict
        A dictionary containing project metadata.
    """
    tomllib = load_tomllib()
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'

    if not pyproject_path.exists():
        raise FileNotFoundError(f"Could not find pyproject.toml at {pyproject_path}")

    try:
        with pyproject_path.open('rb') as f:
            data = tomllib.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse pyproject.toml: {e}")

    if 'project' not in data:
        raise KeyError("The [project] section is missing in pyproject.toml.")

    return data['project']


# Load version metadata
try:
    __version__ = get_metadata().get('version', '0.0.0')
except Exception as e:
    __version__ = "0.0.0"
    warnings.warn(f"Failed to load metadata from pyproject.toml: {e}")