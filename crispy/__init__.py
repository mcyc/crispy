import os
import sys
import warnings

def get_metadata():
    """Load project metadata from pyproject.toml."""
    if sys.version_info < (3, 11):
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError("The 'tomli' module is required for Python < 3.11. Install it using 'pip install tomli'.")
    else:
        import tomllib

    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find pyproject.toml at {pyproject_path}")

    if 'project' not in data:
        raise KeyError("The [project] section is missing in pyproject.toml.")

    project_data = data['project']
    return project_data.get('version', '0.0.0')

try:
    __version__ = get_metadata()
except Exception as e:
    __version__ = "0.0.0"
    warnings.warn(f"Failed to load metadata from pyproject.toml: {e}")