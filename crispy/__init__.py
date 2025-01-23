import os
import sys

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
    with open(pyproject_path, 'rb') as f:
        data = tomllib.load(f)

    if 'project' not in data:
        raise KeyError("The [project] section is missing in pyproject.toml.")

    project_data = data['project']
    name = project_data.get('name', 'Unknown')
    version = project_data.get('version', '0.0.0')
    return name, version

try:
    __name__, __version__ = get_metadata()
except Exception as e:
    __name__ = "crispy-learn"
    __version__ = "0.0.0"
    print(f"Warning: Failed to load metadata from pyproject.toml ({e})")