# Configuration file for the Sphinx documentation builder.
import os
import sys
import tomli  # Use tomli for Python < 3.11

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../crispy'))
sys.path.insert(0, os.path.abspath('../sphinxext'))

# -- Project metadata from pyproject.toml -------------------------------------
def load_metadata():
    """Load project metadata from pyproject.toml."""
    pyproject_path = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        data = tomli.load(f)

    if 'project' not in data:
        raise KeyError("The [project] section is missing in pyproject.toml.")

    metadata = data['project']
    # Extract repository URL
    urls = metadata.get('urls', {})
    metadata['repository'] = urls.get('Source', None)  # Use 'Source' URL for the repository
    return metadata

# Load metadata
metadata = load_metadata()

# Populate Sphinx project information
project = metadata.get('name', 'Unknown Project')
release = metadata.get('version', '0.0.1')
version = ".".join(release.split(".")[:2])
author = "Mike Chen" #metadata.get('authors', [{'name': 'Unknown Author'}])[0]['name']
copyright = f"2025, {author}"
github_url = metadata.get('repository', None)

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
    'sphinx_issues',
    'sphinx.ext.doctest',
    'numpydoc',
    'sphinxext.opengraph',
    'nbsphinx',
    'autoshortsummary',
]

autosummary_generate = True
autosummary_imported_members = False
numpydoc_show_class_members = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'inherited-members': False,
    'special-members': False,
    'exclude-members': '__weakref__',
}

# -- HTML output options ------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {"text": "CRISPy"},
    "github_url": github_url,  # Dynamically populated from pyproject.toml
    "show_toc_level": 4,
    "navbar_align": "left",
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_nav_level": 4,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
}

templates_path = ['_templates']
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = '_static/favicon.ico'
html_logo = "_static/logo.png"

html_sidebars = {
    "quick_start": [],
    "installation": [],
    "guides": [],
    "tutorials/index": [],
}
