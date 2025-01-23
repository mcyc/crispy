import os
import pkgutil
import inspect
import importlib

ROOT_PACKAGE = "crispy"
OUTPUT_FILE = "scripts/api_reference.py"


def get_sphinx_description(obj):
    """Extract the first sentence of a docstring, mimicking Sphinx's autosummary."""
    docstring = inspect.getdoc(obj)
    if not docstring:
        return "Undocumented"  # Fallback if no docstring is available

    # Split docstring into lines, remove leading/trailing whitespace
    lines = [line.strip() for line in docstring.splitlines() if line.strip()]
    if not lines:
        return "Undocumented"  # No meaningful content in docstring

    # Use the first non-empty line and split it by periods to find the first sentence
    first_sentence = lines[0].split(".")[0].strip()
    if not first_sentence.endswith("."):
        first_sentence += "."  # Ensure the sentence ends with a period
    return first_sentence


def discover_package(package_name):
    """Discover all modules, classes, functions, and submodules in a package."""
    package = importlib.import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    api_reference = {}

    # Walk through the package directory to find modules and submodules
    for _, module_name, is_pkg in pkgutil.walk_packages([package_dir], prefix=f"{package_name}."):
        # Skip modules starting with "._"
        if '._' in module_name:
            print(f"Skipping hidden module: {module_name}")
            continue

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Error: Could not import {module_name} - {e}")
            continue

        # Discover classes and functions defined in the module
        classes = [
            (cls, get_sphinx_description(obj))
            for cls, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module_name and not cls.startswith("_")
        ]
        functions = [
            (func, get_sphinx_description(obj))
            for func, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module_name and not func.startswith("_")
        ]

        # Add submodules explicitly for packages
        if is_pkg:
            submodules = [
                submodule_name for _, submodule_name, sub_is_pkg in pkgutil.iter_modules(
                    [os.path.dirname(module.__file__)], prefix=f"{module_name}."
                )
                if '._' not in submodule_name  # Skip hidden submodules
            ]
            members = [{"name": submodule, "type": "module", "description": "No description available."} for submodule in submodules]
        else:
            members = []

        # Add classes and functions to members
        members += [{"name": cls, "type": "class", "description": desc} for cls, desc in classes]
        members += [{"name": func, "type": "function", "description": desc} for func, desc in functions]

        # Store module details in the API reference
        api_reference[module_name] = {
            "module": module_name,
            "description": get_sphinx_description(module),  # Use first sentence of module docstring
            "members": members,
        }

    return api_reference


def write_api_reference(api_reference, output_file):
    """Write the API_REFERENCE dictionary to a Python file with confirmation."""
    if os.path.exists(output_file):
        # Prompt user before overwriting
        confirm = input(f"{output_file} exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborting...")
            return

    with open(output_file, "w") as f:
        f.write("# This file is auto-generated. Edit descriptions and structure as needed.\n\n")
        f.write("API_REFERENCE = {\n")
        for module_name, details in api_reference.items():
            f.write(f"    '{module_name}': {{\n")
            f.write(f"        'module': '{details['module']}',\n")
            f.write(f"        'description': '''{details['description']}''',\n")
            f.write(f"        'members': [\n")
            for member in details["members"]:
                f.write(f"            {{'name': '{member['name']}', 'type': '{member['type']}', 'description': '''{member['description']}'''}}")
                if member != details["members"][-1]:  # Avoid trailing commas
                    f.write(",\n")
            f.write("\n        ]\n")
            f.write("    },\n")
        f.write("}\n")

    print(f"Generated: {output_file}")


if __name__ == "__main__":
    api_reference = discover_package(ROOT_PACKAGE)
    write_api_reference(api_reference, OUTPUT_FILE)
