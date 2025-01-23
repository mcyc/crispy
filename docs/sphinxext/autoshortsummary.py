from sphinx.util.docstrings import prepare_docstring
from docutils.parsers.rst import Directive
from docutils.nodes import paragraph, Text
from importlib import import_module
import logging

class AutoShortSummary(Directive):
    has_content = True

    def run(self):
        if not self.content or len(self.content) == 0:
            raise ValueError("autoshortsummary directive requires the full object name as content")

        # Extract the full object name
        obj_name = self.content[0]
        logging.debug(f"autoshortsummary: Processing object name: {obj_name}")

        # Split the module path and object name
        module_name, _, obj_attr = obj_name.rpartition(".")
        if not module_name or not obj_attr:
            raise RuntimeError(f"Invalid object name: {obj_name}. Must be in the form 'module.object'.")

        # Import the module and resolve the object
        try:
            logging.debug(f"Attempting to import module: {module_name}, object: {obj_attr}")
            module = import_module(module_name)  # Import the module
            obj = getattr(module, obj_attr)     # Resolve the object in the module
            logging.debug(f"Successfully imported: {obj_name}")
        except Exception as e:
            logging.error(f"Import failed for {obj_name}: {e}")
            raise RuntimeError(f"Could not import {obj_name}: {e}")

        # Extract the docstring and generate a short summary
        docstring = prepare_docstring(obj.__doc__ or "No description available.")
        short_summary = docstring[0] if docstring else "No description available."

        # Create a paragraph node containing the summary
        return [paragraph('', Text(short_summary))]


def setup(app):
    app.add_directive("autoshortsummary", AutoShortSummary)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True
    }
