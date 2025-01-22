import os
from jinja2 import Environment, FileSystemLoader
from api_reference import API_REFERENCE

# Paths and template setup
TEMPLATE_DIR = "templates"
INDEX_TEMPLATE = "index.rst.template"  # Template for api/index.rst
INDEX_OUTPUT = "source/api/index.rst"

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

def render_template(template_name, context):
    """Render a Jinja2 template with the given context."""
    template = env.get_template(template_name)
    return template.render(context)


def write_file(output_path, content):
    """Write content to the specified file path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


def transform_api_reference(api_reference):
    """Transform API_REFERENCE to Scikit-learn-style format with type mapping."""
    transformed_reference = {}
    for module_name, module_details in api_reference.items():
        # Extract autosummary and type mapping
        members = module_details.get("members", [])
        autosummary = [member["name"] for member in members]
        types = {member["name"]: member["type"] for member in members}
        descriptions = {member["name"]: member["description"] for member in members}

        # Build transformed entry
        transformed_reference[module_name] = {
            "short_summary": module_details.get("description", "No description available."),
            "description": None,  # Add detailed descriptions if available
            "sections": [
                {
                    "title": None,  # Titles can be added if needed
                    "autosummary": autosummary,
                    "descriptions": descriptions,  # Include descriptions mapping
                }
            ],

            "type": types,  # Include type mapping
        }
    return transformed_reference


def generate_api_index():
    """Generate the api/index.rst file."""
    transformed_reference = transform_api_reference(API_REFERENCE)
    context = {
        "API_REFERENCE": transformed_reference,
        "DEPRECATED_API_REFERENCE": [],  # Add deprecated items if applicable
    }
    content = render_template(INDEX_TEMPLATE, context)
    write_file(INDEX_OUTPUT, content)
    print(f"Generated index .rst: {INDEX_OUTPUT}")


if __name__ == "__main__":
    generate_api_index()
