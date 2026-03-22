import os
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, Any, get_type_hints

# Import the root configuration node
from hydra_types import HydraSettings


def get_readable_type(type_hint) -> str:
    """Converts Python type hints into readable strings dynamically."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Optional[X] (represented as Union[X, NoneType])
    if origin is Union and type(None) in args:
        inner_type = args[0]
        return f"{get_readable_type(inner_type)}/Null"

    # Get the base class (e.g., 'dict' for Dict[str, Any], or 'int' for int)
    base_type = origin if origin is not None else type_hint

    # Dynamically extract the type's name and capitalize it (int -> Int, str -> Str)
    if hasattr(base_type, '__name__'):
        return base_type.__name__.capitalize()

    # Fallback for special typing module objects (like Any)
    if type_hint is Any:
        return "Any"

    return str(type_hint).replace('typing.', '')


def build_yaml_body(dataclass_type, indent_level=1) -> str:
    """Recursively traverses the dataclass DAG to build formatted text."""
    lines = []
    indent = "  " * indent_level

    # Resolve forward references (strings) into actual class objects
    resolved_hints = get_type_hints(dataclass_type)

    # Filter standard fields to calculate alignment padding
    standard_fields = [f for f in fields(dataclass_type) if not is_dataclass(resolved_hints[f.name])]
    max_len = max([len(f.name) for f in standard_fields]) if standard_fields else 0

    for f in fields(dataclass_type):
        name = f.name
        actual_type = resolved_hints[f.name]  # Use the resolved type, not the string

        # Grab the metadata.
        help_text = f.metadata.get("help", "No description provided.")

        if is_dataclass(actual_type):
            # If this is a nested config group, add the header and recurse deeper
            lines.append(f"\n{indent}{name}:")
            lines.append(build_yaml_body(actual_type, indent_level + 1))
        else:
            # If this a leaf node, format it with alignment
            type_str = get_readable_type(actual_type)
            padded_name = f"{name:<{max_len}}"
            lines.append(f"{indent}{padded_name} : ({type_str}) {help_text}")

    return "\n".join(lines)


def generate_yaml_file(output_path: str):
    """Wraps the body in the Hydra template and saves it to disk."""

    # Generate the dynamic body starting at indent level 1
    dynamic_body = build_yaml_body(HydraSettings, indent_level=1)

    # Construct the final YAML string.
    yaml_content = f"""template: |
  ======================================================================
  ==                   NumberModel Training Pipeline                  ==
  ======================================================================

  == Configuration Groups ==
  Compose your configuration from these groups (group=option)

  $APP_CONFIG_GROUPS

  == Parameters & Descriptions ==
{dynamic_body}

  ======================================================================
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"Successfully generated config documentation at: {output_path}")


if __name__ == "__main__":
    target_path = "config/hydra/help/all_parameters_documentation.yaml"
    generate_yaml_file(target_path)