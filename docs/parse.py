# type: ignore

import ast
import json
import os
import re
import argparse


def parse_markdown(text):
    """
    Parse inline Markdown in a text and return a list with mixed types,
    including handling line breaks as "\n" and inline code.

    Args:
        text (str): The text to parse.

    Returns:
        list: A list containing plain text, "\n" for line breaks, and dictionaries for Markdown elements.
    """
    elements = []
    pattern = re.compile(r"(!?\[.*?\]\(.*?\))|(`[^`]+`)")
    parts = pattern.split(text)

    for part in parts:
        if part:
            if part.startswith("!"):
                alt_text = re.search(r"!\[(.*?)\]", part).group(1)
                url = re.search(r"\((.*?)\)", part).group(1)
                elements.append({"image": [url, alt_text]})
            elif part.startswith("["):
                link_text = re.search(r"\[(.*?)\]", part).group(1)
                url = re.search(r"\((.*?)\)", part).group(1)
                if url.startswith("`") and url.endswith("`"):
                    url = url[1:-1]
                    url_parts = url.split("/")
                    elements.append({"ref": [link_text, url_parts]})
                else:
                    elements.append({"link": [url, link_text]})
            elif part.startswith("`") and part.endswith("`"):
                code_text = part[1:-1]
                elements.append({"code": code_text})
            else:
                lines = part.split("\n")
                for i, line in enumerate(lines):
                    if line.strip():
                        # Add a space before this line if the previous element was text
                        # and this isn't the first line in this part
                        if i > 0 and elements and isinstance(elements[-1], str) and elements[-1] != "\n":
                            elements[-1] = elements[-1] + " " + line.strip()
                        else:
                            elements.append(line.strip())
                    else:
                        elements.append("\n")

    return elements


def parse_examples(text):
    """
    Parse the Examples section of a docstring into a single formatted string.

    Args:
        text (str): The examples text to parse.

    Returns:
        str: The entire examples block as a single string, with consistent indentation removed.
    """
    if not text.strip():
        return text

    lines = text.split("\n")

    # Find the minimum indentation (excluding empty lines)
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return text

    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

    # Remove the common indentation from all lines
    cleaned_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            cleaned_lines.append(line[min_indent:] if len(line) >= min_indent else line)
        else:  # Empty line
            cleaned_lines.append("")

    return "\n".join(cleaned_lines)


def extract_types(type_string):
    """
    Extract individual types from a Union or other type hints, stripping out the Union keyword.
    Supports both old Union[A, B] syntax and new A | B syntax.

    Args:
        type_string (str): The type string to parse.

    Returns:
        list: A list of individual types.
    """
    type_string = type_string.strip()

    if type_string.startswith("Union["):
        # strip the 'Union[' and trailing ']'
        content = type_string[len("Union[") : -1].strip()
        return [t.strip() for t in content.split(",")]
    elif "|" in type_string:
        # Handle new Python 3.10+ union syntax (A | B | C)
        return [t.strip() for t in type_string.split("|")]
    else:
        # For everything else, split on commas, then strip.
        return [t.strip() for t in type_string.split(",")]


def parse_args(text, context=""):
    """
    Parse the Args section of a docstring into structured arguments.

    Args:
        text (str): The args text to parse.
        context (str): Source location for error messages.

    Returns:
        list: A list of dictionaries, each representing an argument.
    """
    args = []
    arg_lines = text.split("\n")
    arg_pattern = re.compile(r"^\s*(\*?\*?\w+)\s*\((.*?)\):\s*(.*)")

    i = 0
    while i < len(arg_lines):
        line = arg_lines[i]
        match = arg_pattern.match(line)
        if match:
            name, arg_type, info = match.groups()
            types = extract_types(arg_type)

            # Collect continuation lines for this parameter
            full_info = info
            i += 1
            while i < len(arg_lines):
                next_line = arg_lines[i]
                # Check if this is a continuation line (indented, starts with -, or empty)
                if (
                    next_line.startswith("        ")  # indented continuation
                    or next_line.strip().startswith("-")  # bullet point
                    or (not arg_pattern.match(next_line) and not next_line.startswith("Args:"))
                ):
                    # Add the continuation line, preserving some formatting
                    if next_line.strip():
                        full_info += " " + next_line.strip()
                    i += 1
                else:
                    # This line starts a new parameter or section, back up
                    break

            args.append({"name": name, "type": types, "info": parse_markdown(full_info)})
        else:
            if not line.startswith("Args:") and line.strip():
                print(f"[{context}] Args: no match for line: {line}")
            i += 1

    return args


def parse_raises(text, context=""):
    """
    Parse the Raises section of a docstring into structured raises.

    Args:
        text (str): The raises text to parse.
        context (str): Source location for error messages.

    Returns:
        list: A list of dictionaries, each representing a raised exception.
    """
    raises = []
    raise_lines = text.split("\n")
    raise_pattern = re.compile(r"^\s*(\w+):\s*(.*)")

    for line in raise_lines:
        match = raise_pattern.match(line)
        if match:
            exception, info = match.groups()
            raises.append({"exception": exception, "info": parse_markdown(info)})
        else:
            if line.strip():
                print(f"[{context}] Raises: no match for line: {line}")

    return raises


def parse_returns(text, context=""):
    """
    Parse the Returns section of a docstring into structured returns.

    Args:
        text (str): The returns text to parse.
        context (str): Source location for error messages.

    Returns:
        dict: A dictionary representing the return type and info.
    """
    return_lines = text.strip().split("\n")
    if not return_lines:
        return None

    # Process the return lines directly (header already excluded by split_docstring)
    if return_lines:
        first_line = return_lines[0].strip()
        return_pattern = re.compile(r"^([\w\.\[\], ]+):\s*(.*)$")
        match = return_pattern.match(first_line)

        if match:
            return_type, info = match.groups()
            return_types = extract_types(return_type)
            return_info = [info.strip()]
            return_info.extend(line.strip() for line in return_lines[1:])
            parsed_return = {"type": return_types, "info": parse_markdown(" ".join(return_info))}
            return parsed_return

    print(f"[{context}] Returns: no match for: {return_lines[0].strip() if return_lines else '(empty)'}")
    return None


def parse_data_section(text):
    """
    Parse the Data section of a docstring into a dictionary.

    Args:
        text (str): The data text to parse.

    Returns:
        dict: A dictionary representing the data key-value pairs.
    """
    data = {}
    data_lines = text.strip().split("\n")  # Header already excluded by split_docstring
    data_pattern = re.compile(r"^\s*(\w+):\s*(.*)$")

    for line in data_lines:
        match = data_pattern.match(line)
        if match:
            key, value = match.groups()
            # Attempt to convert value to int or float if applicable
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            data[key] = value

    return data


def split_docstring(docstring, context=""):
    """
    Split a docstring into its main parts: Summary, Args, Returns, Raises, Examples, and Data.

    Args:
        docstring (str): The docstring to split.
        context (str): Source location (e.g. "file.py:funcName") for error messages.

    Returns:
        dict: A dictionary containing strings or lists of each part of the docstring.
    """
    if not docstring:
        return {}

    headers = ["Args:", "Returns:", "Raises:", "Examples:", "Data:"]
    parts = {"Summary": ""}

    current_header = "Summary"
    buffer = []

    for line in docstring.split("\n"):
        if any(line.strip().startswith(header) for header in headers):
            if buffer:
                parts[current_header] = "\n".join(buffer) if current_header != "Examples" else "\n".join(buffer)
                buffer = []
            current_header = line.strip().rstrip(":")
            # Don't add the header line to the buffer - it should be excluded from content
        else:
            buffer.append(line)

    if buffer:
        parts[current_header] = "\n".join(buffer) if current_header != "Examples" else "\n".join(buffer)

    for key, value in parts.items():
        if key == "Examples":
            parts[key] = parse_examples(value)
        elif key == "Args":
            parts[key] = parse_args(value.strip(), context)
        elif key == "Returns":
            parts[key] = parse_returns(value.strip(), context)
        elif key == "Raises":
            parts[key] = parse_raises(value.strip(), context)
        elif key == "Data":
            parts[key] = parse_data_section(value.strip())
        else:
            parts[key] = parse_markdown(value.strip())

    return parts


def get_decorator_name(decorator):
    """
    Safely extract the name of a decorator from an AST node.

    Args:
        decorator: An AST decorator node (ast.Name, ast.Call, or ast.Attribute).

    Returns:
        str or None: The decorator name if it can be extracted, None otherwise.
    """
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Name):
            return decorator.func.id
        elif isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr
    elif isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def get_category_from_path(file_path):
    """
    Extract the category (module name) from a file path.

    Args:
        file_path (str): The file path to extract category from.

    Returns:
        str: The module name without extension.
    """
    import os

    return os.path.splitext(os.path.basename(file_path))[0]


def extract_docstrings(file_path, input_base):
    """
    Extract docstrings from a Python file.

    Args:
        file_path (str): Path to the Python file.
        input_base (str): Base input directory to strip from file paths.

    Returns:
        dict: A dictionary containing module, classes, and functions docstrings with their source information.
    """
    with open(file_path, "r") as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
        except SyntaxError as e:
            print(f"Syntax error in {file_path[len(input_base) :]}: {e}")
            return None

    docstrings = {
        "module": {
            "docstring": split_docstring(ast.get_docstring(tree), file_path[len(input_base) :]),
            "source": [file_path[len(input_base) :], 1],
            "category": get_category_from_path(file_path),
        },
        "classes": {},
        "functions": {},
    }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Skip classes that start with underscore
            if node.name.startswith("_"):
                continue

            class_docstring = ast.get_docstring(node)
            docstrings["classes"][node.name] = {
                "docstring": split_docstring(class_docstring, f"{file_path[len(input_base) :]}:{node.name}"),
                "source": [file_path[len(input_base) :], node.lineno],
                "category": get_category_from_path(file_path),
                "methods": {},
            }
            if not class_docstring:
                print(f"{file_path[len(input_base) :]} > {node.name} : missing documentation")

            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip methods that start with underscore or have @overload
                    if class_node.name.startswith("_"):
                        continue
                    if any(get_decorator_name(d) == "overload" for d in class_node.decorator_list):
                        continue

                    method_docstring = ast.get_docstring(class_node)
                    method_details = {
                        "docstring": split_docstring(
                            method_docstring, f"{file_path[len(input_base) :]}:{node.name}.{class_node.name}"
                        ),
                        "source": [file_path[len(input_base) :], class_node.lineno],
                        "category": get_category_from_path(file_path),
                    }
                    # Check for staticmethod decorator
                    if any(get_decorator_name(decorator) == "staticmethod" for decorator in class_node.decorator_list):
                        method_details["static"] = True
                    # Check for curry decorator
                    if any(get_decorator_name(decorator) == "curry" for decorator in class_node.decorator_list):
                        method_details["curry"] = True
                    docstrings["classes"][node.name]["methods"][class_node.name] = method_details
                    if not method_docstring:
                        print(
                            f"{file_path[len(input_base) :]} > {node.name} > {class_node.name} : missing documentation"
                        )

        elif isinstance(node, ast.FunctionDef):
            # Skip functions that start with underscore or have @overload
            if node.name.startswith("_"):
                continue
            if any(get_decorator_name(d) == "overload" for d in node.decorator_list):
                continue

            func_docstring = ast.get_docstring(node)
            func_details = {
                "docstring": split_docstring(func_docstring, f"{file_path[len(input_base) :]}:{node.name}"),
                "source": [file_path[len(input_base) :], node.lineno],
                "category": get_category_from_path(file_path),
            }
            # Check for staticmethod decorator
            if any(get_decorator_name(decorator) == "staticmethod" for decorator in node.decorator_list):
                func_details["static"] = True
            # Check for curry decorator
            if any(get_decorator_name(decorator) == "curry" for decorator in node.decorator_list):
                func_details["curry"] = True
            docstrings["functions"][node.name] = func_details
            if not func_docstring:
                print(f"{file_path[len(input_base) :]} > {node.name} : missing documentation")

    return docstrings


def parse_type_definitions(file_path):
    """
    Parse a typing.py file to extract type definitions and their summaries.

    Args:
        file_path (str): Path to the typing.py file.

    Returns:
        list: A list of dictionaries, each representing a type definition.
    """
    type_definitions = []
    with open(file_path, "r") as file:
        content = file.read()

    # Pattern to match type definitions and their summaries
    # Supports both old format (Name = value) and new format (Name: TypeAlias = value)
    type_pattern = re.compile(r"^\s*#\s*(.*?)\n\s*(\w+)\s*(?::\s*TypeAlias\s*)?=\s*(.*)", re.MULTILINE)

    for match in type_pattern.finditer(content):
        summary, name, value = match.groups()
        type_definitions.append({"name": name, "value": value.strip(), "summary": summary.strip()})

    return type_definitions


def render_markdown_content(content):
    """
    Convert parsed markdown content back to plain markdown string.

    Args:
        content: Parsed content (list of strings/dicts or string).

    Returns:
        str: Plain markdown string.
    """
    if isinstance(content, str):
        return content
    if not content:
        return ""

    result = []
    for item in content:
        if isinstance(item, str):
            if item == "\n":
                result.append("\n\n")
            else:
                result.append(item)
        elif isinstance(item, dict):
            if "code" in item:
                result.append(f"`{item['code']}`")
            elif "link" in item:
                result.append(f"[{item['link'][1]}]({item['link'][0]})")
            elif "ref" in item:
                result.append(f"`{item['ref'][0]}`")
            elif "image" in item:
                result.append(f"![{item['image'][1]}]({item['image'][0]})")
    return " ".join(result).replace("  ", " ").replace(" \n\n ", "\n\n").strip()


def generate_function_markdown(name, data, prefix=""):
    """
    Generate markdown for a function or method.

    Args:
        name (str): Function name.
        data (dict): Function data with docstring, source, etc.
        prefix (str): Optional prefix for method names (e.g., "ClassName.").

    Returns:
        str: Markdown string for the function.
    """
    lines = []
    doc = data.get("docstring", {})

    # Check if internal
    if doc.get("Data", {}).get("internal"):
        return ""

    lines.append(f"### {prefix}{name}")
    lines.append("")

    # Summary
    if "Summary" in doc:
        lines.append(render_markdown_content(doc["Summary"]))
        lines.append("")

    # Args
    if "Args" in doc and doc["Args"]:
        lines.append("**Arguments:**")
        for arg in doc["Args"]:
            arg_type = ", ".join(arg.get("type", []))
            arg_info = render_markdown_content(arg.get("info", []))
            lines.append(f"- `{arg['name']}` ({arg_type}): {arg_info}")
        lines.append("")

    # Returns
    if "Returns" in doc and doc["Returns"]:
        ret = doc["Returns"]
        ret_type = ", ".join(ret.get("type", []))
        ret_info = render_markdown_content(ret.get("info", []))
        lines.append(f"**Returns:** {ret_type} - {ret_info}")
        lines.append("")

    # Examples
    if "Examples" in doc and doc["Examples"]:
        lines.append("**Example:**")
        lines.append("```python")
        lines.append(doc["Examples"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def generate_class_markdown(name, data):
    """
    Generate markdown for a class and its methods.

    Args:
        name (str): Class name.
        data (dict): Class data with docstring, methods, etc.

    Returns:
        str: Markdown string for the class.
    """
    lines = []
    doc = data.get("docstring", {})

    # Check if internal
    if doc.get("Data", {}).get("internal"):
        return ""

    lines.append(f"## {name}")
    lines.append("")

    # Summary
    if "Summary" in doc:
        lines.append(render_markdown_content(doc["Summary"]))
        lines.append("")

    # Methods
    methods = data.get("methods", {})
    if methods:
        for method_name, method_data in methods.items():
            method_md = generate_function_markdown(method_name, method_data, prefix=f"{name}.")
            if method_md:
                lines.append(method_md)

    return "\n".join(lines)


def convert_to_markdown(all_docstrings, output_file):
    """
    Convert parsed docstrings to a markdown file.

    Args:
        all_docstrings (dict): The parsed docstrings dictionary.
        output_file (str): Output markdown file path.
    """
    lines = []
    lines.append("# Pts.py API Documentation")
    lines.append("")
    lines.append("This documentation is auto-generated from source code docstrings.")
    lines.append("")

    # Type definitions
    if all_docstrings.get("typing"):
        lines.append("## Type Definitions")
        lines.append("")
        for typedef in all_docstrings["typing"]:
            lines.append(f"### {typedef['name']}")
            lines.append(f"{typedef['summary']}")
            lines.append(f"```python")
            lines.append(f"{typedef['name']} = {typedef['value']}")
            lines.append(f"```")
            lines.append("")

    # Packages
    for package_name, package_data in all_docstrings.get("packages", {}).items():
        display_name = "core" if package_name == "_" else package_name
        lines.append(f"# Package: {display_name}")
        lines.append("")

        # Module docstring
        module_doc = package_data.get("module", {}).get("docstring", {})
        if module_doc and "Summary" in module_doc:
            lines.append(render_markdown_content(module_doc["Summary"]))
            lines.append("")

        # Functions
        functions = package_data.get("functions", {})
        if functions:
            lines.append("## Functions")
            lines.append("")
            for func_name, func_data in functions.items():
                func_md = generate_function_markdown(func_name, func_data)
                if func_md:
                    lines.append(func_md)

        # Classes
        classes = package_data.get("classes", {})
        if classes:
            lines.append("## Classes")
            lines.append("")
            for class_name, class_data in classes.items():
                class_md = generate_class_markdown(class_name, class_data)
                if class_md:
                    lines.append(class_md)

    with open(output_file, "w") as md_file:
        md_file.write("\n".join(lines))


def generate_llms_full(preamble_path, docs_md_path, output_path):
    """
    Generate llms-full.txt by concatenating a preamble file with the generated docs.md.

    Args:
        preamble_path (str): Path to the preamble markdown file.
        docs_md_path (str): Path to the generated docs.md file.
        output_path (str): Output path for llms-full.txt.
    """
    with open(preamble_path, "r") as f:
        preamble = f.read()
    with open(docs_md_path, "r") as f:
        docs_md = f.read()
    with open(output_path, "w") as f:
        f.write(preamble)
        f.write(docs_md)


def convert_to_json(source_dir, output_json, output_markdown=None, preamble=None, llms_full=None):
    """
    Convert Python files with Google/Napoleon style documentation to JSON and optionally Markdown.

    Args:
        source_dir (str): Directory containing Python files.
        output_json (str): Output JSON file.
        output_markdown (str): Optional output Markdown file.
        preamble (str): Optional path to preamble file for llms-full.txt generation.
        llms_full (str): Optional output path for llms-full.txt.
    """
    all_docstrings = {"typing": [], "packages": {}}

    typing_file = os.path.join(source_dir, "typing.py")
    if os.path.exists(typing_file):
        all_docstrings["typing"] = parse_type_definitions(typing_file)

    for root, dirs, files in os.walk(source_dir):
        # Skip directories that start with '_'
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        package_name = os.path.relpath(root, source_dir).replace(os.sep, ".")
        if package_name == ".":
            package_name = "_"

        for file in files:
            if file.endswith(".py") and not file.startswith("_") and file != "typing.py":
                file_path = os.path.join(root, file)
                docstrings = extract_docstrings(file_path, source_dir)
                if docstrings is None:
                    continue
                if package_name not in all_docstrings["packages"]:
                    all_docstrings["packages"][package_name] = {"module": {}, "classes": {}, "functions": {}}
                if file == "__init__.py":
                    all_docstrings["packages"][package_name]["module"] = docstrings["module"]
                else:
                    all_docstrings["packages"][package_name]["functions"].update(docstrings["functions"])
                    all_docstrings["packages"][package_name]["classes"].update(docstrings["classes"])

    with open(output_json, "w") as json_file:
        json.dump(all_docstrings, json_file, indent=4)

    if output_markdown:
        convert_to_markdown(all_docstrings, output_markdown)

    if preamble and llms_full and output_markdown:
        generate_llms_full(preamble, output_markdown, llms_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Python docstrings to JSON and Markdown.")
    parser.add_argument("-i", "--input", default="../src/ptspy", help="Input source directory (default: ../src/ptspy)")
    parser.add_argument(
        "-o", "--output", default="./src/assets/docs.json", help="Output JSON file (default: ./src/assets/docs.json)"
    )
    parser.add_argument(
        "-m", "--markdown", default="./public/docs.md", help="Output Markdown file (default: ./public/docs.md)"
    )
    parser.add_argument("-p", "--preamble", default=None, help="Path to preamble file for llms-full.txt generation")
    parser.add_argument("-l", "--llms-full", default=None, help="Output path for llms-full.txt")
    args = parser.parse_args()

    convert_to_json(args.input, args.output, args.markdown, args.preamble, args.llms_full)
