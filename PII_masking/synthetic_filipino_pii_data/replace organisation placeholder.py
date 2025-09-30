import random

# List of organizations to replace {{organization}}
organizations = [
    "Maya Bank", "P&G Collection", "CIC", "Banko Central",
    "BDO", "Metrobank", "UnionBank", "PNG Collection", "Maya Credit"
]

# Input and output file paths
input_file = "templates.txt"
output_file = "updated_templates.txt"

# Read templates from file
try:
    with open(input_file, "r", encoding="utf-8") as f:
        templates = f.read()
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit(1)

# Split templates into a list (assuming Python list format in templates.txt)
try:
    # Evaluate the file content as a Python list
    templates_list = eval(templates.replace("templates = ", ""))
    if not isinstance(templates_list, list):
        raise ValueError("Templates file must contain a valid Python list")
except Exception as e:
    print(f"Error parsing templates: {e}")
    exit(1)

# Replace {{organization}} with a random organization name
updated_templates = []
for template in templates_list:
    if "{{organization}}" in template:
        # Replace with a random organization
        updated_template = template.replace("{{organization}}", random.choice(organizations))
        updated_templates.append(updated_template)
    else:
        updated_templates.append(template)

# Verify that the number of templates remains the same
if len(templates_list) != len(updated_templates):
    print(f"Error: Template count mismatch. Original: {len(templates_list)}, Updated: {len(updated_templates)}")
    exit(1)
else:
    print(f"Template count verified: {len(templates_list)} templates in both original and updated lists")

# Save updated templates
try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("templates = [\n")
        for template in updated_templates:
            f.write(f'    "{template}",\n')
        f.write("]\n")
    print(f"Updated templates saved to {output_file} with {len(updated_templates)} templates")
except Exception as e:
    print(f"Error saving updated templates: {e}")
    exit(1)

# Verify replacement
original_count = sum(1 for t in templates_list if "{{organization}}" in t)
updated_count = sum(1 for t in updated_templates if "{{organization}}" in t)
print(f"Original templates with {{organization}}: {original_count}")
print(f"Updated templates with {{organization}}: {updated_count}")