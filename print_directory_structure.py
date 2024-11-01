import os

def print_directory_structure(path, indent_level=0):
    ignore_dirs = {'venv', 'mlruns', '__pycache__'}
    if indent_level == 0:
        print(f"{os.path.basename(path)}/")
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item not in ignore_dirs and not item.startswith('.'):
            print('  ' * indent_level + '|-- ' + item + '/')
            print_directory_structure(item_path, indent_level + 1)

# Call the function with the current working directory
print_directory_structure(os.getcwd())





