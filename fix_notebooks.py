import json
import re
from pathlib import Path

def fix_notebook(nb_path):
    """Fix common issues in notebooks"""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell.get('source', [])
        if not source:
            continue
        
        new_source = []
        i = 0
        
        while i < len(source):
            line = source[i]
            
            # Fix 1: Update sys.path resolution logic
            if 'for extra in (_base, _base.parent, _base.parent.parent):' in line:
                new_source.append('for extra in (_base, _base.parent, _base.parent.parent, _base.parent.parent.parent):\n')
                i += 1
                # Skip to the candidate line
                while i < len(source) and 'candidate = extra' not in source[i]:
                    new_source.append(source[i])
                    i += 1
                if i < len(source):
                    # Add new logic
                    new_source.append('    candidate = extra / "scripts"\n')
                    new_source.append('    if candidate.exists() and str(candidate) not in sys.path:\n')
                    new_source.append('        sys.path.insert(0, str(candidate))\n')
                    new_source.append('        break\n')
                    # Add original candidate line
                    new_source.append(source[i])
                    i += 1
                modified = True
                continue
            
            # Fix 2: Remove extra indentation on widget definitions at start
            if re.match(r'^    (spot0_slider|spotT_slider|slider_|output = widgets\.Output)', line):
                # Check if previous line is empty or not a function def
                if i == 0 or (i > 0 and not 'def ' in source[i-1]):
                    line = line[4:]  # Remove 4 spaces
                    modified = True
            
            # Fix 3: Fix function definition followed by wrong-level with statement
            if re.match(r'^    def _update', line):
                new_source.append(line[4:])  # Remove 4 spaces from def
                i += 1
                # Fix the with statement that follows
                if i < len(source) and re.match(r'^    with (output|out):', source[i]):
                    new_source.append('    ' + source[i])  # Should be indented 4 spaces from def
                    i += 1
                    modified = True
                continue
            
            # Fix 4: Fix ending for/display statements that are over-indented
            if re.match(r'^        (for sl in|_update\(\)|display\(widgets\.)', line):
                line = line[4:]  # Remove 4 spaces
                modified = True
            
            new_source.append(line)
            i += 1
        
        if modified:
            cell['source'] = new_source
    
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        return True
    return False

# Main
notebooks_dir = Path(r'd:\PythonDProjects\PaperTradingApp\notebooks\GPT')
fixed_count = 0

for nb_path in notebooks_dir.rglob('*.ipynb'):
    if nb_path.name == 'tmp.ipynb':
        continue
    
    try:
        if fix_notebook(nb_path):
            fixed_count += 1
            print(f'Fixed: {nb_path.relative_to(notebooks_dir)}')
    except Exception as e:
        print(f'Error processing {nb_path.name}: {e}')

print(f'\nTotal notebooks fixed: {fixed_count}')
