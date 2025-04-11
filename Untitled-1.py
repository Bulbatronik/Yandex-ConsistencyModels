import nbformat

# Load your notebook
with open('ConsistencyModels.ipynb', 'r') as f:
    notebook = nbformat.read(f, as_version=4)

# Check and update metadata for widgets
for cell in notebook.cells:
    if 'widgets' in cell.metadata:
        for widget in cell.metadata['widgets']:
            if 'state' not in widget:
                widget['state'] = {}  # Provide an empty state or necessary state data

# Save the notebook with updated metadata
with open('ConsistencyModels_fixed.ipynb', 'w') as f:
    nbformat.write(notebook, f)