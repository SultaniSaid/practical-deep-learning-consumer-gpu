import nbformat as nbf
import os
from pathlib import Path

def repair_notebooks():
    notebooks = [
        'lesson1_segmentdata.ipynb',
        'lesson1_collabdata.ipynb',
        'lesson1_visiondata.ipynb',
        'lesson1_textdata.ipynb',
        'lesson1_tabulardata.ipynb',
        'lesson1_gpu_vs_cpu.ipynb'
    ]
    
    base_path = Path('c:/Users/Said/k3sh4v_practicaldeeplearningconsumergpu/practical-deep-learning-consumer-gpu')
    
    setup_code = [
        "import torch_directml\n",
        "import re, sys, os\n",
        "from pathlib import Path\n",
        "from dml_fastai_utils import setup_dml, get_local_path\n",
        "from fastai.vision.all import *\n",
        "\n",
        "# Initialize DirectML and apply global Vanguard patches\n",
        "dml = setup_dml()\n",
        "local_data_path = get_local_path()"
    ]

    for nb_name in notebooks:
        nb_path = base_path / nb_name
        if not nb_path.exists():
            print(f"Skipping {nb_name} (not found)")
            continue
            
        print(f"Repairing {nb_name}...")
        nb = nbf.read(nb_path, as_version=4)
        
        # 1. Look for existing DML setup cells and replace them
        # 2. Or inject at the top
        new_cells = []
        found_import = False
        
        # Remove old patching cells
        for cell in nb.cells:
            # Detect manual patching cells
            src = cell.source.lower()
            if "setup_dml" in src:
                if not found_import:
                    cell.source = "".join(setup_code)
                    new_cells.append(cell)
                    found_import = True
            elif "old_init = normalize.__init__" in src or "learner.freeze_to =" in src:
                print(f"  Removing manual patch cell in {nb_name}")
                continue # Skip old patch cells
            elif "learn = learn" in src and "disables mixed precision" in src:
                # Update the FP16 line to be cleaner
                cell.source = cell.source.replace("learn = learn", "# MixedPrecision disabled globally by setup_dml()")
                new_cells.append(cell)
            else:
                new_cells.append(cell)
        
        if not found_import:
            print(f"  Injecting setup_dml at the top of {nb_name}")
            import_cell = nbf.v4.new_code_cell("".join(setup_code))
            new_cells.insert(0, import_cell)
            
        nb.cells = new_cells
        nbf.write(nb, nb_path)
        print(f"  Successfully repaired {nb_name}")

if __name__ == "__main__":
    repair_notebooks()
