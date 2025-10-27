# Fitness Wingman

Lightweight research project demonstrating pose estimation using the MoveNet model (TensorFlow Hub) inside a Jupyter notebook (poc.ipynb). The notebook contains code to download images, run MoveNet on a folder of images, and display pose overlays in a grid.

## Requirements
- Python 3.8+
- pip
- Jupyter / JupyterLab (recommended for running the notebook)

Recommended packages (used in poc.ipynb):
- tensorflow
- tensorflow-hub
- opencv-python
- imageio
- matplotlib
- ddgs
- tqdm
- git+https://github.com/tensorflow/docs

## Quick setup (Windows)
Open PowerShell or cmd:

# Create & activate venv (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install deps
python -m pip install --upgrade pip
pip install tensorflow tensorflow-hub opencv-python imageio matplotlib ddgs tqdm
pip install git+https://github.com/tensorflow/docs

## Notebook (poc.ipynb)
- Open `research/poc.ipynb` in Jupyter.
- Cells download images (into `research/workout_images_ddgs`), load MoveNet, and run inference.
- The notebook includes `draw_prediction_on_image` and helper functions that convert Matplotlib figures to numpy arrays robustly.
- To run inference over a folder and display results in a grid, use the helper in `research/run_pose_on_folder.py`:
  - process_folder(folder)  — sequential display
  - process_folder_grid(folder, cols=3, batch_size=None) — grid display per batch

Example (inside notebook cell):
```python
%load_ext autoreload
%autoreload 2
from research.run_pose_on_folder import process_folder_grid
process_folder_grid(r".\\research\\workout_images_ddgs", cols=3, batch_size=6)
```

Notes:
- If you edit `research/run_pose_on_folder.py`, re-import in the notebook:
  - Option A (manual reload):
    ```python
    import importlib
    import research.run_pose_on_folder as rpf
    importlib.reload(rpf)
    from research.run_pose_on_folder import process_folder_grid
    ```
  - Option B (autoreload):
    ```python
    %load_ext autoreload
    %autoreload 2
    from research.run_pose_on_folder import process_folder_grid
    ```
- If images appear stacked, use the grid display (`process_folder_grid`) or adjust `cols` / `batch_size`.
- Image downloads are placed in `research/workout_images_ddgs` by the notebook's DDGS helper.

## Usage tips
- Restart kernel if package versions change.
- For GPU acceleration, install the TensorFlow package matching your CUDA/cuDNN setup.
- The notebook is the primary research artifact — use it to iterate quickly.

## License
Public domain (CC0). See LICENSE for details.