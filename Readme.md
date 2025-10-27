# Fitness Wingman

Lightweight project demonstrating pose estimation using the MoveNet model from TensorFlow Hub. Intended as a starting point for building workout form detection, repetition counting, or pose-based feedback.

## Requirements
- Python 3.8+
- pip

Recommended Python packages:
- tensorflow
- tensorflow-hub
- opencv-python
- imageio
- git+https://github.com/tensorflow/docs

## Quick setup (cross-platform)

Create and activate a virtual environment:

```sh
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (cmd)
python -m venv venv
venv\Scripts\activate.bat
```

Install dependencies:

```sh
pip install --upgrade pip
pip install tensorflow tensorflow-hub opencv-python imageio
pip install git+https://github.com/tensorflow/docs
```

## Usage
1. Prepare input video or camera feed.
2. Follow the MoveNet tutorial linked above to load the model and run inference.
3. Add post-processing logic for reps, form checks, or visual overlays.

## Notes
- For GPU acceleration, install the appropriate TensorFlow package for your CUDA/cuDNN setup.
- This repository contains only starter material; extend with data collection, evaluation, and UI as needed.

## License
This project is dedicated to the public domain under CC0 1.0 Universal. See [LICENSE](LICENSE) for details.

## Files
- [Readme.md](Readme.md)
- [LICENSE](LICENSE)
- [.gitignore](.gitignore)