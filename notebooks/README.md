# Interactive Notebooks

This directory contains Jupyter notebooks for interactive experimentation, visualization, and faculty demonstrations.

---

## 1. Available Notebooks

- **`deepfake_detection_walkthrough.ipynb`**:
  - Environment and GPU device verification
  - EfficientNet-B0 backbone architecture inspection (5.3M parameters, classification head)
  - Interactive demonstration and visual display of real-world post-processing perturbations (JPEG compression, bilinear resizing, Gaussian blur, Gaussian noise, contrast/sharpness enhancement)
  - Comparative benchmark visualization reproducing Table II results from the research paper

---

## 2. Launching Jupyter Notebooks

Ensure your virtual environment is active and dependencies are installed:

```bash
# Install jupyter if not already installed
pip install jupyterlab ipywidgets

# Launch Jupyter Lab or Notebook
jupyter lab
# or:
# jupyter notebook notebooks/deepfake_detection_walkthrough.ipynb
```
