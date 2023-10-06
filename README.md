# nisc

Miscellaneous tools for neuroimaging data.

- [`cifti`](nisc/cifti.py): utils for working with CIFTI files
  - `get_cifti_surf_data`: get surface data from a metric CIFTI file
- [`surface`](nisc/surface.py): utils for working with surface data
  - `Surface`: a triangular mesh surface representation
  - `load_flat`: a helper for loading a [pycortex](https://github.com/gallantlab/pycortex) flat map
- [`resample`](nisc/resample.py): resample data from a flat surface to a regular image grid.


## Installation

```bash
pip install git+https://github.com/clane9/nisc.git
```
