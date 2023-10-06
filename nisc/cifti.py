"""
Utility function to extract cortical surface data from a cifti image.

Example::
    cifti = nib.load("cifti.dscalar.nii")
    lh_data = get_cifti_surf_data(cifti, "lh")
    rh_data = get_cifti_surf_data(cifti, "rh")

References:
    https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    https://neurostars.org/t/separate-cifti-by-structure-in-python/17301/2
"""

from typing import Literal, Optional

import numpy as np
from nibabel.cifti2 import BrainModelAxis
from nibabel.cifti2.cifti2 import Cifti2Image


def get_cifti_surf_data(
    cifti: Cifti2Image,
    hemi: Optional[Literal["lh", "rh"]] = None,
):
    if hemi not in (None, "lh", "rh"):
        raise ValueError(f"Invalid hemi {hemi}")

    if hemi is None:
        lh_data = get_cifti_surf_data(cifti, "lh")
        rh_data = get_cifti_surf_data(cifti, "rh")
        return np.concatenate([lh_data, rh_data], axis=0)

    struct = {
        "lh": "CIFTI_STRUCTURE_CORTEX_LEFT",
        "rh": "CIFTI_STRUCTURE_CORTEX_RIGHT",
    }[hemi]
    return get_cifti_struct_data(cifti, struct)


def get_cifti_struct_data(cifti: Cifti2Image, struct: str) -> np.ndarray:
    """
    Get cifti scalar/series data for a given brain structure.
    """
    axis = get_brain_model_axis(cifti)
    if axis is None:
        raise ValueError("No brain model axis found in cifti")

    data = cifti.get_fdata().T

    for name, indices, model in axis.iter_structures():
        if name == struct:
            num_verts = model.vertex.max() + 1
            struct_data = np.zeros((num_verts,) + data.shape[1:], dtype=data.dtype)
            struct_data[model.vertex] = data[indices]
            return struct_data
    raise ValueError(f"Invalid cifti struct {struct}")


def get_brain_model_axis(cifti: Cifti2Image) -> Optional[BrainModelAxis]:
    for ii in range(cifti.ndim):
        axis = cifti.header.get_axis(ii)
        if isinstance(axis, BrainModelAxis):
            return axis
    return None
