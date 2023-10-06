"""
Cortical surface representation.

Example::

    surf = load_flat()
    rois = cortex.get_roi_verts("fsaverage")
    roi_polys = {k: surf.roi_to_poly(indices=v) for k, v in rois.items()}
    patch = surf.extract_patch(rois["V1"])
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional, Tuple, Union

import cortex
import numpy as np
import shapely


@dataclass
class Surface:
    """
    A triangulated surface.
    """

    points: np.ndarray
    polys: np.ndarray

    def __post_init__(self):
        self.polys = self.polys.astype(np.int64)

        if not (self.points.ndim == 2 and self.points.shape[1] in (2, 3)):
            raise ("Invalid points; expected shape (num_points, {2, 3})")
        if not (self.polys.ndim == 2 and self.polys.shape[1] == 3):
            raise ValueError("Invalid polys; expected shape (num_polys, 3)")
        if not (self.polys.min() >= 0 and self.polys.max() < len(self.points)):
            raise ValueError("Invalid indices in polys")

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """
        Get the mask of valid vertices.
        """
        valid_points = np.unique(self.polys)
        valid_mask = _indicator(valid_points, len(self))
        return valid_mask

    def merge(self, other: "Surface") -> "Surface":
        """
        Merge surface with another.
        """
        points = np.concatenate([self.points, other.points])
        polys = np.concatenate([self.polys, other.polys + len(self)])
        return Surface(points, polys)

    def roi_to_poly(
        self,
        *,
        indices: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        simplify_tolerance: Optional[float] = None,
    ) -> shapely.MultiPolygon:
        """
        Convert an ROI to a shapely multi polygon.

        Args:
            indices: ROI indices
            mask: ROI mask, if indices not given
            simplify_tolerance: shapely polygon simplification tolerance in
                surface coordinate units.
        """
        if indices is None and mask is None:
            raise ValueError("indices or mask is required")

        if indices is not None:
            mask = _indicator(indices, len(self))
        else:
            mask = mask.astype(bool)

        if not (mask.ndim == 1 and len(mask) == len(self)):
            raise ValueError("Invalid mask; expected shape (num_points,)")

        poly_mask = mask[self.polys]
        poly_mask = np.all(poly_mask, axis=1)
        mask_polys = self.polys[poly_mask]
        mask_poly_pts = self.points[mask_polys]

        geoms = shapely.polygons(mask_poly_pts)
        boundary = shapely.unary_union(geoms)
        if simplify_tolerance:
            boundary = boundary.simplify(simplify_tolerance)
        if not isinstance(boundary, shapely.MultiPolygon):
            boundary = shapely.MultiPolygon([boundary])
        return boundary

    def extract_patch(
        self,
        *,
        indices: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> "Surface":
        """
        Extract the surface patch for the ROI indices or mask.
        """
        if indices is None and mask is None:
            raise ValueError("indices or mask is required")

        if indices is not None:
            mask = _indicator(indices, len(self))
        else:
            mask = mask.astype(bool)

        mask_points = self.points[mask]
        mask_indices = np.cumsum(mask) - 1
        poly_mask = mask[self.polys]
        poly_mask = np.all(poly_mask, axis=1)
        mask_polys = self.polys[poly_mask]
        mask_polys = mask_indices[mask_polys]
        return Surface(mask_points, mask_polys)

    def __len__(self) -> int:
        return len(self.points)


def _indicator(indices: np.ndarray, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[indices] = True
    return mask


def load_flat(
    subject: str = "fsaverage",
    hemi: Optional[Literal["lh", "rh"]] = None,
    padding: float = 2.0,
) -> Surface:
    """
    Load canonical fsaverage flat surface from pycortex.

    Args:
        hemi: "lh" or "rh". If None, load both and concatenate.
    """
    if hemi not in (None, "lh", "rh"):
        raise ValueError(f"Invalid hemi {hemi}")

    if hemi is None:
        surf_lh = load_flat(subject=subject, hemi="lh")
        surf_rh = load_flat(subject=subject, hemi="rh")
        surf = surf_lh.merge(surf_rh)
    else:
        points, polys = cortex.db.get_surf(subject, "flat", hemisphere=hemi)

        # Keep only x, y
        points = points[:, :2].copy()

        # Separate hemispheres
        if hemi == "lh":
            points[:, 0] = points[:, 0] - points[:, 0].max() - padding
        else:
            points[:, 0] = points[:, 0] - points[:, 0].min() + padding
        surf = Surface(points, polys)
    return surf
