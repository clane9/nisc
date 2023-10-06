"""
Resample scattered data to a fixed grid using Gaussian weighted averaging.

Example::

    points = np.random.rand(100, 2)
    label = np.random.randint(0, 10, 100)

    resampler = Resampler(pixel_size=0.1, rect=(0, 1, 0, 1)).fit(points)
    label_img = resampler.transform(label, categorical=True)
    label_inv = resampler.inverse(label_img)
"""

import math
from typing import Any, List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


class Bbox(NamedTuple):
    """
    Bounding box with format (xmin, xmax, ymin, ymax).
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float


class Resampler:
    """
    Resample scattered data to a fixed grid using Gaussian weighted averaging.

    Args:
        pixel_size: size of desired pixels in original units.
        fwhm: Gaussian FWHM in pixel units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.
    """

    def __init__(
        self,
        pixel_size: float,
        fwhm: float = 1.0,
        rect: Optional[Bbox] = None,
        pad_width: Optional[Union[int, List[Tuple[int, int]]]] = None,
        pad_to_multiple: Optional[int] = None,
    ):
        self.pixel_size = pixel_size
        self.fwhm = fwhm
        self.rect = None if rect is None else Bbox(*rect)
        self.pad_width = pad_width
        self.pad_to_multiple = pad_to_multiple
        # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
        self._sigma = (pixel_size * fwhm) / 2.35482004503

        self._reset_data()

    def _reset_data(self):
        self.points_ = None
        self.bbox_ = None
        self.grid_ = None
        self.x_ = None
        self.y_ = None
        self.weight_ = None
        self.density_ = None
        self.mask_ = None
        self.point_mask_ = None

    @property
    def grid_shape_(self) -> Optional[Tuple[int, int]]:
        """
        Shape of grid, (height, width).
        """
        return None if self.grid_ is None else self.grid_.shape[:2]

    @property
    def grid_size_(self) -> Optional[Tuple[int, int]]:
        """
        Size of grid, (width, height).
        """
        return None if self.grid_ is None else self.grid_.shape[1::-1]

    @property
    def flat_grid_(self) -> Optional[np.ndarray]:
        """
        Flattened grid, shape (height * width, 2).
        """
        return None if self.grid_ is None else self.grid_.reshape(-1, 2)

    def fit(self, points: np.ndarray) -> "Resampler":
        """
        Fit resampler to scattered points, shape (n_points, 2).
        """
        self._reset_data()
        self.points_ = points
        self.grid_, self.bbox_ = fit_grid(
            points,
            pixel_size=self.pixel_size,
            rect=self.rect,
            pad_width=self.pad_width,
            pad_to_multiple=self.pad_to_multiple,
        )
        self.x_ = np.ascontiguousarray(self.grid_[0, :, 0])
        self.y_ = np.ascontiguousarray(self.grid_[:, 0, 1])

        # Sparse nearest neighbors graph
        nbrs = NearestNeighbors()
        nbrs.fit(points)
        radius = 3 * self._sigma
        weight = nbrs.radius_neighbors_graph(
            self.flat_grid_, radius=radius, mode="distance"
        )

        # Gaussian averaging weights
        weight.data = np.exp(-0.5 * weight.data**2 / self._sigma**2)
        density = np.asarray(weight.sum(axis=1))
        weight = weight.multiply(1 / np.where(density == 0, 1e-8, density))
        self.weight_ = weight.tocsr()
        self.density_ = density.reshape(self.grid_shape_)
        self.mask_ = (density > 0).reshape(self.grid_shape_)

        # Mask of points contained in bbox
        self.point_mask_ = (
            (points[:, 0] >= self.bbox_.xmin)
            & (points[:, 0] <= self.bbox_.xmax)
            & (points[:, 1] >= self.bbox_.ymin)
            & (points[:, 1] <= self.bbox_.ymax)
        )
        return self

    def transform(
        self,
        data: np.ndarray,
        categorical: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Transform scattered data, shape (n_points, c), into an image, shape (h, w, c)
        using Gaussian weighted averaging.
        """
        self._check_fit()
        if categorical is None:
            categorical = is_categorical(data.dtype)
        if not (data.ndim in (1, 2) and len(data) == len(self.points_)):
            raise ValueError(
                "Invalid data; expected shape (n_points, c) or (n_points,)"
            )
        if categorical and data.ndim == 2 and data.shape[1] > 1:
            raise ValueError("Multiple categorical channels not supported")

        # Apply the Gaussian weighted averaging using sparse matrix multiply.
        shape = data.shape
        if categorical:
            data, uniq = label_to_one_hot(data)
        image = self.weight_ @ (data[:, None] if data.ndim == 1 else data)
        if categorical:
            image = one_hot_to_label(image, uniq)
        image = image.reshape(self.grid_shape_ + shape[1:])
        return image

    def inverse(
        self,
        image: np.ndarray,
        categorical: Optional[bool] = None,
        interpolation: Literal["linear", "nearest"] = "nearest",
    ) -> np.ndarray:
        """
        Transform an image, shape (h, w, c), back onto scattered points, shape
        (n_points, c), using interpolation.
        """
        self._check_fit()
        if categorical is None:
            categorical = is_categorical(image.dtype)
        if image.shape[:2] != self.grid_shape_:
            raise ValueError("Image doesn't match grid")
        if image.ndim not in (2, 3):
            raise ValueError("Invalid image shape; expected (h, w), or (h, w, c)")

        if categorical:
            image, uniq = label_to_one_hot(image)
        # reversing x, y since interpn expects 'ij' ordering
        points = self.points_[:, [1, 0]]
        data = np.zeros((len(points),) + image.shape[2:], dtype=image.dtype)
        data[self.point_mask_] = interpolate.interpn(
            (self.y_, self.x_), image, points[self.point_mask_], method=interpolation
        )
        if categorical:
            data = one_hot_to_label(data, uniq).astype(image.dtype)
        return data

    def apply_mask(self, image: np.ndarray, fill_value: Any = np.nan) -> np.ndarray:
        """
        Apply the valid point mask to the image.
        """
        self._check_fit()
        if image.shape[:2] != self.grid_shape_:
            raise ValueError("Image doesn't match grid")
        if image.ndim not in (2, 3):
            raise ValueError("Invalid image shape; expected (h, w), or (h, w, c)")
        mask = self.mask_ if image.ndim == 2 else self.mask_[..., None]
        image = np.where(mask, image, fill_value)
        return image

    def _check_fit(self):
        assert self.points_ is not None, "resampler still needs to be fit"


def fit_grid(
    points: np.ndarray,
    pixel_size: float,
    rect: Optional[Bbox] = None,
    pad_width: Optional[Union[int, List[Tuple[int, int]]]] = None,
    pad_to_multiple: Optional[int] = None,
) -> Tuple[np.ndarray, Bbox]:
    """
    Fit a pixel grid to scattered points with desired padding and pixel size.

    Args:
        points: array of (x, y) points, shape (num_points, 2).
        pixel_size: pixel size in data units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.

    Returns:
        A tuple containing the grid points, shape (height, width, 2), and the grid
            bounding box. The origin of the grid is the upper left corner, (xmin, ymax).
    """
    if pad_width and pad_to_multiple:
        raise ValueError("Cannot use both pad_width and pad_to_multiple")

    if rect is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
    else:
        xmin, xmax, ymin, ymax = rect

    xmin = pixel_size * math.floor(xmin / pixel_size)
    xmax = pixel_size * math.ceil(xmax / pixel_size)
    ymin = pixel_size * math.floor(ymin / pixel_size)
    ymax = pixel_size * math.ceil(ymax / pixel_size)

    if pad_to_multiple is not None:
        w = round((xmax - xmin) / pixel_size) + 1
        h = round((ymax - ymin) / pixel_size) + 1
        padw = math.ceil(w / pad_to_multiple) * pad_to_multiple - w
        padh = math.ceil(h / pad_to_multiple) * pad_to_multiple - h
        pad_width = [
            (padh // 2, padh - padh // 2),
            (padw // 2, padw - padw // 2),
        ]

    if pad_width is not None:
        if isinstance(pad_width, int):
            pad_width = [(pad_width, pad_width)] * 2
        ymax += pixel_size * pad_width[0][0]
        ymin -= pixel_size * pad_width[0][1]
        xmin -= pixel_size * pad_width[1][0]
        xmax += pixel_size * pad_width[1][1]

    w = round((xmax - xmin) / pixel_size) + 1
    h = round((ymax - ymin) / pixel_size) + 1
    x = xmin + pixel_size * np.arange(w)
    y = ymax - pixel_size * np.arange(h)
    grid = np.stack(np.meshgrid(x, y), axis=-1)
    return grid, Bbox(xmin, xmax, ymin, ymax)


def is_categorical(dtype: np.dtype) -> bool:
    """
    Return True if the dtype is categorical.
    """
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)


def label_to_one_hot(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a categorical label to one-hot representation. Return the one-hot
    representation and the unique label values.
    """
    shape = label.shape
    label = label.flatten()
    uniq, label = np.unique(label, return_inverse=True)
    one_hot = np.zeros((len(label), len(uniq)))
    one_hot[np.arange(len(label)), label] = 1.0
    if len(shape) > 1:
        one_hot = one_hot.reshape(shape + (len(uniq),))
    return one_hot, uniq


def one_hot_to_label(
    one_hot: np.ndarray, uniq: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a one-hot to categorical representation.
    """
    label = np.argmax(one_hot, axis=-1)
    if uniq is not None:
        label = uniq[label]
    return label
