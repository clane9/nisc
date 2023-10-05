from typing import Optional, Tuple

import numpy as np
import pytest

from flatmaps.resample import Bbox, Resampler


@pytest.fixture
def points() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(size=(100, 2))


@pytest.fixture
def label() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 10, size=(100,))


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize(
    "rect,pad_to_multiple,shape",
    [
        (None, None, (101, 97)),
        (Bbox(0.0, 1.0, 0.0, 1.0), None, (101, 101)),
        (Bbox(0.0, 1.0, 0.0, 1.0), 16, (112, 112)),
    ],
)
def test_resampler(
    points: np.ndarray,
    label: np.ndarray,
    interpolation: str,
    rect: Optional[Bbox],
    pad_to_multiple: Optional[int],
    shape: Tuple[int, int],
):
    resampler = Resampler(0.01, rect=rect, pad_to_multiple=pad_to_multiple)
    resampler.fit(points)

    # Check the grid shape
    assert resampler.grid_shape_ == shape

    # Check that the bounding box is correct
    xmin, xmax, ymin, ymax = resampler.bbox_
    assert np.allclose(resampler.x_[[0, -1]], [xmin, xmax])
    assert np.allclose(resampler.y_[[0, -1]], [ymax, ymin])

    # Check the transformed image shape
    img = resampler.transform(label, categorical=True)
    assert img.shape == resampler.grid_shape_

    # Check that the inverse transform is correct
    img = resampler.apply_mask(img, fill_value=0)
    label2 = resampler.inverse(img, categorical=True, interpolation=interpolation)
    assert np.all(label == label2)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
