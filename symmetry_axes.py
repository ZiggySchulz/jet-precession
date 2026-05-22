from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve


def _noise_map(
    shape: tuple[int, int], sensitivity: float, kernel: Gaussian2DKernel
) -> np.ndarray:
    assert (
        kernel.model is not None
    ), "Gaussian2DKernel must have a model attribute to retrieve x_stddev and y_stddev"
    x_stddev = kernel.model.x_stddev.value
    y_stddev = kernel.model.y_stddev.value
    pixels_per_beam = np.pi * x_stddev * y_stddev * (2 * np.log(2))
    sigma = sensitivity / np.sqrt(10.0) * np.sqrt(pixels_per_beam)
    return convolve(np.random.normal(loc=0.0, scale=sigma, size=shape), kernel)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(
        np.nan_to_num(image, copy=True, nan=0.0), angle, reshape=False, order=3
    )


def _symmetry_index_squared(
    image: np.ndarray, detectable: np.ndarray, power: int
) -> float:
    """Compute squared symmetry index of order 'power' for the image.
    Parameters as below
    """
    columns = image.shape[1]
    weighted_image = image**power
    left = np.sum(
        weighted_image[:, : columns // 2] * detectable[:, : columns // 2], axis=1
    )
    right = np.sum(
        weighted_image[:, (columns + 1) // 2 :] * detectable[:, (columns + 1) // 2 :],
        axis=1,
    )
    total = left + right
    return float(np.sum((left - right) ** 2 * total) / np.sum(total))


def symmetry_index(image: np.ndarray, detectable: np.ndarray, power: int) -> float:
    """Compute symmetry index of order 'power' for the image.

    Parameters
    ----------
    image : np.ndarray
        Image data array.
    detectable : np.ndarray
        Binary mask of detectable pixels.
    power : int
        Power to raise image values to (0, 1, or 2).

    Returns
    -------
    float
        Symmetry index value.
    """
    return np.sqrt(_symmetry_index_squared(image, detectable, power=power))


def calculate_symmetry_axes(
    angles: ArrayLike,
    surface_brightness: np.ndarray,
    sensitivity: float,
    gauss_kernel: Gaussian2DKernel,
    nsamples: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute symmetry axes using a Monte Carlo simulation for n=0,1,2.

    Parameters
    ----------
    angles : array-like
        Candidate axis angles in degrees. Use this to specify the range and resolution of angles to test for symmetry.
    surface_brightness : ndarray
        Image data array
    sensitivity : float
        Detection threshold for one or two images.
    gauss_kernel : astropy.convolution.Gaussian2DKernel
        Beam kernel(s) used for noise convolution.
    nsamples : int
        Number of Monte Carlo noise realizations.

    Returns
    -------
    tuple of angles in degrees and std deviations for n=0,1,2 symmetry axes as
        ((angle1, deviation1), (angle2, deviation2), (angle3, deviation3)).
    """
    angles = np.asarray(angles)
    image = np.asarray(surface_brightness)
    image_shape = image.shape
    rot_img = np.stack([_rotate_image(image, a) for a in angles])

    results = np.zeros((3, nsamples, len(angles)))
    powers = (0, 1, 2)

    for i in range(nsamples):
        nse_img = _noise_map(image_shape, sensitivity, gauss_kernel)

        for j in range(len(angles)):
            image_mc = rot_img[j] + nse_img
            detectable = np.zeros_like(image_mc, dtype=np.int_)
            detectable[image_mc >= sensitivity] = 1

            for k, power in enumerate(powers):
                results[k, i, j] = _symmetry_index_squared(
                    image_mc, detectable, power=power
                )

    idx = np.argmin(results, axis=2)
    angles_best = [float(np.mean(angles[idx_k])) for idx_k in idx]
    deviations = [float(np.std(angles[idx_k])) for idx_k in idx]

    return (
        (angles_best[0], deviations[0]),
        (angles_best[1], deviations[1]),
        (angles_best[2], deviations[2]),
    )
