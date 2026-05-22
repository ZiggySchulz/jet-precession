from __future__ import annotations

import numpy as np
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
from typing import Sequence


def _noise_map(
    shape: tuple[int, int], sensitivity: float, kernel: Gaussian2DKernel
) -> np.ndarray:
    x_stddev = kernel.x_stddev.value
    y_stddev = kernel.y_stddev.value
    pixels_per_beam = np.pi * x_stddev * y_stddev * (2 * np.log(2))
    sigma = sensitivity / np.sqrt(10.0) * np.sqrt(pixels_per_beam)
    return convolve(np.random.normal(loc=0.0, scale=sigma, size=shape), kernel)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(
        np.nan_to_num(image, copy=True, nan=0.0), angle, reshape=False, order=3
    )


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


def calculate_symmetry_axes(
    angles: Sequence[float],
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
    surface_brightness : ndarray or sequence of two ndarrays
        Image data array, or a pair of images used for spectral-axis computation.
    sensitivity : float or sequence of two floats
        Detection threshold for one or two images.
    gauss_kernel : astropy.convolution.Gaussian2DKernel or sequence of two kernels
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
                results[k, i, j] = symmetry_index(image_mc, detectable, power=power)

    idx = np.argmin(results, axis=2)
    angles_best = [np.mean(angles[idx_k]) for idx_k in idx]
    deviations = [np.std(angles[idx_k]) for idx_k in idx]

    return (
        (angles_best[0], deviations[0]),
        (angles_best[1], deviations[1]),
        (angles_best[2], deviations[2]),
    )

def calculate_spectral_index_symmetry_axes(
    angles: Sequence[float],
    image1: np.ndarray,
    image2: np.ndarray,
    sensitivity1: float,
    sensitivity2: float,
    kernel1: Gaussian2DKernel,
    kernel2: Gaussian2DKernel,
    nsamples: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute symmetry axes for spectral index maps using a Monte Carlo simulation for n=0,1,2."""
    
    angles = np.asarray(angles)
    image_shape = image1.shape
    rot_img1 = np.stack([_rotate_image(image1, a) for a in angles])
    rot_img2 = np.stack([_rotate_image(image2, a) for a in angles])

    results = np.zeros((3, nsamples, len(angles)))
    powers = (0, 1, 2)

    for i in range(nsamples):
        nse_img1 = _noise_map(image_shape, sensitivity1, kernel1)
        nse_img2 = _noise_map(image_shape, sensitivity2, kernel2)
        
        for j in range(len(angles)):
            image1_mc = rot_img1[j] + nse_img1
            image2_mc = rot_img2[j] + nse_img2
            spectral_index = np.nan_to_num(
                1.0 / (-np.log(image2_mc / image1_mc) / np.log(6000.0 / 1400.0))
            )
            image_mc = np.clip(spectral_index, 0.0, 1.0 / 0.5)
            detectable = np.zeros_like(image_mc, dtype=int)
            detectable[np.logical_and(image1_mc >= sensitivity1, image2_mc >= sensitivity2)] = 1

            for k, power in enumerate(powers):
                results[k, i, j] = symmetry_index(image_mc, detectable, power=power)

    idx = np.argmin(results, axis=2)
    angles_best = [np.mean(angles[idx_k]) for idx_k in idx]
    deviations = [np.std(angles[idx_k]) for idx_k in idx]

    return (
        (angles_best[0], deviations[0]),
        (angles_best[1], deviations[1]),
        (angles_best[2], deviations[2]),
    )


__all__ = [
    "calculate_symmetry_axes",
    "symmetry_index",
    "calculate_spectral_index_symmetry_axes",
]
