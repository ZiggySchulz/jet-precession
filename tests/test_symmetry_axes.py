import numpy as np
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
import pytest

from symmetry_axes import calculate_symmetry_axes, _symmetry_index_squared


def symmetry_component_analysis_ref(angles, surface_brightness, sensitivity, gauss_kernel, nsamples):
    angles = np.asarray(angles)
    image = np.asarray(surface_brightness)
    model = getattr(gauss_kernel, "_model", getattr(gauss_kernel, "model", None))
    if model is None:
        raise AttributeError("Unable to retrieve x_stddev/y_stddev from Gaussian2DKernel")
    x_stddev = model.x_stddev.value
    y_stddev = model.y_stddev.value
    pixels_per_beam = np.pi * x_stddev * y_stddev * (2 * np.log(2))

    rot_img = np.stack(
        [
            ndimage.rotate(np.nan_to_num(image, copy=True, nan=0.0), a, reshape=False, order=3)
            for a in angles
        ]
    )

    result1 = np.zeros((nsamples, len(angles)))
    result2 = np.zeros((nsamples, len(angles)))
    result3 = np.zeros((nsamples, len(angles)))

    for i in range(nsamples):
        nse_img = convolve(
            np.random.normal(
                loc=0.0,
                scale=sensitivity / np.sqrt(10.0) * np.sqrt(pixels_per_beam),
                size=image.shape,
            ),
            gauss_kernel,
        )

        for j in range(len(angles)):
            image_mc = rot_img[j] + nse_img
            detectable = np.zeros_like(image_mc, dtype=np.int_)
            detectable[image_mc >= sensitivity] = 1

            left = np.sum(detectable[:, : image.shape[1] // 2], axis=1)
            right = np.sum(
                detectable[:, (image.shape[1] + 1) // 2 :], axis=1
            )
            leftb = np.sum(
                image_mc[:, : image.shape[1] // 2] * detectable[:, : image.shape[1] // 2],
                axis=1,
            )
            rightb = np.sum(
                image_mc[:, (image.shape[1] + 1) // 2 :] * detectable[:, (image.shape[1] + 1) // 2 :],
                axis=1,
            )
            leftbb = np.sum(
                image_mc[:, : image.shape[1] // 2] ** 2
                * detectable[:, : image.shape[1] // 2],
                axis=1,
            )
            rightbb = np.sum(
                image_mc[:, (image.shape[1] + 1) // 2 :] ** 2
                * detectable[:, (image.shape[1] + 1) // 2 :],
                axis=1,
            )

            result1[i, j] = np.sum((left - right) ** 2 * (left + right)) / np.sum(left + right)
            result2[i, j] = np.sum((leftb - rightb) ** 2 * (leftb + rightb)) / np.sum(leftb + rightb)
            result3[i, j] = np.sum((leftbb - rightbb) ** 2 * (leftbb + rightbb)) / np.sum(leftbb + rightbb)

    idx1 = np.argmin(result1, axis=1)
    idx2 = np.argmin(result2, axis=1)
    idx3 = np.argmin(result3, axis=1)

    return (
        (np.mean(angles[idx1]), np.std(angles[idx1])),
        (np.mean(angles[idx2]), np.std(angles[idx2])),
        (np.mean(angles[idx3]), np.std(angles[idx3])),
    )


def test_symmetry_index_matches_reference_formula():
    rng = np.random.RandomState(123)
    image = rng.normal(size=(6, 6)).astype(float)
    detectable = (image > 0.0).astype(int)

    for power in (0, 1, 2):
        weighted = image**power
        left = np.sum(weighted[:, : image.shape[1] // 2] * detectable[:, : image.shape[1] // 2], axis=1)
        right = np.sum(
            weighted[:, (image.shape[1] + 1) // 2 :] * detectable[:, (image.shape[1] + 1) // 2 :],
            axis=1,
        )
        total = left + right
        expected = float(np.sum((left - right) ** 2 * total) / np.sum(total))

        assert _symmetry_index_squared(image, detectable, power=power) == pytest.approx(expected)


def test_calculate_symmetry_axes_matches_symmetry_component_analysis(monkeypatch):
    angles = np.linspace(-15, 15, 7)
    image = np.clip(np.random.default_rng(0).normal(loc=0.5, scale=0.2, size=(16, 16)), 0.0, 1.0)
    sensitivity = 0.3
    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.2)
    nsamples = 20

    rng = np.random.RandomState(123)

    def deterministic_normal(loc=0.0, scale=1.0, size=None):
        return rng.normal(loc=loc, scale=scale, size=size)

    monkeypatch.setattr(np.random, "normal", deterministic_normal)
    expected = symmetry_component_analysis_ref(
        angles, image, sensitivity, kernel, nsamples
    )

    rng = np.random.RandomState(123)
    monkeypatch.setattr(np.random, "normal", deterministic_normal)
    result = calculate_symmetry_axes(angles, image, sensitivity, kernel, nsamples)

    for actual_pair, expected_pair in zip(result, expected):
        assert actual_pair[0] == pytest.approx(expected_pair[0], rel=1e-12, abs=1e-12)
        assert actual_pair[1] == pytest.approx(expected_pair[1], rel=1e-12, abs=1e-12)
