"""Smoothing Point-to-Points."""

from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate
from typing_extensions import Literal

_SUPPORTED_INTERPOLATION_METHODS = {}


def spline_interpolate(
    coordinates: npt.ArrayLike,
    factor: Union[int, float, np.integer, np.floating]
) -> np.ndarray:
    """
    Spline interpolation
    """
    coordinates = np.asarray(coordinates)
    assert coordinates.ndim == 2, "coordinates must be 2D array"
    assert coordinates.shape[1] == 3, "axis 1 must have 3 elements"
    assert coordinates.shape[0] > 2, "coordinates must have more than 2 points"
    if not isinstance(factor, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"factor must be int or float, not {type(factor)}"
        )

    coordinates = coordinates.T

    m = len(coordinates[0])
    npts = round(m * factor)

    if npts > 255:
        raise ValueError(
            "Cannot interpolate coordinates with more than 255 points. "
            f"({npts} > 255)"
        )

    # create knots vector
    tck, _ = interpolate.splprep(coordinates, s=m + np.sqrt(m), k=4)
    u_factor = np.linspace(0, 1, round(m * factor))

    # fitting
    smoothed = np.vstack(interpolate.splev(u_factor, tck)).T
    return smoothed


def catmull_clark_subdivision(
    coordinates: npt.ArrayLike,
    factor: Union[int, np.integer]
) -> np.ndarray:
    """
    Catmull-Clark subdivision
    """
    coordinates = np.asarray(coordinates)
    assert coordinates.ndim == 2, "coordinates must be 2D array"
    assert coordinates.shape[1] == 3, "axis 1 must have 3 elements"
    assert coordinates.shape[0] > 2, "coordinates must have more than 2 points"
    if not isinstance(factor, (int, np.integer)):
        raise TypeError(
            f"factor must be scalar, not {type(factor)}"
        )

    size = 2 ** factor * len(coordinates) - 2 ** factor + 1

    if size > 255:
        raise ValueError(
            "Cannot interpolate coordinates with more than 255 points. "
            f"({size} > 255)"
        )

    smoothed = coordinates.copy()

    for _ in range(factor - 1):
        size = len(smoothed)
        smoothed_ = np.zeros((size * 2 - 1, 3))
        smoothed_[::2] = smoothed
        smoothed_[1::2] = ((smoothed[:-1] + smoothed[1:]) / 2)
        smoothed_[2:-2:2] = (
            smoothed_[1:-2:2] + smoothed_[3::2] + smoothed_[2:-2:2]
        ) / 3
        smoothed = smoothed_
    return smoothed


_SUPPORTED_INTERPOLATION_METHODS['spline'] = spline_interpolate
_SUPPORTED_INTERPOLATION_METHODS['catmull-clark'] = catmull_clark_subdivision


def _fill_settings(
    new_setting: npt.ArrayLike,
    how: str
):
    prev_value = new_setting[0]
    iterator = range(len(new_setting))
    if how == 'right_fill':
        iterator = reversed(iterator)
        prev_value = new_setting[-1]
    for i, idx in enumerate(iterator):
        if i > 0 and np.any(new_setting[idx] != prev_value):
            prev_value = new_setting[idx]
        new_setting[idx] = prev_value

    return new_setting


def interpolate_impl(
    coordinates: np.ndarray,
    settings: Union[np.ndarray, tuple[np.ndarray, ...], list[np.ndarray]],
    factor: Union[int, float, np.integer, np.floating],
    method: Literal['spline', 'catmull-clark'] = 'spline',
    point_settigns: Literal['left_fill', 'right_fill', 'zeros'] = 'zeros'
) -> tuple[np.ndarray, Union[list[np.ndarray], np.ndarray]]:
    """
    Smoothing Point-to-Points, for ENPT, ITPT, POTI.

    Args:
        coordinates (np.ndarray): Coordinates of the points (X, Y, Z).
        settings (np.ndarray): Settings of the points.
        factor (int or float): Number of points to interpolate.
        method (str): Interpolation method. Must be 'spline' or 'catmull-clark'. Defaults to 'spline'.
        point_settigns (str): How to handle the settings of the points.
        Available options are 'left_fill', 'right_fill', 'zeros'. Defaults to 'zeros'.

    Returns:
        tuple[np.ndarray, np.ndarray]: Interpolated coordinates and settings.
    """
    try:
        method_fn = _SUPPORTED_INTERPOLATION_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unsupported interpolation method: {method}"
        ) from None
    smoothed = method_fn(coordinates, factor)

    update_idx = []
    for c in coordinates:
        norm = np.linalg.norm(smoothed - c[None], axis=1)
        update_idx.append(np.argmin(norm))

    # tile = np.tile(coordinates, (len(smoothed), 1, 1))
    # norm = np.linalg.norm(tile - smoothed[:,None], axis=2)
    # update_idx = np.argmin(norm, axis=1)

    if isinstance(settings, np.ndarray):
        settings = (settings,)

    ret_settings = []
    for setting in settings:
        if setting.ndim == 1:
            newsettings = np.zeros((len(smoothed),), dtype=setting.dtype)
        else:
            newsettings = np.zeros(
                (len(smoothed), len(setting[1])), dtype=setting.dtype)
        newsettings[update_idx] = setting.copy()

        if point_settigns == 'zeros':
            ret_settings.append(newsettings)
            continue

        newsettings = _fill_settings(newsettings, point_settigns)
        ret_settings.append(newsettings)

    if len(ret_settings) == 1:
        return smoothed, ret_settings[0]
    return smoothed, ret_settings
