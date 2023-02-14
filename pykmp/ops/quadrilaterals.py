"""Convex Quadrilaterals operations."""
import re
import warnings

import numpy as np
from typing_extensions import Literal


def is_convex_quadrilateral(
    pts1: np.ndarray, pts2: np.ndarray
) -> bool:
    """
    Check if two quadrilaterals are convex quadrilaterals or not.

    Args:
        pts1, pts2(np.ndarray): (Lx, Lz, Rx, Rz) array of 2D points.

    Returns:
        bool: True if convex, False if not.
    """
    assert pts1.size == pts2.size == 4, "pts1 and pts2 must be 1x4 array."

    # L->R->R->L
    pos = np.vstack(
        [pts1.reshape(2,2), pts2.reshape(2,2)[::-1]]
    )
    pos[:, 1] *= -1

    # check no duplicate points
    if pos.size != np.unique(pos).size:
        return False

    # algorithm: The sum of the green angles should be 2π
    # https://wiki.tockdom.com/wiki/File:Caron_is_cq.jpg
    vector = np.vstack(
        [pos - np.roll(pos, 1, 0), np.roll(pos, 1, 0) - pos]
    )
    vector = np.hstack([vector[:4], np.roll(vector[4:], 1, 0)])
    # cos(theta)
    cos = np.sum(vector[:, :2] * vector[:, 2:], axis=1) / (
        np.linalg.norm(vector[:, :2], axis=1) * np.linalg.norm(vector[:, 2:], axis=1)
    )
    theta = 180 - np.rad2deg(np.arccos(cos))
    res = np.where(theta >= 360, theta - 360, theta)
    res = np.round(res.sum(), 2)

    return res.astype(int) == 360


def fix_nonconvex(
    pts1: np.ndarray,
    pts2: np.ndarray,
    moving_factor: float = 0.1,
    max_moving_iteration: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fix nonconvex quadrilaterals given by pts1 and pts2.
    If pts1 and pts2 are already convex quadrilaterals, return them.

    Args:
        pts1, pts2(np.ndarray): (Lx, Lz, Rx, Rz) array of 2D points.
        moving_factor(float): The factor to move the points. Default is 0.1.
        max_moving_iteration(int): The maximum number of iterations to move the points.
        Default is 20.

    Returns:
        tuple[np.ndarray, np.ndarray]: Fixed pts1 and pts2.

    Raises:
        ValueError: If the fixed points are still nonconvex quadrilaterals
        after end of iteration.
    """
    if is_convex_quadrilateral(pts1, pts2):
        return pts1, pts2

    if np.all(pts1 == pts2):
        raise ValueError(
            "Cannot fix nonconvex quadrilaterals with same points.")

    # L->R->R->L
    pos = np.vstack([pts1.reshape(2,2), pts2.reshape(2,2)[::-1]])
    # Convert to vector, 8 x 2
    vector = np.vstack([np.roll(pos, 2) - pos, np.roll(pos, -2) - pos])

    # compute bisector
    uv, vv = np.split(vector, 2, axis=0)
    a, b = np.linalg.norm(uv, axis=1), np.linalg.norm(vv, axis=1)
    a_apb, b_apb = a / (a + b), b / (a + b)
    uvec_i2 = np.tile(b_apb, (2, 1)).T * uv
    vvec_i2 = np.tile(a_apb, (2, 1)).T * vv
    vector_i2 = uvec_i2 + vvec_i2

    # find the clockwise point
    res = np.rad2deg(np.arctan2(vector.T[1], vector.T[0]))
    res = 180 - (res - np.roll(res, 4))[:4]
    res = np.where(res >= 360, res - 360, res)
    f_index = np.where(res > 180)[0][0]
    target_index = [f_index, f_index + 1] if f_index != 3 else [f_index, 0]

    retvals = None
    for i in range(1, max_moving_iteration):
        k = pos.copy()
        k[target_index] = (
            k[target_index] + (vector_i2[target_index] * moving_factor * i))
        p0, p1 = np.vsplit(k, 2)
        p0 = p0.flatten()
        p1 = p1[::-1].flatten()
        if is_convex_quadrilateral(p0, p1):
            retvals = p0, p1
            break
    else:
        raise ValueError(
            "Cannot fix nonconvex quadrilaterals. "
            "try to increase moving_factor or max_moving_iteration."
        )

    return retvals


# alias
fix_nonconvex_quadrilateral = fix_nonconvex


def fix_all_nonconvex(
    ckpt_positions: np.ndarray,
    moving_factor: float = 0.1,
    moving_iteration: int = 20,
    raises: Literal["warn", "raise", "ignore"] = "warn",
) -> np.ndarray:
    """
    Fix ALL nonconvex quadrilaterals in the checkpoint.

    Args:
        checkpoint_positions(np.ndarray): (N, 4) array of 2D points.
        moving_factor(float): The factor to move the points. Default is 0.1.
        max_moving_iteration(int): The maximum number of iterations to move the points.
        Default is 20.
        raises(str): Behavior when cannot fix nonconvex quadrilaterals. Default is "warn".

    Returns:
        np.ndarray: Fixed checkpoint positions. same shape as checkpoint_positions.

    Raises:
        ValueError: If the fixed points are still nonconvex quadrilaterals
        after end of iteration. Only raised when `raises` is "raise".
    """

    assert ckpt_positions.ndim == 2, "ckpt_positions must be 2D array."
    assert ckpt_positions.shape[1] == 4, "ckpt_positions must be Nx4 array."
    assert ckpt_positions.shape[0] > 2, "ckpt_positions must have more than 2 checkpoints."
    assert raises in ["warn", "raise", "ignore"], "raises must be one of 'warn', 'raise', 'ignore'."

    result = ckpt_positions.copy()
    for idx in range(result.shape[0]):
        if idx == result.shape[0] - 1:
            lidx, ridx = idx, 0
        else:
            lidx, ridx = idx, idx + 1

        pts1, pts2 = result[idx], result[ridx]

        try:
            npts1, npts2 = fix_nonconvex(
                pts1, pts2, moving_factor, moving_iteration)
        except ValueError as e:
            is_convex_error = re.search(
                "Cannot fix nonconvex quadrilaterals", str(e))
            if is_convex_error:
                if all(pts1 == pts2):
                    msg = (
                        "Cannot fix nonconvex quadrilaterals with same points"
                        f" (checkpoint #{lidx} and #{ridx}).")
                else:
                    msg = (
                        "Cannot fix nonconvex quadrilaterals "
                        f" (checkpoint #{lidx} and #{ridx}). "
                        "try to increase moving_factor or max_moving_iteration.")
                if raises == "warn":
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        warnings.warn(msg, RuntimeWarning)
                    continue
                elif raises == "ignore":
                    continue
                else:
                    raise ValueError(msg) from e
            raise e
        result[idx] = npts1
        result[idx + 1] = npts2

    return result


def test(mode: str = 'warn'):
    ncq1 = np.array([21.13, -8.89, 33.1, -7.91])
    ncq2 = np.array([25.29, -2.39, 36.9, -8.54])
    a = np.vstack([ncq1, ncq1, ncq2, ncq1])
    return fix_all_nonconvex(a, raises=mode)
