"""Automatic Height Correction for Y-Axis"""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal, Self

from pykmp._io._wavefront import Wavefront, read_wavefront


def _inplace_pinv(T: np.ndarray) -> np.ndarray:
    """
    Calculate the pseudo-inverse of a matrix in-place.

    Args:
        T: The matrix to invert. 2D or 3D.

    Returns:
        np.ndarray: 2D or 3D matrix.
        if `T` is 3D, the first dimension is the same as `T`.
    """
    if T.ndim >= 3:
        assert T.ndim == 3
        if T.shape[1] != T.shape[2]:
            assert T.shape[0] == T.shape[1]
            T = T.T
    dt = np.linalg.det(T)
    invalid_mask = np.isclose(dt, 0)
    if not np.any(invalid_mask):
        # can be inverted
        return np.linalg.inv(T)
    if np.any(invalid_mask) and T.ndim == 2:
        # apply pseudo inverse
        return np.linalg.pinv(T)

    # in-place inversion
    T_inv_invalid = np.linalg.pinv(T[invalid_mask])
    T_ = T.copy()
    # replace to dummy
    T_[invalid_mask] = np.eye(T_.shape[-1])
    T_inv_ = np.linalg.inv(T_)
    T_inv_[invalid_mask] = T_inv_invalid
    return T_inv_


def autoY(
    obj: Union[Wavefront, str],
    pos: npt.ArrayLike,
    index: Union[int, Iterable[int], None] = None,
    priority: Literal['topmost', 'lowest', 'nearest'] = 'topmost',
    drop_undriveable_face: bool = True,
    inplace_inv: bool = False,
) -> np.ndarray:
    """
    Automatic Height Correction for Y-Axis.

    Args:
        obj (Wavefront, str): Wavefront object or path to .obj file.
        pos (np.ndarray, list, tuple, etc.): 2D or 3D array of the coordinates of the object.
        Last dimension must be 3.
        index (int or iterable, optional): Index of the object.
        priority (str, optional): Priority when there are multiple faces.
        Defaults to 'topmost'.
        drop_undriveable_face (bool, optional): Drop surfaces with normals
        less than or equal to 0.
        inplace_inv (bool, optional): Enable in-place matrix inversion.
        Use if inverse calculation fails. Defaults to False.

    Returns:
        np.ndarray: Array of the same shape as `pos`.
    """
    assert priority in ('topmost', 'lowest', 'nearest'),\
        "`priority` must be 'topmost', 'lowest', or 'nearest'."

    if not isinstance(obj, Wavefront):
        obj = read_wavefront(obj)

    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim not in (1, 2):
        raise ValueError(f'`pos` must be 1D or 2D array. Got {pos.ndim}D array.')
    if pos.shape[-1] != 3:
        raise ValueError(f'Last dimension of `pos` must be 3. Got {pos.shape[-1]}.')

    posdim_pre = pos.ndim

    if posdim_pre == 1:
        pos = pos[None]

    if index is None:
        index = list(range(pos.shape[0]))
    elif isinstance(index, int):
        index = [index]
    elif isinstance(index, Iterable):
        assert all(isinstance(i, int) for i in index), index
        index = list(index)

    pos_ = pos.copy() / 100.
    face_tris_ = obj.facetriangles.copy() / 100.

    # Calculate barycentric coordinate
    # See: https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    r1, r2, r3 = map(
        lambda x: x.squeeze(axis=1), np.split(face_tris_, 3, axis=1))
    # T = \left{
        # \begin{matrix}
        # x_1 - x_3 & x_2 - x_3 \\
        # y_1 - y_3 & y_2 - y_3
        # \end{matrix}
    # \right}
    T1 = np.concatenate(
        [r1[None, :, 0] - r3[None, :, 0], r2[None, :, 0] - r3[None, :, 0]])
    T2 = np.concatenate(
        [r1[None, :, -1] - r3[None, :, -1], r2[None, :, -1] - r3[None, :, -1]])
    T = np.concatenate([T1[None], T2[None]], axis=0)
    try:
        T_inv = np.linalg.inv(T.T)
    except np.linalg.LinAlgError as e:
        if not inplace_inv:
            raise ValueError(
                'Cannot calculate inverse with this object. '
                'If you want to calculate anyway, set `inplace_inv` to True.'
            ) from e
        T_inv = _inplace_pinv(T.T)

    result = []
    for idx in index:
        # target position
        r = pos_[idx]
        # \Lambda = T^{-1} (r - r_3)
        l = np.einsum("Nab, Nb->Na", T_inv, (r - r3)[:, [0, 2]])
        l1 = l[:, 0]
        l2 = l[:, 1]
        l3 = 1 - l1 - l2

        cand, = np.where(
            (
                (0 <= l1) & (l1 <= 1)
                & (0 <= l2) & (l2 <= 1)
                & (0 <= l3) & (l3 <= 1)
            )
        )

        # compute fixed Y
        faceZs = obj.facetriangles[cand].astype(np.float64)
        fcross = np.cross(
            faceZs[:, 1] - faceZs[:, 0], faceZs[:, 2] - faceZs[:, 0])
        if drop_undriveable_face:
            fnorm = fcross / np.linalg.norm(fcross, axis=-1, keepdims=True)
            cand = cand[fnorm[:, 1] > 0]
            if cand.size == 0:
                raise ValueError('Cannot find a face that can be driven.')
        d = -np.einsum("Na, Na->N", fcross, faceZs[:, 2])
        fixedY = -np.divide(
            fcross[:, 0] * r[0] + fcross[:, 2] * r[2] + d, fcross[:, 1]
        ).astype(np.float32)

        if cand.size == 1:
            newpos = np.array([pos[idx, 0], fixedY[0], pos[idx, 2]])
        else:
            n_ = len(fixedY)
            fixedrs = np.stack(
                [np.tile(pos[idx, 0], n_), fixedY, np.tile(pos[idx, 2], n_)]).T
            dist = np.linalg.norm(fixedrs - r, axis=1)

            if priority == 'nearest':
                newpos = fixedrs[np.argmin(dist)]
            elif priority == 'topmost':
                newpos = fixedrs[np.argmax(fixedY)]
            else:
                newpos = fixedrs[np.argmin(fixedY)]

        result.append(newpos.astype(pos_.dtype))

    result = np.stack(result)

    if posdim_pre == 1:
        result = result[0]
    return result


class _AutoYSupport:
    """Add .autoY method to class."""
    def autoY(
        self: Self,
        obj: str,
        index: Union[int, Iterable[int], None] = None,
        priority: Literal['topmost', 'lowest', 'nearest'] = 'topmost',
        drop_undriveable_face: bool = True,
        inplace_inv: bool = False,
        copy: bool = True,
    ) -> Self:
        """
        Automatic Height Correction for Y-Axis.

        Args:
            obj (Wavefront, str): Wavefront object or path to .obj file.
            index (int or iterable, optional): Index of the object. if None, all objects are used.
            priority (str, optional): Priority when there are multiple faces.
            Defaults to 'topmost'.
            drop_undriveable_face (bool, optional): Drop surfaces with normals
            less than or equal to 0.
            inplace_inv (bool, optional): Enable in-place matrix inversion.
            Use if inverse calculation fails. Defaults to False.
            copy (bool, optional): Return a copy of the object. Defaults to True.

        Returns:
            self (KTPT): Returns self if copy is False.
        """
        def fn(cls: Self):
            cls.pos = autoY(
                obj, cls.pos, index, priority,
                drop_undriveable_face, inplace_inv
            )
            return cls
        if copy:
            return fn(self.copy())
        return fn(self)
