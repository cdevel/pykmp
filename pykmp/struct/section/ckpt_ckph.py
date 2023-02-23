import dataclasses

import numpy as np
from typing_extensions import Literal, Self

from pykmp._typing import XY, Byte, Float, Group, Int16
from pykmp.ops import quadrilaterals
from pykmp.plot import _GraphvizSupport
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import inplace_or_copy, section_add_attrs


@dataclasses.dataclass(eq=False)
class CKPTStruct(BaseStruct):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte


@section_add_attrs(CKPTStruct)
class CKPT(BaseSection):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte

    def find_nonconvex_points(self: Self):
        left_right = np.concatenate((self.left, self.right), axis=1)
        loop = np.concatenate((left_right, left_right[:1]), axis=0)
        nonconvex_indices = []
        for i in range(self.entries):
            if not quadrilaterals.is_convex_quadrilateral(
                loop[i], loop[i + 1]
            ):
                nonconvex_indices.append((i, i + 1))
        return nonconvex_indices

    def fix_nonconvex(
        self: Self,
        indices: int | list[int] | None = None,
        moving_factor: float = 0.1,
        max_moving_iteration: int = 20,
        raises: Literal["warn", "raise", "ignore"] = "warn",
        copy: bool = True
    ) -> Self:
        """
        Fix nonconvex quadrilaterals in the checkpoint.

        Args:
            indices (int or list[int] or None): The indices of the points to fix.
            moving_factor(float): The factor to move the points. Default is 0.1.
            max_moving_iteration(int): The maximum number of iterations to move the points.
            Default is 20.
            raises(str): Behavior when cannot fix nonconvex quadrilaterals. Default is "warn".
            only meaningful when indices is None.
        """
        if isinstance(indices, int):
            indices = [indices]
        def fn(cls):
            pos = np.concatenate((cls.left, cls.right), axis=1)
            if indices is None:
                pos = quadrilaterals.fix_all_nonconvex(
                    pos, moving_factor, max_moving_iteration, raises
                )
            else:
                for index in indices:
                    if isinstance(index, int):
                        pi, ni = index, index + 1
                    elif isinstance(index, (list, tuple)):
                        pi, ni = index
                    else:
                        raise TypeError(
                            'index must be int or list/tuple of int.')
                    lpos, rpos = quadrilaterals.fix_nonconvex(
                        pos[pi], pos[ni], moving_factor, max_moving_iteration
                    )
                    pos[pi] = lpos
                    pos[ni] = rpos
            cls.left, cls.right = np.split(pos, 2, axis=1)
            return cls
        return inplace_or_copy(self, copy, fn)


@dataclasses.dataclass(eq=False)
class CKPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@section_add_attrs(CKPHStruct)
class CKPH(BaseSection, _GraphvizSupport):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


def fix_pt_prev_next(ckpt: CKPT, ckph: CKPH):
    """
    Fix the prev and next of the checkpoint (CKPT).
    Note this function will modify the `ckpt` in-place.

    Args:
        ckpt (CKPT): The checkpoint. must have linked CKPH.
        ckph (CKPH): The checkpoint path. must have linked CKPT.
    """
    new_p_n: np.ndarray = None
    for start, length in zip(ckph.start, ckph.length):
        arange = np.arange(start, start + length)
        p_n = np.r_[
            '1,2,0',
            np.r_[255, arange[:-1]], np.r_[arange[1:], 255]
        ]
        if new_p_n is None:
            new_p_n = p_n
        else:
            new_p_n = np.vstack((new_p_n, p_n))
    ckpt.prev, ckpt.next = map(lambda x: x.flatten(), np.hsplit(new_p_n, 2))
