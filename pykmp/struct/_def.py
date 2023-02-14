import dataclasses
from collections import defaultdict
from types import EllipsisType
from typing import Iterable, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal, Self

from pykmp._typing import (NXYZ, XY, XYZ, Bit, Byte, Float, Group, Int16,
                           NScalar, Settings, UInt16, UInt32, get_colname,
                           get_dtype_and_size)
from pykmp.ops import quadrilaterals
from pykmp.ops.autoY import _AutoYSupport
from pykmp.plot import _GraphvizSupport
from pykmp.struct._base_struct import (GOBJ_SPEC, POTI_SPEC, STGI_SPEC,
                                       BaseSection, BaseStruct,
                                       section_add_attrs)

_KMP_CLASSES = []
_LEX_CLASSES = []


def _kmp_register(cls):
    _KMP_CLASSES.append((cls.__name__, cls))
    return cls


def _lex_register(cls):
    _LEX_CLASSES.append((cls.__name__, cls))
    return cls


def _inplace_or_copy(cls, copy: bool, func):
    if copy:
        return func(cls.copy())
    return func(cls)


@dataclasses.dataclass(eq=False)
class KTPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16


@_kmp_register
@section_add_attrs(KTPTStruct)
class KTPT(BaseSection, _AutoYSupport):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16


@dataclasses.dataclass(eq=False)
class ENPTStruct(BaseStruct):
    pos: Float[XYZ]
    widthfactor: Float
    property1: UInt16
    property2: Byte
    property3: Byte


@_kmp_register
@section_add_attrs(ENPTStruct)
class ENPT(BaseSection, _AutoYSupport):
    pos: Float[XYZ]
    widthfactor: Float
    property1: UInt16
    property2: Byte
    property3: Byte


@dataclasses.dataclass(eq=False)
class ENPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    dispatch: Int16


@_kmp_register
@section_add_attrs(ENPHStruct)
class ENPH(BaseSection, _GraphvizSupport):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    dispatch: Int16


@dataclasses.dataclass(eq=False)
class ITPTStruct(BaseStruct, _AutoYSupport):
    pos: Float[XYZ]
    widthfactor: Float
    property1: UInt16
    property2: UInt16


@_kmp_register
@section_add_attrs(ITPTStruct)
class ITPT(BaseSection):
    pos: Float[XYZ]
    widthfactor: Float
    property1: UInt16
    property2: UInt16


@dataclasses.dataclass(eq=False)
class ITPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@_kmp_register
@section_add_attrs(ITPHStruct)
class ITPH(BaseSection, _GraphvizSupport):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@dataclasses.dataclass(eq=False)
class CKPTStruct(BaseStruct):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte


@_kmp_register
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
        indices: Union[int, list[int], None] = None,
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
        return _inplace_or_copy(self, copy, fn)


@dataclasses.dataclass(eq=False)
class CKPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@_kmp_register
@section_add_attrs(CKPHStruct)
class CKPH(BaseSection, _GraphvizSupport):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@dataclasses.dataclass(eq=False)
class GOBJStruct(BaseStruct):
    defobj_cond: Byte
    defobj_enable: Bit
    defobj_preserved: Byte
    objectID: UInt16
    referenceID: UInt16
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    routeID: UInt16
    settings: UInt16[Settings]
    mode: Byte
    parameters: Byte
    dummy: Byte
    presence_flag: Byte

    def tobytes(self) -> bytes:
        b = b''
        data = dataclasses.asdict(self)
        # definition object
        defobj_cond = np.unpackbits(data['defobj_cond'])[-3:] # 3 bits
        defobj_enable = np.unpackbits(
            data['defobj_enable'].astype(np.uint8))[-1:] # 1 bit
        defobj_preserved = np.unpackbits(data['defobj_preserved'])[-2:] # 2 bits
        p16 = np.power(2, np.arange(16))[::-1]
        objID = ((data['objectID'][None] & p16)
                 .astype('?').astype('u1')[6:]) # 10 bits
        originobjID = np.hstack([
            defobj_cond, defobj_enable, defobj_preserved, objID
        ])
        originobjID = np.dot(originobjID, p16).astype(np.uint16)
        b += originobjID.newbyteorder('>').tobytes()

        # presence flag
        mode = np.unpackbits(data['mode'])[-4:] # 4 bits
        parameters = np.unpackbits(data['parameters'])[-6:] # 6 bits
        dummy = np.unpackbits(data['dummy'])[-3:] # 3 bits
        pf = np.unpackbits(data['presence_flag'])[-3:] # 3 bits

        presence_flag = np.hstack([mode, parameters, dummy, pf])
        presence_flag = np.dot(presence_flag, p16).astype(np.uint16)

        skips = (
            'defobj_cond', 'defobj_enable',
            'defobj_preserved', 'objectID',
            'mode', 'parameters', 'dummy', 'presence_flag'
        )

        for k, v in data.items():
            if k in skips:
                continue
            b += v.newbyteorder('>').tobytes()
        b += presence_flag.newbyteorder('>').tobytes()
        return b


@_kmp_register
@section_add_attrs(GOBJStruct, custom_fn=GOBJ_SPEC)
class GOBJ(BaseSection, _AutoYSupport):
    defobj_cond: Byte
    defobj_enable: Bit
    defobj_preserved: Byte
    objectID: UInt16
    referenceID: UInt16
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    routeID: UInt16
    settings: UInt16[Settings]
    mode: Byte
    parameters: Byte
    dummy: Byte
    presence_flag: Byte


@dataclasses.dataclass(eq=False)
class PotiPoints:
    index: Union[int, slice, Iterable[int], EllipsisType]
    pos: Float[XYZ]
    property1: UInt16
    property2: UInt16


@dataclasses.dataclass(eq=False)
class POTIStruct(BaseStruct):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    property1: UInt16[NScalar]
    property2: UInt16[NScalar]

    def __len__(self: Self):
        return self.numpoints

    def tobytes(self: Self) -> bytes:
        b = self.numpoints.newbyteorder('>').tobytes()
        b += self.smooth.newbyteorder('>').tobytes()
        b += self.forward_backward.newbyteorder('>').tobytes()

        for i in range(self.numpoints):
            b += self.pos[i].byteswap().tobytes()
            b += self.property1[i].newbyteorder('>').tobytes()
            b += self.property2[i].newbyteorder('>').tobytes()

        return b

    def __getitem__(self: Self, index: int | slice | Iterable[int]):
        if (isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)):
            raise TypeError("Index must be int, slice, or iterable of ints")
        return PotiPoints(
            index,
            self.pos[index],
            self.property1[index],
            self.property2[index],
        )

    def __setitem__(
        self,
        index: int | slice | Iterable[int] | EllipsisType,
        value: PotiPoints
    ):
        if (isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)):
            raise TypeError("Index must be int, slice, or iterable of ints")
        self.pos[index] = value.pos
        self.property1[index] = value.property1
        self.property2[index] = value.property2


@_kmp_register
@section_add_attrs(POTIStruct, custom_fn=POTI_SPEC)
class POTI(BaseSection):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    property1: UInt16[NScalar]
    property2: UInt16[NScalar]


@dataclasses.dataclass(eq=False)
class AREAStruct(BaseStruct):
    shape: Byte
    type: Byte
    cameindex: Byte
    priority: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    setting1: UInt16
    setting2: UInt16
    potiID: Byte
    enptID: Byte
    padding: UInt16


@_kmp_register
@section_add_attrs(AREAStruct)
class AREA(BaseSection, _AutoYSupport):
    shape: Byte
    type: Byte
    cameindex: Byte
    priority: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    setting1: UInt16
    setting2: UInt16
    potiID: Byte
    enptID: Byte
    padding: UInt16


@dataclasses.dataclass(eq=False)
class CAMEStruct(BaseStruct):
    type: Byte
    next: Byte
    unknown: Byte
    route: Byte
    vcame: UInt16
    vzoom: UInt16
    vpt: UInt16
    unknown2: Byte
    unknown3: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    zoombeg: Float
    zoomend: Float
    viewbegpos: Float[XYZ]
    viewendpos: Float[XYZ]
    time: Float


@_kmp_register
@section_add_attrs(CAMEStruct)
class CAME(BaseSection, _AutoYSupport):
    type: Byte
    next: Byte
    unknown: Byte
    route: Byte
    vcame: UInt16
    vzoom: UInt16
    vpt: UInt16
    unknown2: Byte
    unknown3: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    zoombeg: Float
    zoomend: Float
    viewbegpos: Float[XYZ]
    viewendpos: Float[XYZ]
    time: Float


@dataclasses.dataclass(eq=False)
class JGPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    unknown: UInt16
    range: Int16


@_kmp_register
@section_add_attrs(JGPTStruct)
class JGPT(BaseSection, _AutoYSupport):
    pos: Float[XYZ]
    rot: Float[XYZ]
    unknown: UInt16
    range: Int16


@dataclasses.dataclass(eq=False)
class CNPTStruct(BaseStruct):
    destination: Float[XYZ]
    direction: Float[XYZ]
    positionID: UInt16
    effect: Int16


@_kmp_register
@section_add_attrs(CNPTStruct)
class CNPT(BaseSection):
    destination: Float[XYZ]
    direction: Float[XYZ]
    positionID: UInt16
    effect: Int16


@dataclasses.dataclass(eq=False)
class MSPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    entryID: UInt16
    unknown: Int16


@_kmp_register
@section_add_attrs(MSPTStruct)
class MSPT(BaseSection, _AutoYSupport):
    pos: Float[XYZ]
    rot: Float[XYZ]
    entryID: UInt16
    unknown: Int16


@dataclasses.dataclass(eq=False)
class STGIStruct(BaseStruct):
    lap: Byte
    poleposition: Byte
    distancetype: Byte
    flareflash: Byte
    flarecolor: UInt32
    transparency: Byte
    padding: Byte
    speedfactor: Float


@_kmp_register
@section_add_attrs(STGIStruct, indexing=False, custom_fn=STGI_SPEC)
class STGI(BaseSection):
    lap: Byte
    poleposition: Byte
    distancetype: Byte
    flareflash: Byte
    flarecolor: UInt32
    transparency: Byte
    padding: Byte
    speedfactor: Float

    def tobytes(self):
        return super().tobytes()[:-2]


# @dataclasses.dataclass(eq=False)
# class FEATStruct(BaseStruct):
#     ...


# @dataclasses.dataclass(eq=False)
# class SET1truct(BaseStruct):
#     item_pos_factor: Float[XYZ]
#     start_item: Byte
#     padding: Byte[XYZ]  # this XYZ is no meaning


# @_lex_register
# @section_add_attrs(SET1truct, indexing=False)
# class SET1(BaseSection):
#     item_pos_factor: Float[XYZ]
#     start_item: Byte
#     padding: Byte[XYZ]


# @dataclasses.dataclass(eq=False)
# class CANNProperty:
#     speed: Float
#     height: Float
#     deceleration_factor: Float
#     end_eceleration: Float


# @dataclasses.dataclass(eq=False)
# class CANNStruct(BaseStruct):
#     numtypes: UInt16
#     speed: Float[NScalar]
#     height: Float[NScalar]
#     deceleration_factor: Float[NScalar]
#     end_eceleration: Float[NScalar]


# @_lex_register
# @section_add_attrs(CANNStruct)
# class CANN(BaseSection):
#     numtypes: UInt16
#     speed: Float[NScalar]
#     height: Float[NScalar]
#     deceleration_factor: Float[NScalar]
#     end_eceleration: Float[NScalar]


del _kmp_register


def _section_classes(type_: str = 'kmp') -> list[tuple[str, type]]:
    if type_ == 'kmp':
        return _KMP_CLASSES
    raise NotImplementedError('Currently only kmp is supported.')

