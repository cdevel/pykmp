import dataclasses
from types import EllipsisType
from typing import Iterable, Union

import numpy as np

from pykmp._typing import (NXYZ, XY, XYZ, Bit, Byte, Float, Group, Int16,
                           NScalar, Settings, UInt16, UInt32)
from pykmp.struct._base_struct import (GOBJ_SPEC, POTI_SPEC, STGI_SPEC,
                                       BaseSection, BaseStruct,
                                       section_add_attrs)


@dataclasses.dataclass(eq=False)
class KTPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16


@section_add_attrs(KTPTStruct)
class KTPT(BaseSection):
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


@section_add_attrs(ENPTStruct)
class ENPT(BaseSection):
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


@section_add_attrs(ENPHStruct)
class ENPH(BaseSection):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    dispatch: Int16


@dataclasses.dataclass(eq=False)
class ITPTStruct(BaseStruct):
    pos: Float[XYZ]
    wfactor: Float
    property1: UInt16
    property2: UInt16


@section_add_attrs(ITPTStruct)
class ITPT(BaseSection):
    pos: Float[XYZ]
    wfactor: Float
    property1: UInt16
    property2: UInt16


@dataclasses.dataclass(eq=False)
class ITPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@section_add_attrs(ITPHStruct)
class ITPH(BaseSection):
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


@section_add_attrs(CKPTStruct)
class CKPT(BaseSection):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte


@dataclasses.dataclass(eq=False)
class CKPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@section_add_attrs(CKPHStruct)
class CKPH(BaseSection):
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


@section_add_attrs(GOBJStruct, custom_fn=GOBJ_SPEC)
class GOBJ(BaseSection):
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
    value: UInt16
    unknown: UInt16


@dataclasses.dataclass(eq=False)
class POTIStruct(BaseStruct):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    value: UInt16[NScalar]
    unknown: UInt16[NScalar]

    def tobytes(self) -> bytes:
        b = self.numpoints.newbyteorder('>').tobytes()
        b += self.smooth.newbyteorder('>').tobytes()
        b += self.forward_backward.newbyteorder('>').tobytes()

        for i in range(self.numpoints):
            b += self.pos[i].byteswap().tobytes()
            b += self.value[i].newbyteorder('>').tobytes()
            b += self.unknown[i].newbyteorder('>').tobytes()

        return b

    def __getitem__(self, index: Union[int, slice, Iterable[int]]):
        if (isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)):
            raise TypeError("Index must be int, slice, or iterable of ints")
        return PotiPoints(
            index,
            self.pos[index],
            self.value[index],
            self.unknown[index],
        )

    def __setitem__(
        self,
        index: Union[int, slice, Iterable[int], EllipsisType],
        value: PotiPoints
    ):
        if (isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)):
            raise TypeError("Index must be int, slice, or iterable of ints")
        self.pos[index] = value.pos
        self.value[index] = value.value
        self.unknown[index] = value.unknown


@section_add_attrs(POTIStruct, custom_fn=POTI_SPEC)
class POTI(BaseSection):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    value: UInt16[NScalar]
    unknown: UInt16[NScalar]


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


@section_add_attrs(AREAStruct)
class AREA(BaseSection):
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


@section_add_attrs(CAMEStruct)
class CAME(BaseSection):
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


@section_add_attrs(JGPTStruct)
class JGPT(BaseSection):
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


@section_add_attrs(MSPTStruct)
class MSPT(BaseSection):
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


@section_add_attrs(
    STGIStruct, indexing=False, custom_fn=STGI_SPEC
)
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


def _section_classes():
    defined_globals = globals()
    return [
        (n, cls) for n, cls in defined_globals.items() \
            if hasattr(cls, '__struct__')
    ]
