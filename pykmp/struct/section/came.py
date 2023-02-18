import dataclasses

from pykmp._typing import XYZ, Byte, Float, UInt16
from pykmp.ops.autoY import _AutoYSupport
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


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
