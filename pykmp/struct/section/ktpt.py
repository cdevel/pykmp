import dataclasses

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.ops.autoY import _AutoYSupport
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class KTPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16


@section_add_attrs(KTPTStruct)
class KTPT(BaseSection, _AutoYSupport):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16

