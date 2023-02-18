import dataclasses
import warnings

import numpy as np

from pykmp._io._parser import _BinaryParser as Parser
from pykmp._typing import XYZ, Bit, Byte, Float, Settings, UInt16
from pykmp.ops.autoY import _AutoYSupport
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import CustomFnSpec, section_add_attrs


def _parse_object_id(parser: Parser):
    """
    Parse object_id from GOBJ for extended presence flag.

    Offset reference:
    https://wiki.tockdom.com/wiki/Extended_presence_flags/Technical_Description
    """
    assert parser.is_read_contiuously

    object_id = parser.read_uint16()

    # mask 0x03ff -> objectID
    # mask 0x0c00 -> preserved
    # mask 0x1000 -> lecode_show
    # mask 0xe000 -> defobj_type
    defobj_type = np.uint8((object_id & 0xe000) >> 13) # > 0x2000
    lecode_show = np.bool_((object_id & 0x1000) >> 12) # bit 12
    preserved = np.uint8((object_id & 0x0c00) >> 10) # bit 10-11, unused yet
    object_id = np.uint16(object_id & 0x03ff) # bit 0-9

    return {
        'defobj_type': defobj_type,
        'lecode_show': lecode_show,
        'preserved': preserved,
        'objectID': object_id
    }


def _parse_presence_flag(parser: Parser):
    """
    Parse presence_flag from GOBJ for extended presence flag.

    Offset reference:
    https://wiki.tockdom.com/wiki/Extended_presence_flags/Technical_Description#Presence_flag_.28and_MODE.29
    """
    assert parser.is_read_contiuously

    pf = parser.read_uint16()
    # mask 0xf000
    mode = np.uint8((pf & 0xf000) >> 12)
    # mask 0x0fc0
    parameters = np.uint8((pf & 0x0fc0) >> 6)
    # mask 0x0038
    unused = np.uint8((pf & 0x0038) >> 3)
    # mask 0x0007
    presence_flag = np.uint8(pf & 0x0007)
    pf_3p4p = np.bool_(presence_flag >> 2)
    pf_2p = np.bool_((presence_flag & 0x02) >> 1)
    pf_1p = np.bool_(presence_flag & 0x01)

    return {
        'mode': mode,
        'parameters': parameters,
        'unused': unused,
        'pf_3p4p': pf_3p4p,
        'pf_2p': pf_2p,
        'pf_1p': pf_1p
    }


GOBJ_SPEC = {
        'objectID': CustomFnSpec(
            _parse_object_id, (
                'defobj_type', 'lecode_show',
                'preserved', 'objectID'
            )
        ),
        'pf_1p': CustomFnSpec(
            _parse_presence_flag, (
                'mode', 'parameters', 'unused',
                'pf_3p4p', 'pf_2p', 'pf_1p'
            )
        )
    }


def _to_object_id(
    defobj_type: np.uint8,
    lecode_show: np.bool_,
    preserved: np.uint8,
    objectID: np.uint16
) -> np.uint16:
    defobj_type = np.unpackbits(defobj_type)[-3:] # 3 bits
    lecode_show = np.unpackbits(lecode_show.astype(np.uint8))[-1:] # 1 bit
    preserved = np.unpackbits(preserved)[-2:] # 2 bits
    p16 = np.power(2, np.arange(16))[::-1]
    objID = ((objectID[None] & p16).astype('?').astype('u1')[6:]) # 10 bitss
    originobjID = np.hstack([
        defobj_type, lecode_show, preserved, objID
    ])
    originobjID = np.dot(originobjID, p16).astype(np.uint16)
    return originobjID


@dataclasses.dataclass(eq=False)
class GOBJStruct(BaseStruct):
    defobj_type: Byte
    lecode_show: Bit
    preserved: Byte
    objectID: UInt16
    referenceID: UInt16
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    routeID: UInt16
    settings: UInt16[Settings]
    mode: Byte
    parameters: Byte
    unused: Byte
    pf_3p4p: Bit
    pf_2p: Bit
    pf_1p: Bit

    def tobytes(self) -> bytes:
        b = b''
        data = dataclasses.asdict(self)
        # definition object
        originobjID = _to_object_id(
            data['defobj_type'], data['lecode_show'],
            data['preserved'], data['objectID']
        )
        b += originobjID.newbyteorder('>').tobytes()

        # presence flag
        mode = np.unpackbits(data['mode'])[-4:] # 4 bits
        parameters = np.unpackbits(data['parameters'])[-6:] # 6 bits
        unused = np.unpackbits(data['unused'])[-3:] # 3 bits
        pf = np.stack([data['pf_3p4p'], data['pf_2p'], data['pf_1p']])
        pf = pf.astype(np.uint8)

        p16 = np.power(2, np.arange(16))[::-1]
        presence_flag = np.hstack([mode, parameters, unused, pf])
        presence_flag = np.dot(presence_flag, p16).astype(np.uint16)

        skips = (
            'defobj_type', 'lecode_show',
            'preserved', 'objectID',
            'mode', 'parameters', 'unused', 'pf_3p4p', 'pf_2p', 'pf_1p'
        )

        for k, v in data.items():
            if k in skips:
                continue
            b += v.newbyteorder('>').tobytes()
        b += presence_flag.newbyteorder('>').tobytes()
        return b

    def check(self, raises: bool = True, fix_if_possible: bool = False):
        return super().check(raises, fix_if_possible)


@section_add_attrs(GOBJStruct, custom_fn=GOBJ_SPEC)
class GOBJ(BaseSection, _AutoYSupport):
    defobj_type: Byte
    lecode_show: Bit
    preserved: Byte
    objectID: UInt16
    referenceID: UInt16
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    routeID: UInt16
    settings: UInt16[Settings]
    mode: Byte
    parameters: Byte
    unused: Byte
    pf_3p4p: Bit
    pf_2p: Bit
    pf_1p: Bit

    @property
    def le_mode(self) -> bool:
        return np.any(self.mode > 0).item()

    def _check_struct(self, index: int, data: GOBJStruct):
        super()._check_struct(index, data)
        def _raise_if_over(value, max_value, name):
            if value > max_value:
                raise ValueError(
                    f"The {name} of GOBJ #{index:X} is too large. "
                    f"The maximum value is 0x{max_value:X}, but the "
                    f"value is 0x{value:X}."
                )
        _raise_if_over(data.defobj_type, 0x07, 'defobj_type')
        _raise_if_over(data.preserved, 0x03, 'preserved')
        _raise_if_over(data.objectID, 0x3FF, 'objectID')
        _raise_if_over(data.parameters, 0x3F, 'parameters')
        _raise_if_over(data.mode, 0x0F, 'mode')
        _raise_if_over(data.unused, 0x07, 'unused')

        if data.mode == 0 and (
            data.defobj_type != 0 or
            data.lecode_show != 0 or
            data.preserved != 0
        ):
            # this object is not show in the game
            objID = _to_object_id(
                data.defobj_type, data.lecode_show,
                data.preserved, data.objectID
            )
            warnings.warn(
                f"The object (ID: 0x{objID:04X}) of GOBJ #{index:X} is not show "
                "in the game. To show it, set mode to 1 or higher."
            )
        elif data.mode == 1:
            # defobj_type supports 0, 1, 2, 3
            if data.defobj_type not in [0, 1, 2, 3]:
                warnings.warn(
                    f"For defobj_type of GOBJ #{index:X}, [0, 1, 2, 3] "
                    f"are supported, but {data.defobj_type} is given."
                    " This object may not show in the game."
                )
            elif data.defobj_type in [1, 2, 3]:
                def _maybe_warn(vec, name):
                    if vec.any():
                        warnings.warn(
                            f"For defobj_type={data.defobj_type} of "
                            f"GOBJ #{index:X}, all of {name} should be 0. "
                            "pykmp will set it to 0."
                        )
                        vec = np.zeros_like(vec)
                    return vec
                data.pos = _maybe_warn(data.pos, 'pos')
                data.rot = _maybe_warn(data.rot, 'rot')
                data.scale = _maybe_warn(data.scale, 'scale')
                if data.routeID != 0xFFFF:
                    warnings.warn(
                        f"For defobj_type={data.defobj_type} of GOBJ #{index:X}, "
                        "routeID should be 0xFFFF. pykmp will set it to 0xFFFF."
                    )
                    data.routeID = np.uint16(0xFFFF)
            # predefined condition
            # XXX: 0x2000 -> ?
            if (
                (0 < data.referenceID <= 0x1FFF)
                and (
                    not (0x1000 <= data.referenceID <= 0x17FF)
                    and not (0x1e00 <= data.referenceID <= 0x1e7f)
                    and not (0x1f00 <= data.referenceID <= 0x1fff)
                )
            ):
                warnings.warn(
                    f"Unknown referenceID (0x{data.referenceID:04X})"
                    f" of GOBJ #{index:X}. Value should be 0 or "
                    "0x1000-0x17FF (Hard coded conditions)"
                    ", 0x1e00-0x1e7f (Engine Selection) "
                    "or 0x1f00-0x1fff (Random Scenarios)."
                )
