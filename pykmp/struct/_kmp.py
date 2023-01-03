import dataclasses
import itertools
import os
import pathlib
import tempfile
import typing
from typing import Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from pykmp._io import _parser
from pykmp.struct import _def


@dataclasses.dataclass
class KMP:
    """KMP file class"""
    header: str
    byte_length: np.uint32
    total_sections: np.uint16
    header_length: np.uint16
    version_number: np.uint32
    section_offsets: npt.NDArray[np.uint32]
    KTPT: _def.KTPT
    ENPT: _def.ENPT
    ENPH: _def.ENPH
    ITPT: _def.ITPT
    ITPH: _def.ITPH
    CKPT: _def.CKPT
    CKPH: _def.CKPH
    GOBJ: _def.GOBJ
    POTI: _def.POTI
    AREA: _def.AREA
    CAME: _def.CAME
    JGPT: _def.JGPT
    CNPT: _def.CNPT
    MSPT: _def.MSPT
    STGI: _def.STGI

    def write(self, path: str):
        # collect section bytes
        sectiondata = b""
        section_size = []
        for name, _ in _def._section_classes():
            data = getattr(self, name).tobytes()
            sectiondata += data
            section_size.append(len(data))

        # write headers
        # header
        b = self.header.encode('utf-8')
        # byte length
        b += np.uint32(
            len(sectiondata) + self.header_length
        ).newbyteorder('>').tobytes()
        # total sections
        b += self.total_sections.newbyteorder('>').tobytes()
        # header length
        b += self.header_length.newbyteorder('>').tobytes()
        # version number
        b += self.version_number.newbyteorder('>').tobytes()
        # section offsets
        b += np.uint32(
            [0] + list(itertools.accumulate(section_size))[:-1]
        ).byteswap().tobytes()

        with open(path, 'wb') as f:
            f.write(b)
            f.write(sectiondata)

    def apply_wiimm(self: Self, command: str, inplace: bool = False):
        raise NotImplementedError


def read_kmp(
    path: Union[str, pathlib.Path, os.PathLike],
):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    parser = _parser._BinaryParser(path.read_bytes())

    kwgs = {}
    kwgs['header'] = parser.header
    with parser.read_contiuously(start=0x04, back=True):
        kwgs['byte_length'] = parser.read_uint32()
        kwgs['total_sections'] = parser.read_uint16()
        kwgs['header_length'] = parser.read_uint16()
        kwgs['version_number'] = parser.read_uint32()
        kwgs['section_offsets'] = parser.read_uint32(
            int(kwgs['total_sections'])
        )
    section_cls = _def._section_classes()
    kwgs['section_offsets'] = kwgs['section_offsets'][:len(section_cls)]
    for i, (name, cls) in enumerate(section_cls):
        kwgs[name] = cls(
            parser,
            offset=int(kwgs['header_length'] + kwgs['section_offsets'][i])
        )

    del parser
    return KMP(**kwgs)
