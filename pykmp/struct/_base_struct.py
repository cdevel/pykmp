import dataclasses
import warnings
from struct import pack as fpack
from types import EllipsisType
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

from typing_extensions import Self

import pykmp._typing as t
from pykmp._io._parser import _BinaryParser as Parser
from pykmp.struct import pandas_utils
from pykmp.struct.descriptor import DataDescriptor, _ListMax255, _SpecialName


def _parse_object_id(parser: Parser):
    """Parse object_id from GOBJ for extended presence flag."""
    assert parser.is_read_contiuously

    object_id = parser.read_uint16()
    defobj_cond = np.uint8(object_id >> 13)
    defobj_enable = np.bool_(object_id >> 12)
    defobj_preserved = np.uint8(object_id >> 10)
    object_id = np.uint16(object_id & 0x03ff)
    return {
        'defobj_cond': defobj_cond,
        'defobj_enable': defobj_enable,
        'defobj_preserved': defobj_preserved,
        'objectID': object_id
    }


def _parse_presence_flag(parser: Parser):
    """Parse presence_flag from GOBJ for extended presence flag."""
    assert parser.is_read_contiuously

    pf = parser.read_uint16()
    # mask 0xf000
    mode = np.uint8(pf >> 12)
    # mask 0x0fc0
    parameters = np.uint8(pf >> 6)
    # mask 0x0038
    dummy = np.uint8(pf >> 3)
    presence_flag = np.uint8(pf & 0x0007)

    return {
        'mode': mode,
        'parameters': parameters,
        'dummy': dummy,
        'presence_flag': presence_flag
    }


def _parse_poti_points(parser: Parser, numpoints: int):
    """Parse POTI Route points from POTI."""
    assert parser.is_read_contiuously

    pos = []
    property1 = []
    property2 = []

    for _ in range(numpoints):
        pos.append(parser.read_float32(3))
        property1.append(parser.read_uint16()[None])
        property2.append(parser.read_uint16()[None])

    return {
        'pos': np.array(pos, dtype=np.float32),
        'property1': np.array(property1, dtype=np.uint16),
        'property2': np.array(property2, dtype=np.uint16)
    }


def _parse_stgi_speedfactor(parser: Parser):
    """Parse STGI Speed factor from STGI for max speed modifier"""
    assert parser.is_read_contiuously

    padding = parser.read_uint8()
    speedfactor = parser.read_number('>f4', 1, size=2, fillbyte=b'\x00\x00')

    return {
        'padding': padding,
        'speedfactor': speedfactor
    }


@dataclasses.dataclass
class _CustomFnSpec:
    """
    Special function specification for custom parsing.
    Private use only. You don't need to use this class.

    Args:
        fn (callable): A function to parse. first argument must be a parser.
        ret_keys (list): A list of keys to return.
        additional_args (list): A list of additional argument "name" for `fn`.
    """
    fn: Callable[[Parser, Any], Dict[str, Any]]
    ret_keys: Sequence[str]
    additional_args: Sequence[str] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        self._args = {}

    def set_args(self: Self, keydict: Dict[str, Any]):
        """
        Set additional arguments for the function.
        If `additional_args` is empty, this method does nothing.

        Args:
            keydict (dict): A dict of key and value.
            args should be in the dict.
        """
        if not self.additional_args or not keydict:
            return
        self._args = {k: keydict.get(k) for k in self.additional_args}

    def __call__(self: Self, parser: Parser) -> Dict[str, Any]:
        if self._args:
            ret = self.fn(parser, **self._args)
            self._args.clear()
        else:
            ret = self.fn(parser)
        return ret


GOBJ_SPEC = {
        'objectID': _CustomFnSpec(
            _parse_object_id, (
                'defobj_cond', 'defobj_enable',
                'defobj_preserved', 'objectID'
            )
        ),
        'presence_flag': _CustomFnSpec(
            _parse_presence_flag, (
                'mode', 'parameters', 'dummy', 'presence_flag'
            )
        )
    }


POTI_SPEC = {
    'pos': _CustomFnSpec(
        _parse_poti_points, ('pos', 'property1', 'property2'),
        additional_args=('numpoints',)
    )
}


STGI_SPEC = {
    'speedfactor': _CustomFnSpec(
        _parse_stgi_speedfactor, ('padding', 'speedfactor')
    )
}


def section_add_attrs(
    struct: t.DataClass,
    indexing: bool = True,
    custom_fn: Optional[Dict[str, _CustomFnSpec]] = None,
):
    """
    Add attributes to a `BaseSection` class.
    Private use only. You don't need to use this decorator.

    Args:
        struct (Class): A class that has `__annotations__` attribute.
        offset (int): Section offset of kmp. See https://wiki.tockdom.com/wiki/KMP_(File_Format)#Mario_Kart_Wii_specific_file_header
        indexing (bool): Whether the struct can use indexing.
        Only STR0 or WIM0 is assumed.
        custom_fn (dict, optional): A dict of _CustomFnSpec.
        Only GOBJ, POTI or STGI is assumed.
    """
    def wrapper(cls):
        if not dataclasses.is_dataclass(struct):
            raise TypeError('struct should be a dataclass.')
        cls.__struct__ = struct
        cls.__indexing__ = indexing
        cls.__rname__ = cls.__name__

        if not (custom_fn is None or isinstance(custom_fn, dict)):
            raise TypeError(
                "custom_fn should be a dict of _CustomFnSpec. or None.")
        elif (custom_fn is not None
            and not all(
                    isinstance(k, _CustomFnSpec) for k in custom_fn.values()
                )
            ):
                raise TypeError(
                    "custom_fn should be a dict of _CustomFnSpec."
                )
        if custom_fn:
            # check duplicate ret_keys
            ret_keys = set()
            for spec in custom_fn.values():
                for key in spec.ret_keys:
                    if key in ret_keys:
                        raise ValueError(
                            f"Duplicate ret_key {key} in custom_fn."
                        )
                    ret_keys.add(key)
        cls.__custom_fn__ = custom_fn or {}
        return cls
    return wrapper


class HexPrinter:
    def __init__(self, cls: Union["BaseSection", "BaseStruct"]):
        self._cls = cls.copy()

    def __getattr__(self, name: str) -> Any:
        x = getattr(self._cls, name)
        if isinstance(x, np.ndarray) and x.dtype.kind in 'u?':
            return np.vectorize(hex)(x)
        elif isinstance(x, (float, np.float32)):
            return hex(int.from_bytes(fpack('>f', x), 'big'))
        elif isinstance(x, np.ndarray) and x.dtype.kind == 'f':
            return np.vectorize(
                lambda x: hex(int.from_bytes(fpack('>f', x), 'big'))
            )(x)
        try:
            return hex(x)
        except:
            return x


class _NamePrinter:
    ...


class BaseStruct:
    """Base class for all KMP structs"""
    def __eq__(self: Self, other: Self) -> bool:
        _data = dataclasses.asdict(self)
        _other_data = dataclasses.asdict(other)
        return all(np.array_equal(_data[k], _other_data[k]) for k in _data)

    def __copy__(self: Self) -> Self:
        _data = {k: v.copy() for k, v in dataclasses.asdict(self).items()}
        return self.__class__(**_data)

    def __setattr__(self: Self, __name: str, __value: Any) -> None:
        if hasattr(self, __name):
            _tvalue = getattr(self, __name)
            __value = np.array(__value, dtype=_tvalue.dtype)
            if __value.shape != _tvalue.shape:
                raise ValueError(
                    f"Cannot change the shape of {__name} to {__value.shape}"
                )
        super().__setattr__(__name, __value)

    def tolist(self: Self) -> Dict[str, Any]:
        ret = []
        for v in dataclasses.asdict(self).values():
            if v.shape == ():  # scalar
                ret.append(v.item())
            elif len(v.shape) == 1:  # 1D array
                ret.extend(v.tolist())
            else: # 2D array (poti)
                ret.append(v)
        return ret

    def tobytes(self: Self) -> bytes:
        _bytes = []
        for v in dataclasses.asdict(self).values():
            if v.dtype.kind in 'ui':
                v = v.newbyteorder('big')
            else:
                if v.shape == ():
                    v = np.array(float(v), dtype='>f4')
                else:
                    v = v.astype('>f4')
            _bytes.append(v.tobytes())
        return b''.join(_bytes)

    @property
    def hex(self: Self):
        """Use this method to print values as hex"""
        return HexPrinter(self)

    def copy(self: Self) -> Self:
        """Copy the struct."""
        return self.__copy__()


def _efficient_read(
    parser: Parser,
    length: int,
    annotations: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Read data usin np.frombuffer and np.recarray"""
    assert parser.is_read_contiuously

    def _tostr(dtype: np.dtype):
        return dtype.byteorder + dtype.kind + str(dtype.alignment)

    dtype_map = []
    # create dtype arg
    # like [('foo', '>f4'), ('bar', '>i4''), ...]
    for k, v in annotations.items():
        dt, elems = t.get_dtype_and_size(v)
        if elems > 1:
            for e in range(elems):
                dtype_map.append((k + '_' + str(e), _tostr(dt)))
        else:
            dtype_map.append((k, _tostr(dt)))

    # read data
    *read_result, = zip(*np.frombuffer(parser._read(length), dtype=dtype_map))

    # create args for BaseSection.__init__
    init_kwgs = {}
    index = 0
    for k, v in annotations.items():
        dt, elems = t.get_dtype_and_size(v)
        if elems > 1:
            *arys, = zip(*[read_result[index + j] for j in range(elems)])
        else:
            arys = read_result[index]
        init_kwgs[k] = np.array(arys, dtype=annotations[k].__type__)
        index += elems
    return init_kwgs


class BaseSection:
    """Base class for all KMP/LEX sections"""
    def __init_subclass__(cls, *args, **kwargs) -> None:
        annotations = cls.__annotations__
        if not annotations:
            raise TypeError(
                "Annotations shoud be defined "
                f"when subclassing {cls.__name__}"
            )
        cls._metadata = {}
        for k, value in annotations.items():
            try:
                dt, v = t.get_dtype_and_size(value)
            except TypeError:
                raise TypeError(f"Unsupported type {value}") from None
            else:
                if isinstance(value, str):
                    raise TypeError("String is not supported")
            cls._metadata[k] = (dt, v)
        super().__init_subclass__(*args, **kwargs)

    def __init__(
        self: Self,
        obj: Union[Parser, DataDescriptor, pd.DataFrame],
        offset: Optional[int] = None
    ) -> None:
        """
        Create a new instance of the structure

        Args:
            obj (_BinaryParser, DataDescriptor, pd.DataFrame): Input object
            offset (int, optional): Offset of the section in the file. Defaults to None.
        """
        if self.__class__.__name__ == "BaseSection":
            raise TypeError("BaseSection cannot be instantiated")

        if isinstance(obj, Parser):
            if offset is None:
                raise TypeError("Offset should be provided when using parser")
            with obj.read_contiuously(offset, back=True):
                self._init_from_parser(obj)
        elif isinstance(obj, DataDescriptor):
            self._init_from_descriptor(obj)
        elif isinstance(obj, pd.DataFrame):
            self._init_from_dataframe(obj)
        else:
            raise TypeError(
                "Input should be either a Parser or a DataDescriptor instance. "
                f"Got {type(obj)}"
            )

    def __getitem__(
        self: Self,
        itemkey: Union[int, slice, Sequence[int], EllipsisType]
    ) -> Self:
        assert self.__indexing__, f"{self.section} is not indexing"
        if isinstance(itemkey, int):
            return self._rdata[itemkey]
        elif isinstance(itemkey, (slice, tuple, list, np.ndarray)):
            return self.__class__(self._to_descriptor(itemkey))
        elif isinstance(itemkey, EllipsisType):
            return self.copy()
        raise TypeError(f"Unsupported type {type(itemkey)}")

    def __getattr__(self: Self, __name: str):
        if __name in self._metadata:
            return self._pgetter(__name)
        try:
            return super().__getattr__(__name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{__name}'"
            ) from None

    def __setattr__(self: Self, __name: str, __value: Any) -> None:
        if __name in self._metadata:
            return self._psetter(__name, __value)
        super().__setattr__(__name, __value)

    def __len__(self: Self):
        return self.entries

    def __copy__(self: Self) -> Self:
        return self.__class__(self._to_descriptor(None, copy=True))

    def __eq__(self: Self, other: Self):
        if not isinstance(other, self.__class__):
            return False
        return self._to_descriptor(None) == other._to_descriptor(None)

    def _repr_html_(self: Self) -> str:
        return self.to_dataframe()._repr_html_()

    def add(
        self: Self, obj: Union[Self, BaseStruct], copy: bool = True
    ) -> Self:
        """Add a new struct/section to the section"""
        assert self.__indexing__, f"Cannot add to {self.section}"
        if dataclasses.is_dataclass(obj):
            if self._rdata[0] != obj:
                raise ValueError("Cannot append different Struct.")
            self._rdata.append(obj)
        elif isinstance(obj, self.__class__):
            if self._rdata[0] != obj._rdata[0]:
                raise ValueError("Cannot add different Section.")
            self._rdata.extend(obj._rdata)
        else:
            raise TypeError(
                f"Cannot add {obj.__class__.__name__} to "
                f"{self.section}."
            )
        self._sync_entries()
        if copy:
            return self.copy()
        return self

    def copy(self: Self) -> Self:
        """Copy the section."""
        return self.__copy__()

    @property
    def metadata(self: Self) -> dict[str, int | tuple[None, int]]:
        """Return the metadata of the section"""
        return self._metadata

    @property
    def hex(self: Self) -> HexPrinter:
        return HexPrinter(self)

    def _pgetter(self: Self, name: str) -> np.ndarray:
        if not self.__indexing__:
            return getattr(self._rdata[0], name)
        # gather values from self._rdata
        arrdata = [getattr(rd, name) for rd in self._rdata]
        if not all(arrdata[0].shape == arr.shape for arr in arrdata[1:]):
            # poti
            arry = np.concatenate(arrdata, axis=0)
        else:
            arry = np.array(arrdata, dtype=self._metadata[name][0])
        return arry

    def _psetter(self: Self, name: str, value: np.ndarray):
        if not self.__indexing__ and value.ndim == 0:
            value = value[None]
        elem = self._metadata[name][1]
        if isinstance(elem, int):
            elem = (elem,)
        expected_shape = (int(self.entries), *elem)
        if len(expected_shape) != value.ndim:
            if not (
                len(expected_shape) > 1
                and expected_shape[-1] == 1
                and value.ndim == len(expected_shape) - 1
            ):
                raise ValueError(
                    f"Shape mismatch. Expected {expected_shape}, got {value.shape}"
                )
        for i, rd in enumerate(self._rdata):
            setattr(rd, name, value[i])

    def _to_descriptor(
        self: Self,
        itemkey : Union[slice, Sequence[int], None],
        copy: bool = False
    ) -> DataDescriptor:
        kwg = dict()
        kwg['section'] = self.section

        if isinstance(itemkey, slice):
            itemkeys = list(range(*itemkey.indices(self.entries)))
            kwg['entries'] = np.uint16(len(itemkeys))
        else:
            itemkeys = list(itemkey or range(self.entries))
            kwg['entries'] = np.uint16(len(itemkeys))

        special_name = _SpecialName.get(self.section, 'ignored')
        if self.section == 'CAME':
            op_camera = getattr(self, special_name[0])
            if op_camera not in itemkeys:
                warnings.warn(
                    'The CAME opening index is not in the itemkey. '
                    'This may cause freezing in the game.'
                )
            padding = getattr(self, special_name[1])
            kwg['additional'] = op_camera
            kwg['padding'] = padding
            kwg['special_name'] = special_name
        elif special_name == 'total_points':
            kwg['additional'] = np.uint16(sum(a.numpoints for a in self._rdata))
            kwg['padding'] = None
            kwg['special_name'] = special_name
        else:
            kwg['additional'] = np.uint16(getattr(self, special_name))
            kwg['padding'] = None
            kwg['special_name'] = None
        kwg['descriptor'] = _ListMax255()

        for i in itemkeys:
            rd = self._rdata[i]
            if copy:
                rd = rd.copy()
            kwg['descriptor'].append(rd)

        return DataDescriptor(**kwg)

    def tobytes(self: Self):
        """Convert section to bytes. Used for writing to file."""
        desciptor = self._to_descriptor(None)
        b = b''
        b += desciptor.section.encode('utf-8')
        b += desciptor.entries.newbyteorder('>').tobytes()

        b += desciptor.additional.newbyteorder('>').tobytes()
        if desciptor.padding is not None:
            b += desciptor.padding.newbyteorder('>').tobytes()
        b += b''.join(rd.tobytes() for rd in desciptor.descriptor)
        return b

    def to_dataframe(self: Self):
        """
        Convert section data to a pandas DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the data.
        """
        return pandas_utils.to_dataframe(self.section, self)

    def _init_from_dataframe(self, df: pd.DataFrame):
        pandas_utils.from_dataframe(df, self)

    def _init_from_descriptor(self: Self, descriptor: DataDescriptor):
        assert self.__rname__ == descriptor.section, (
            "Cannot init from descriptor of different section."
        )
        setattr(self, "section", descriptor.section)
        setattr(self, "entries", descriptor.entries)

        if descriptor.special_name is not None:
            if type(descriptor.special_name) is list:
                setattr(
                    self, descriptor.special_name[0], descriptor.additional)
                setattr(
                    self, descriptor.special_name[1], descriptor.padding)
            else:
                setattr(self, descriptor.special_name, descriptor.additional)
                setattr(self, "padding", descriptor.padding)
        else:
            setattr(self, "ignored", descriptor.additional)

        setattr(self, "_rdata", descriptor.descriptor)

    def _init_from_parser(self: Self, parser: Parser) -> None:
        assert parser.is_read_contiuously, "Parser is not read continuously"

        section = parser.read_string(4)
        entries = parser.read_uint16()

        # set attributes header info
        setattr(self, "section", section)
        setattr(self, "entries", entries)
        special_name = _SpecialName.get(section, None)
        if type(special_name) is str:
            setattr(self, special_name, parser.read_uint16())
        elif type(special_name) is list and len(special_name) == 2:
            values = parser.read_uint8(2)
            for name, value in zip(special_name, values):
                setattr(self, name, value)
        else:
            setattr(self, "ignored", parser.read_uint16())

        _rdata = _ListMax255()
        struct_cls = self.__struct__

        pass_ret_keys = []
        for k, v in self.__custom_fn__.items():
            v = list(v.ret_keys)
            v.remove(k)
            pass_ret_keys.extend(v)
        for _ in range(entries):
            # read normally
            _init_kwargs = {}
            for key, value in self.__annotations__.items():
                if key in pass_ret_keys:
                    continue
                # check if the key is a custom function
                spec = self.__custom_fn__.get(key, None)
                if spec is not None:
                    spec.set_args(_init_kwargs)
                    for attrk, attv in spec(parser).items():
                        _init_kwargs[attrk] = attv
                    continue
                # n in expected to be an integer
                dt, n = t.get_dtype_and_size(value)
                if not isinstance(n, int):
                    raise TypeError(
                        "Invalid type for n in get_dtype_and_size. "
                        "Use if `custom_fn` instead.`"
                    )
                data = parser.read_number(dt, n)
                _init_kwargs[key] = data
            _rdata.append(struct_cls(**_init_kwargs))

        self._rdata = _rdata

    def _to_str(self: Self, brackets: str = '()'):
        lb, rb = brackets
        _str = self.section + lb + 'entries=' + str(self.entries)
        pad_name = _SpecialName.get(self.section, 'ignored') # type: ignore
        if isinstance(pad_name, list):
            _str += (
                ', ' + str(pad_name[0]) + '=' + str(getattr(self, pad_name[0]))
            )
            _str += (
                ', ' + str(pad_name[1]) + '=' + str(getattr(self, pad_name[1]))
            )
        else:
            _str += (', ' + str(pad_name) + '=' + str(getattr(self, pad_name)))

        meta_str = []
        for key, data in self._metadata.items():
            dtype, elements = data
            if isinstance(elements, tuple):
                elements = elements[-1]

            if elements == 1:
                size_str = 'scalar, '
            else:
                size_str = f'{elements} elements, '
            meta_str.append(key + "=(" + size_str + str(dtype) + ")")
        meta_str = ", ".join(meta_str)
        _str += ', ' + meta_str + rb
        return _str

    def _sync_entries(self: Self):
        """Sync `entries` with the length of the data."""
        if hasattr(self, 'total_points'):
            total_points = np.uint16(sum(x.numpoints for x in self._rdata))
            if total_points > 255:
                raise OverflowError(
                    "This section exceeds the maximum number of points "
                    f"({total_points} > 255)."
                )
            self.total_points = total_points
        self.entries = np.uint16(len(self._rdata))

    def __str__(self: Self) -> str:
        if self.__indexing__:
            return self._to_str('()')
        return repr(self._rdata[0])

    __repr__ = __str__
