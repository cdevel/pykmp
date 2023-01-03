import numpy as np
import pytest

from pykmp._io import _parser as parser
from pykmp._io._parser import _BinaryParser as BP

_KMP_PATH = '/home/qb53/kdev/course.kmp'


def test_bp_exist():
    """ Test if the file exists. """
    with pytest.raises(FileNotFoundError):
        parser._BinaryParser('/tmp/not_exist')

def test_bp_non_empty():
    """ Test if the file is empty. """
    with pytest.raises(AssertionError):
        parser._BinaryParser(b'')


@pytest.fixture()
def binaryparser():
    bp = parser._BinaryParser(_KMP_PATH)
    yield bp


@pytest.mark.parametrize(
    'offset, raises',
    [(-1, ValueError), (0, None),
     (10**6, ValueError), (0.1, TypeError),
     ('0', TypeError), (None, TypeError)]
)
@pytest.mark.usefixtures('binaryparser')
def test_seek(binaryparser: BP, offset, raises):
    """Check invalid seek offset"""
    if raises is not None:
        with pytest.raises(raises):
            binaryparser.seek(offset)
    else:
        binaryparser.seek(offset)


class TestString:
    @pytest.mark.parametrize(
        'size, raises',
        [
            (-1, ValueError), # negative size
            (0, ValueError), # empty string
            (2, None),
            (4, None),
            (6, None),
            (8, ValueError), # cannot decode 0x9c
            (10**6, OverflowError), # overflow
        ]
    )
    @pytest.mark.usefixtures('binaryparser')
    def test_read_string(self, binaryparser: BP, size, raises):
        """Test `read_string`"""
        if raises is not None:
            with pytest.raises(raises):
                binaryparser.read_string(size, 0, True)
        else:
            assert binaryparser.read_string(size, 0, True)

    def test_rkmd(self, binaryparser: BP):
        header = binaryparser.read_string(4, back=False)
        assert header == 'RKMD'
        with pytest.raises(ValueError):
            binaryparser.read_string(4)
        binaryparser.seek(0)


class TestNumber:
    """Test `read_number`"""
    check_group = [
        '>u1', '>u2', '>u4', '>i2', '>f4',
        np.uint8, np.uint16, np.uint32, np.int16, np.float32
    ]
    @pytest.fixture
    def read_base(self):
        fp = open(_KMP_PATH, 'rb')
        _ = fp.read(4)
        yield fp
        fp.close()

    @pytest.mark.parametrize(
        'dtype, size, raises',
        [
            ('a', None, TypeError), # string is not allowed
            (np.complex64, None, TypeError), # complex is not allowed
            ('>u4', 1, ValueError),# buffer size < uint32 size
        ]
    )
    @pytest.mark.usefixtures('binaryparser')
    def test_read_number(self, binaryparser: BP, dtype, size, raises):
        """Check invalid dtype and size"""
        with pytest.raises(raises):
            binaryparser.read_number(dtype, size=size)

    @pytest.mark.parametrize('dtype', check_group)
    @pytest.mark.usefixtures('read_base')
    @pytest.mark.usefixtures('binaryparser')
    def test_number(
        self, binaryparser: BP, read_base, dtype
    ):
        """Check the dtype of the result"""
        result = binaryparser.read_number(dtype, start=4)
        expected_dtype = parser._convert_dtype_to_little_endian(dtype)
        expected = np.frombuffer(
            read_base.read(expected_dtype.alignment), dtype=expected_dtype
        )[0]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.usefixtures('binaryparser')
    def test_fillbyte(self, binaryparser: BP):
        """Check the fillbyte"""
        result = binaryparser.read_number(
            np.float32, size=2, start=0x2f9A, fillbyte=b'\xcc\xcd',
        )
        # expected float32 of 0x3dcccccd
        np.testing.assert_array_equal(result, np.array(0.1, dtype=np.float32))

    @pytest.mark.parametrize('dtype, n', zip(check_group, [4,4,4,4,4,2,2,2,2,2]))
    @pytest.mark.usefixtures('read_base')
    @pytest.mark.usefixtures('binaryparser')
    def test_read_n_arrays(self, binaryparser: BP, read_base, dtype, n):
        dtype = parser._convert_dtype_to_little_endian(dtype)
        call = 'read_{}'.format(dtype.name.lower())
        result = getattr(binaryparser, call)(n, start=4)
        expected = np.frombuffer(read_base.read(dtype.alignment*n), dtype=dtype, count=n)
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.usefixtures('read_base')
    @pytest.mark.usefixtures('binaryparser')
    def test_continuously(self, binaryparser: BP, read_base):
        with binaryparser.read_contiuously(start=0x54, back=True):
            pos = binaryparser.read_float32(3)
            rot = binaryparser.read_float32(3)
            player_idx = binaryparser.read_int16()
            padding = binaryparser.read_uint16(3)

        read_base.seek(0x54, 0)
        e_pos = np.frombuffer(read_base.read(12), dtype='>f4', count=3)
        e_rot = np.frombuffer(read_base.read(12), dtype='>f4', count=3)
        e_player_idx = np.frombuffer(read_base.read(2), dtype='>i2', count=1)
        e_padding = np.frombuffer(read_base.read(6), dtype='>u2', count=3)

        np.testing.assert_almost_equal(pos, e_pos)
        np.testing.assert_almost_equal(rot, e_rot)
        np.testing.assert_almost_equal(player_idx, e_player_idx)
        np.testing.assert_almost_equal(padding, e_padding)


@pytest.mark.usefixtures('binaryparser')
def test_double_read_contiuously(binaryparser: BP):
    """Check `read_contiuously`. Double call should raise an error"""
    with pytest.raises(RuntimeError):
        with binaryparser.read_contiuously():
            with binaryparser.read_contiuously():
                pass
