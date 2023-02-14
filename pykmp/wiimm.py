import os
import subprocess
import tempfile

from pykmp.struct._kmp import KMP

_WSZST_BIN_PATH = None


def _run_cmd(cmd):
    """Run command and return the output.

    Args:
        cmd (list): The command to run.

    Returns:
        str: The output of the command.
    """
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()


def set_wszst_path(path: str):
    """Set the path of wszst binary.

    Args:
        path (str): The path of wszst binary.
    """
    global _WSZST_BIN_PATH
    if not os.path.exists(path):
        raise FileNotFoundError('Cannot find wszst binary in {}'.format(path))
    _WSZST_BIN_PATH = path


def get_wszst_path() -> str:
    """Get the path of wszst binary.

    Returns:
        str: The path of wszst binary.
    """
    return find_wszst_path()


def find_wszst_path() -> str:
    """Find wszst binary in the system.

    Returns:
        str: The path of wszst binary.
    """
    global _WSZST_BIN_PATH
    if _WSZST_BIN_PATH is not None:
        return _WSZST_BIN_PATH
    # find wszst binary
    if os.name == 'nt':
        where_cmd = 'where'
    elif os.name == 'posix':
        where_cmd = 'which'
    else:
        raise NotImplementedError('Unsupported platform: {}'.format(os.name))

    result = subprocess.run([where_cmd, 'wszst'], stdout=subprocess.PIPE)
    binary_path = result.stdout.decode('utf-8').strip()

    if not binary_path:
        raise RuntimeError(
            'Cannot find wszst binary in the system. '
            'If you installed wszst manually, set the path of wszst binary '
            'using `pykmp.set_wszst_path(path)`.'
        )

    # split
    binary_path, _ = os.path.split(binary_path)
    _WSZST_BIN_PATH = binary_path
    return binary_path


_DRAW_SUPPRTED = [
    'CKPT',
    'CJGPT',
    'JGPT',
    'KTPT',
    'ENPT',
    'ITPT',
    'POTI',
    'CNPT',
    'ITEMBOXES',
    'COINS',
    'ROADOBJECTS',
    'SOLIDOBJECTS',
    'DECORATION',
    'BLACK',
    'WHITE',
    'KCL',
    'NONE',
    'ALL',
    'DETAILED',
    'DISPATCH',
    'WARNINGS'
]
# negative keywords
_DRAW_SUPPRTED += ['-' + d for d in _DRAW_SUPPRTED[:-5]]


def DRAW(
    kmp: KMP,
    *keywords: str,
    dest: str = None,
    overwrite: bool = False,
    **kwargs
):
    if keywords:
        for key in keywords:
            if key.upper() not in _DRAW_SUPPRTED:
                raise ValueError(
                    f'Unsupported keyword: key'
                    f'Supported keywords: {_DRAW_SUPPRTED}'
                )
    else:
        keywords = ('All', 'DETAILED', 'WARNINGS')

    cmdline = [get_wszst_path(), 'DRAW', ','.format(keywords)]


def _apply_to_kmp(kmp: KMP, command: str) -> KMP:
    """Apply the command to the KMP.

    Args:
        kmp (KMP): The KMP to apply the command.
        command (str): The command to apply.

    Returns:
        KMP: The KMP after applying the command.
    """
    # write kmp to temp file
    with tempfile.NamedTemporaryFile(suffix='.kmp') as kmp_file:
        kmp.write(kmp_file.name)
        # run command
        cmd = [os.path.join(find_wszst_path(), command), kmp_file.name]
        _run_cmd(cmd)
        # read kmp
        return KMP.fromfile(kmp_file.name)
