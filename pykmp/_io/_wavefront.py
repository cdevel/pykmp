import dataclasses
import hashlib
from concurrent.futures import ProcessPoolExecutor
from typing import ClassVar, Union

import numpy as np
from typing_extensions import Literal, Self

_int_ptns = [
    'f' + ('\s' + '/'.join([r'+([-+]?\d+)'] * i)) * 3 for i in range(1, 4)]
_face_vert_dtypes = [
    [('v1', 'u2'), ('v2', 'u2'), ('v3', 'u2')],
    [('v1', 'u2'), ('vt1', 'u2'),
     ('v2', 'u2'), ('vt2', 'u2'),
     ('v3', 'u2'), ('vt3', 'u2')],
    [('v1', 'u2'), ('vt1', 'u2'), ('vn1', 'u2'),
     ('v2', 'u2'), ('vt2', 'u2'), ('vn2', 'u2'),
     ('v3', 'u2'), ('vt3', 'u2'), ('vn3', 'u2')]
]

def _get_regex_dtype(header) -> tuple[list[str], list[list[tuple[str, str]]]]:
    if header == 'v':
        return (
            [header + r'\s+([-+]?\d+\.?\d*|\.\d+[eE][-+]?\d+?)' * 3],
            [[('v1', 'f4'), ('v2', 'f4'), ('v3', 'f4')]]
        )
    elif header == 'f':
        return (_int_ptns, _face_vert_dtypes)
    raise ValueError(f'Unsupported header: {header}')


def _regex_parse(
    file: str, header: str
) -> Union[np.ndarray, None]:
    regex, dtype = _get_regex_dtype(header)
    for r, dt in zip(regex, dtype):
        try:
            data = np.fromregex(file=file, regexp=r, dtype=np.dtype(dt))
        except UnicodeDecodeError:
            data = np.fromregex(
                file=file, regexp=r, dtype=np.dtype(dt), encoding='cp932')
        # recarray -> ndarray
        data = (
            data.view(np.recarray).view(dt[0][-1]).reshape(-1, len(dt))
        )
        if data.shape[0] != 0:
            return data
    raise Exception(f"Failed to read {header} data.")


def _compute_normals_from_face(face_tris: np.ndarray) -> np.ndarray:
    """Compute normals from facetriangles.

    Args:
        face_tris (np.ndarray): The facetriangles.

    Returns:
        np.ndarray: The normals.
    """
    assert face_tris.shape[1] == 3, \
        "facetriangles must be (n, 3) array"

    # cross product
    cross_face = np.cross(
        face_tris[:, 1] - face_tris[:, 0], face_tris[:, 2] - face_tris[:, 0]
    )

    # compute normals
    normals = cross_face / np.linalg.norm(cross_face, axis=1, keepdims=True)
    return normals


@dataclasses.dataclass(frozen=True)
class Wavefront:
    facetriangles: np.ndarray
    faces: np.ndarray
    vertices: np.ndarray
    _attr: ClassVar[dict[str, str]] = {'v': 'vertices', 'f': 'facetriangles'}

    # TODO: edge: https://stackoverflow.com/questions/66141584/how-do-i-a-generate-a-list-of-edges-from-a-given-list-of-vertex-indices

    @classmethod
    def read(
        cls: Self,
        path: str,
        backend: Literal['python', 'open3d'] = 'python'
    ) -> Self:
        """Read the wavefront file.

        Args:
            path (str): The path of the wavefront file.
            backend (str, optional): The backend to read the file. Defaults to 'python'.

        Returns:
            Wavefront: The wavefront object. attributes are vertices and facetriangles.
        """
        assert backend in ('python', 'open3d'), \
            "backend must be 'python' or 'open3d'"

        if backend == 'open3d':
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(path)
            v = np.asarray(mesh.vertices, dtype=np.float32)
            face = np.asarray(mesh.triangles, dtype=np.uint32)
            return cls(facetriangles=v[face], faces=face, vertices=v)

        files = [path] * len(cls._attr)
        init_kwgs = {}
        with ProcessPoolExecutor() as executor:
            fn = [executor.submit(_regex_parse, file=f, header=r)\
                for f, r in zip(files, cls._attr.keys())]
            for header, f in zip(cls._attr.keys(), fn):
                data = f.result()
                if data is None:
                    continue
                if header == 'v' and data.shape[1] == 6:
                    data = data[:, :3]
                init_kwgs[cls._attr[header]] = data

        if len(init_kwgs['vertices']) == 0:
            raise RuntimeError(
                'Failed to read Vertices: {}'.format(path))
        if len(init_kwgs['facetriangles']) == 0:
            raise RuntimeError(
                'Failed to read Face indexes: {}'.format(path))

        lsize = init_kwgs['facetriangles'].shape[1]
        if lsize == 3:
            s = ...
        elif lsize == 6:
            s = slice(None, None, 2)
        else:
            s = slice(None, None, 3)

        init_kwgs['faces'] = init_kwgs['facetriangles'][:, s] - 1
        init_kwgs['facetriangles'] = init_kwgs['vertices'][init_kwgs['faces']]
        return cls(**init_kwgs)


_WAVEFRONT_CACHE: dict[str, Wavefront] = {}


def _compute_md5(path: str) -> str:
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return md5


def read_wavefront(
    path: str,
    backend: Literal['python', 'open3d'] = 'python',
    cache: bool = True
) -> Wavefront:
    """Read the wavefront file.

    Args:
        path (str): The path of the wavefront file.
        backend (str, optional): The backend to read the file. Defaults to 'python'.
        cache (bool, optional): Cache the wavefront object. Defaults to True.

    Returns:
        Wavefront: The wavefront object. attributes are vertices and facetriangles.
    """

    md5 = _compute_md5(path)
    if cache and md5 in _WAVEFRONT_CACHE:
        return _WAVEFRONT_CACHE[md5]

    wf = Wavefront.read(path, backend=backend)
    if cache:
        _WAVEFRONT_CACHE[md5] = wf
    return wf
