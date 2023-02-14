import warnings
from collections import defaultdict
from itertools import groupby
from typing import TypeVar

import numpy as np
import pandas as pd

from pykmp import _typing as t
from pykmp.struct.descriptor import DataDescriptor, _ListMax255, _SpecialName

Section = TypeVar('Section')
Struct = TypeVar('Struct')

def merge_pt_ph(pt: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    cols = list(pt.columns) + list(ph.columns)
    df = pd.DataFrame(columns=cols)

    for i in range(ph.shape[0]):
        loc_pt = pt.loc[
            ph.loc[i, 'start']:ph.loc[i, 'start'] + ph.loc[i, 'length'] - 1]
        loc_pt = loc_pt.reset_index(drop=True)
        loc_ph = ph.iloc[i:i+1, :]
        # add nan rows
        nan_df = pd.concat(
            [loc_ph] * (ph.loc[i, 'length'] - 1), ignore_index=True)
        # all values are NaN
        for col in nan_df.columns:
            nan_df[col] = np.nan

        loc_ph = pd.concat([loc_ph, nan_df], ignore_index=True, axis=0)
        df_ = pd.concat([loc_pt, loc_ph], axis=1)
        df = pd.concat([df, df_], ignore_index=True, axis=0)

    return df


def split_pt_ph(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)
    start_idx = cols.index('start')
    df_pt = df.iloc[:, :start_idx]
    df_ph = (
        df.iloc[:, start_idx:]
        .dropna(axis=0, how='all').reset_index(drop=True))
    return df_pt, df_ph


def to_dataframe(name: str, section: Section) -> pd.DataFrame:
    cols = []
    if name == 'POTI':
        cols.append('index')
    elif name == 'CAME':
        cols.append('op_camera')
    for k, dt in section.__annotations__.items():
        colname = t.get_colname(k, dt)
        if isinstance(colname, list):
            cols.extend(colname)
        else:
            cols.append(colname)

    # create the data
    data = defaultdict(list)
    toT = lambda x: x.T if isinstance(x, np.ndarray) else x
    for index, rd in enumerate(section._rdata):
        vals = rd.tolist()
        appender = 'append'
        if name == 'POTI':
            *rdl, = map(toT, [index] + vals)
            numpts = rd.numpoints
            # convert all elements to same size
            vals = []
            for val in rdl:
                if isinstance(val, np.ndarray):
                    vals.extend(val.tolist())
                else:
                    vals.append([val] + [np.nan] * (numpts - 1))
            appender = 'extend'
        elif name == 'CAME':
            vals = [1 if section.op_camera == index else np.nan] + vals
        for col, val in zip(cols, vals):
            getattr(data[col], appender)(val)

    df = pd.DataFrame(data, columns=cols)
    if name == 'POTI':
        df.drop(columns='numpoints', inplace=True)
    return df


def _gby_key(name):
    head = name.split('_')[0]
    if head == 'defobj':
        return name
    return head


def from_dataframe(
    df: pd.DataFrame,
    section: Section,
) -> None:
    init_kwgs = {}
    init_kwgs['section'] = section.__rname__
    df = df.copy()

    if section.__rname__ == 'POTI':
        init_kwgs['entries'] = np.uint16(len(df.dropna(axis=0)))
    else:
        init_kwgs['entries'] = np.uint16(len(df))

    special_name = _SpecialName.get(section.__rname__, 'ignored')
    iloc_index = list(range(init_kwgs['entries']))

    if section.__rname__ == 'CAME':
        try:
            op_camera = int(df.query('op_camera > 0')['op_camera'])
        except (ValueError, TypeError) as e:
            raise ValueError(
                'Cannot find the opening camera. See the error below.') from e
        if op_camera not in list(range(init_kwgs['entries'])):
            warnings.warn(
                'The CAME opening index is not in the itemkey. '
                'This may cause freezing in the game.'
            )
        init_kwgs['additional'] = np.uint8(op_camera)
        init_kwgs['padding'] = np.uint8(0)
        init_kwgs['special_name'] = special_name
        df.drop(columns='op_camera', inplace=True)
    elif section.__rname__ == 'POTI':
        init_kwgs['additional'] = np.uint16(len(df))
        init_kwgs['padding'] = None
        init_kwgs['special_name'] = special_name
        current = 0
        iloc_index = defaultdict(list)
        for i, idx in enumerate(df['index']):
            if not np.isnan(idx):
                current = int(idx)
            iloc_index[current].append(i)
        df.drop(columns='index', inplace=True)
    else:
        init_kwgs['additional'] = np.uint8(0)
        init_kwgs['padding'] = None
        init_kwgs['special_name'] = None

    attr_names = []
    colindex = 0
    for k, v in groupby(df.columns, key=_gby_key):
        v = list(v)
        if len(v) == 1:
            attr_names.append((v[0], colindex))
            colindex += 1
        else:
            attr_names.append((k, slice(colindex, colindex + len(v))))
            colindex += len(v)

    del colindex

    _rdata = _ListMax255()
    annotations = section.__annotations__

    for i in range(init_kwgs['entries']):
        _init_kwgs = {}
        if section.__rname__ == 'POTI':
            loc_df = df.iloc[iloc_index[i]]
            _init_kwgs['numpoints'] = np.uint16(len(loc_df))
        else:
            loc_df = df.iloc[i]
        for j, (attr_name, cols) in enumerate(attr_names):
            dt, _ = t.get_dtype_and_size(annotations[attr_name])
            if section.__rname__ == 'POTI' and j < 2:
                data = loc_df.iloc[0, cols]
            elif isinstance(loc_df, pd.DataFrame):
                data = loc_df.iloc[:, cols]
            else:
                data = loc_df.iloc[cols]
            data = dt.type(data)
            _init_kwgs[attr_name] = data
        _rdata.append(section.__struct__(**_init_kwgs))
    init_kwgs['descriptor'] = _rdata
    descriptor = DataDescriptor(**init_kwgs)
    section._init_from_descriptor(descriptor)
