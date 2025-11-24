import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


class CustomSafeDumper(yaml.SafeDumper):
    pass

def _seq(d, seq):
    return d.represent_sequence(yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, list(seq))

def _df_representer(dumper, df: pd.DataFrame):
    node = {
        "columns": df.columns.tolist(),
        "index": df.index.tolist(),
        "data": df.to_numpy().tolist(),
    }
    return dumper.represent_mapping("!dataframe", node)

# stdlib
CustomSafeDumper.add_representer(tuple, _seq)
CustomSafeDumper.add_representer(set,   lambda d, v: _seq(d, sorted(v, key=str)))
CustomSafeDumper.add_representer(Path, lambda d, v: d.represent_str(str(v)))
CustomSafeDumper.add_representer(datetime,   lambda d, v: d.represent_str(v.isoformat()))

# numpy
CustomSafeDumper.add_multi_representer(np.integer,  lambda d, v: d.represent_int(int(v)))
CustomSafeDumper.add_multi_representer(np.floating, lambda d, v: d.represent_float(float(v)))
CustomSafeDumper.add_multi_representer(np.bool_,    lambda d, v: d.represent_bool(bool(v)))
CustomSafeDumper.add_multi_representer(np.ndarray,  lambda d, v: d.represent_list(v.tolist()))

# pandas
CustomSafeDumper.add_representer(pd.Timestamp, lambda d, v: d.represent_str(v.isoformat()))
CustomSafeDumper.add_representer(pd.Timedelta, lambda d, v: d.represent_str(str(v)))
CustomSafeDumper.add_representer(type(pd.NaT), lambda d, v: d.represent_none("null"))
CustomSafeDumper.add_representer(pd.Series, lambda d, v: d.represent_list(v.tolist()))
CustomSafeDumper.add_representer(pd.Index,  lambda d, v: d.represent_list(v.tolist()))
CustomSafeDumper.add_representer(pd.DataFrame, _df_representer)

def safe_dump_yaml(data, stream=None, **kwargs):
    """YAML dump using CustomSafeDumper; returns a string if stream is None."""
    params = dict(allow_unicode=True, sort_keys=False, default_flow_style=False)
    params.update(kwargs)
    if stream is None:
        return yaml.dump(data, Dumper=CustomSafeDumper, **params)
    yaml.dump(data, stream, Dumper=CustomSafeDumper, **params)