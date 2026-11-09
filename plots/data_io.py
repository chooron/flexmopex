"""Small IO and formatting helpers shared by the figure scripts."""
import json
import os

import pandas as pd
from matplotlib.ticker import FuncFormatter

MINUS = "\u2212"


def read_csv(data_dir, name):
    data = pd.read_csv(os.path.join(data_dir, name), dtype={"gauge_id": str})
    if "gauge_id" in data:
        data["gauge_id"] = data["gauge_id"].str.zfill(8)
    return data


def read_json(data_dir, name):
    with open(os.path.join(data_dir, name)) as handle:
        return json.load(handle)


def neg(text):
    """Typographic minus sign in printed numbers."""
    return str(text).replace("-", MINUS)


FMT = FuncFormatter(lambda v, _: neg(f"{v:g}"))
