# Required forcing / attribute data

Training and the synthetic generator need the prepared CAMELS-US bundle. It is
not shipped because of its size (~255 MB); place these files here before running
`train/`.

| File | Content | Shape / dtype |
|---|---|---|
| `camels_dataset` | pickle of `(forcings, streamflow, attributes)` | forcings `(671, 12418, 3)` float, streamflow `(671, 12418, 1)`, attributes `(671, 35)` |
| `gage_id.npy` | basin IDs in canonical order | `(671,)` int |
| `camels_dates.npy` | daily dates for the forcing axis | `(12418,)` datetime64, 1980-10-01 to 2014-09-30 |
| `basin_groups/group_11.npy` … `group_17.npy` | the seven leave-one-region-out holdouts | integer ID arrays |
| `basin_groups/loro_group_*.npy` | auxiliary grouping used by the legacy analysis | integer ID arrays |

Forcing order is `prcp`, `tmean`, `pet`. The attribute order is the 35-name list
in `configs/flex.yaml` (`delta_model.nn_model.attributes`); the categorical
entries `dom_land_cover`, `geol_1st_class` and `geol_2nd_class` are excluded from
the correlation analysis.

`camels_dataset`, `gage_id.npy` and `camels_dates.npy` are in the package's
original data root; `FLEXMOPEX_DATA_DIR` or `--data` can point elsewhere. The
shipped `gage_id.npy`, `camels_dates.npy` and `basin_groups/` are the small
arrays actually used by the released checks.
