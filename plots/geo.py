"""Map layers for Figures 2 and 6.

Rules (fixes the boundary mismatch seen in earlier drafts):
  * The seven regional polygons (R1-R7, merged HUC02 units) define the map
    outline. Their shape is used as-is.
  * State boundaries are only a light background. They are reprojected to the
    SAME CRS as the regions and clipped to the union of the regions, so the
    two layers share one coordinate system and one outer outline.
  * Catchment points are projected from lon/lat into the same CRS.
All layers are drawn in EPSG:5070 (CONUS Albers Equal Area).
"""
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.ops import unary_union

CRS = "EPSG:5070"
REGION_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]

# Only used to build the DEMO region layer from state polygons.
_DEMO_GROUPS = {
    "R1": ["Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
           "New York", "New Jersey", "Pennsylvania", "Delaware", "Maryland", "District of Columbia"],
    "R2": ["Virginia", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama",
           "Tennessee", "Mississippi"],
    "R3": ["Michigan", "Wisconsin", "Minnesota", "Iowa", "Illinois", "Indiana", "Ohio", "Kentucky",
           "West Virginia", "Missouri"],
    "R4": ["North Dakota", "South Dakota", "Nebraska", "Kansas", "Montana", "Wyoming"],
    "R5": ["Louisiana", "Arkansas", "Oklahoma", "Texas", "New Mexico"],
    "R6": ["Colorado", "Utah", "Arizona", "Nevada", "California"],
    "R7": ["Washington", "Oregon", "Idaho"],
}
_EXCLUDE = {"Alaska", "Hawaii", "Puerto Rico"}


def _states_conus(states_path):
    st = gpd.read_file(states_path)
    name_col = next(c for c in ["name", "NAME", "STATE_NAME"] if c in st.columns)
    st = st[~st[name_col].isin(_EXCLUDE)].copy()
    st["name"] = st[name_col]
    return st


def build_demo_regions(states_path):
    """DEMO ONLY: approximate R1-R7 by dissolving states. Real runs must pass
    the merged-HUC02 region file used in the study."""
    st = _states_conus(states_path)
    lut = {s: r for r, ss in _DEMO_GROUPS.items() for s in ss}
    st["region"] = st["name"].map(lut)
    st = st.dropna(subset=["region"])
    reg = st.dissolve(by="region").reset_index()[["region", "geometry"]]
    return reg


def load_layers(regions_path, states_path, region_field="region", demo=False):
    """Return (regions, states_clipped, outline) in EPSG:5070."""
    if demo:
        regions = build_demo_regions(states_path)
    else:
        regions = gpd.read_file(regions_path)
        if region_field not in regions.columns:
            raise KeyError(f"region field '{region_field}' not in {regions_path}")
        regions = regions.rename(columns={region_field: "region"})
        regions = regions.dissolve(by="region").reset_index()[["region", "geometry"]]
    if regions.crs is None:
        raise ValueError("Region layer has no CRS; fix the source file before plotting.")
    regions = regions.to_crs(CRS)
    regions["geometry"] = regions.buffer(0)

    states = _states_conus(states_path)
    if states.crs is None:
        raise ValueError("State layer has no CRS; fix the source file before plotting.")
    states = states.to_crs(CRS)
    outline = unary_union(regions.geometry.values)
    states["geometry"] = states.geometry.intersection(outline)
    states = states[~states.geometry.is_empty]
    return regions, states, outline


_TF = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def project_points(lon, lat):
    x, y = _TF.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def map_extent(outline, padx=0.01, pady=0.02):
    xmin, ymin, xmax, ymax = outline.bounds
    dx, dy = (xmax - xmin) * padx, (ymax - ymin) * pady
    return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)


def draw_base(ax, regions, states, outline, region_lw=0.9, extent=None):
    """Light land fill, thin state lines, dark region boundaries."""
    gpd.GeoSeries([outline], crs=CRS).plot(ax=ax, color="#F4F4F4", edgecolor="none", zorder=0)
    states.boundary.plot(ax=ax, color="#D6D6D6", lw=0.3, zorder=1)
    regions.boundary.plot(ax=ax, color="#2E2E2E", lw=region_lw, zorder=3)
    x0, x1, y0, y1 = extent or map_extent(outline)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def region_label_points(regions):
    """Representative point inside each region (used for R labels)."""
    rp = regions.copy()
    rp["pt"] = rp.geometry.representative_point()
    return {r: (p.x, p.y) for r, p in zip(rp["region"], rp["pt"])}
