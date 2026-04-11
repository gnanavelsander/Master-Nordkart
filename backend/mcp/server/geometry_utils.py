
import logging

import geopandas as gpd
import pandas as pd
from shapely import wkb as shapely_wkb, wkt as shapely_wkt

logger = logging.getLogger(__name__)




def _decode_geom(value):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, bytes):
        return shapely_wkb.loads(value)
    if isinstance(value, memoryview):
        return shapely_wkb.loads(bytes(value))
    if isinstance(value, str):
        s = value.strip()
        # Strip hex prefixes from PostgreSQL
        if s.startswith("\\x") or s.startswith("0x"):
            s = s[2:]
        # Try WKB hex first — must be even length and all hex chars
        if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
            try:
                return shapely_wkb.loads(bytes.fromhex(s))
            except Exception:
                pass
        # Fall back to WKT
        try:
            return shapely_wkt.loads(s)
        except Exception:
            return value
    return value

def _to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()



















def _detect_geom_col(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    preferred = ["geom", "geometry", "wkb_geom", "omrade_union", "omrade_union_valid"]
    for col in preferred:
        if col in df.columns:
            sample = df[col].dropna()
            if sample.empty:
                continue
            try:
                geom = _decode_geom(sample.iloc[0])
                if hasattr(geom, "geom_type"):
                    return col
            except Exception:
                pass

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        try:
            geom = _decode_geom(sample.iloc[0])
            if hasattr(geom, "geom_type"):
                return col
        except Exception:
            continue

    return None












def _to_gdf(
    rows: list[dict],
    geom_col: str | None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)

    detected_geom = geom_col or _detect_geom_col(df)
    if detected_geom is None:
        raise ValueError(f"No geometry column found. Available columns: {list(df.columns)}")

    if detected_geom not in df.columns:
        raise ValueError(
            f"Geometry column '{detected_geom}' not found. Available: {list(df.columns)}"
        )

    df = df.copy()
    df[detected_geom] = df[detected_geom].apply(_decode_geom)

    # Drop invalid rows instead of crashing — backend must be stable
    valid_mask = df[detected_geom].apply(lambda g: g is not None and hasattr(g, "geom_type"))
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        logger.warning(
            "_to_gdf: dropped %d/%d rows with invalid geometry in column '%s'",
            n_dropped, len(df), detected_geom,
        )
    df = df[valid_mask].reset_index(drop=True)

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)

    return gpd.GeoDataFrame(df, geometry=detected_geom, crs=crs)
