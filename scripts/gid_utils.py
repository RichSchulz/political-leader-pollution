from __future__ import annotations

INVALID_GID_VALUES = frozenset({"", ".", "?", "nan"})


def valid_gid_mask(series):
    cleaned = series.astype("string").str.strip()
    return cleaned.notna() & ~cleaned.isin(INVALID_GID_VALUES)
