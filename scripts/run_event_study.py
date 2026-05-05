"""
Event study: birth-region treatment around leader spell entry.

Design
------
  Core window : t = -2, -1 (pre), 0, +1, +2 (post-entry). Ref = t=-1.

  This script now matches the paper's headline 2005--2017 samples rather than
  the older 2005--2019 joint panel. Each outcome is estimated on its own sample:

    - ln_ntl          : harmonized nightlights panel
    - no2_intensity   : nightlights x NO2 joint panel
    - pm25_intensity  : nightlights x PM2.5 joint panel

  A spell is eligible only if it stays in office through t=+2. We then require
  conflict-free observations in every core bin. This produces a clean in-office
  event study with no post-exit points in the main figure.

Outputs
-------
  data/event_study_results.csv  -- coefficients
  data/regression_table.csv     -- static regression table on matched samples
  analysis/event_study.pdf      -- three-panel figure
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(exist_ok=True)

PLAD_PATH = DATA / "political leaders" / "PLAD_April_2024_gadm41.parquet"
NTL_PATH = DATA / "nightlights_adm2_green_favoritism_panel.parquet"
NO2_PATH = DATA / "no2_adm2_acag_panel.parquet"
PM25_PATH = DATA / "pm25_adm2_acag_panel.parquet"

YEAR_LO, YEAR_HI = 2005, 2017
ENTRY_LO = YEAR_LO + 2   # 2007  (so t=-2 is always >= YEAR_LO)
ENTRY_HI = YEAR_HI - 2   # 2015  (so t=+2 is always <= YEAR_HI)
REF_PERIOD = -1
CORE_T = list(range(-2, 3))          # -2 ... +2
DUMMY_T = [t for t in CORE_T if t != REF_PERIOD]

OUTCOMES = ["ln_ntl", "no2_intensity", "pm25_intensity"]
OUTCOME_LABELS = {
    "ln_ntl": "Log Nightlights",
    "no2_intensity": r"NO$_2$ Pollution Intensity",
    "pm25_intensity": r"PM$_{2.5}$ Pollution Intensity",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_plad() -> pd.DataFrame:
    plad = pd.read_parquet(PLAD_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    plad = plad.loc[valid_gid_mask(plad["gid_2"])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].astype(str).str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)
    fixed = plad.copy()
    for _, grp in fixed.groupby("gid_0"):
        idxs = grp.index.tolist()
        for i in range(len(idxs) - 1):
            if fixed.loc[idxs[i], "endyear"] >= fixed.loc[idxs[i + 1], "startyear"]:
                fixed.loc[idxs[i], "endyear"] = fixed.loc[idxs[i + 1], "startyear"] - 1
    return fixed


def build_leader_years(plad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for idx, row in plad.iterrows():
        for yr in range(max(int(row["startyear"]), YEAR_LO),
                        min(int(row["endyear"]), YEAR_HI) + 1):
            rows.append({
                "GID_2": row["gid_2"],
                "GID_0": row["gid_0"],
                "year": yr,
                "spell_id": idx,
                "birth_region_leader": 1,
            })
    ly = pd.DataFrame(rows).drop_duplicates(["GID_2", "year"])
    spell_map = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])
    return ly, spell_map


def assemble_panel(base: pd.DataFrame,
                   plad: pd.DataFrame,
                   *,
                   add_no2: bool = False,
                   add_pm25: bool = False) -> pd.DataFrame:
    ly, spell_map = build_leader_years(plad)
    p = base[base["year"].between(YEAR_LO, YEAR_HI)].copy()
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]], on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(spell_map, on=["GID_0", "year"], how="left")
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    p = p.sort_values(["GID_2", "year"]).reset_index(drop=True)
    p["brl_lag1"] = p.groupby("GID_2")["birth_region_leader"].shift(1).fillna(0).astype(int)
    p["brl_lag2"] = p.groupby("GID_2")["birth_region_leader"].shift(2).fillna(0).astype(int)

    if add_no2:
        no2 = pd.read_parquet(NO2_PATH)
        no2 = no2.loc[valid_gid_mask(no2["GID_2"])].dropna(subset=["no2_mean"])
        no2 = no2[(no2["year"].between(YEAR_LO, YEAR_HI)) & (no2["no2_mean"] >= 0)]
        p = p.merge(no2[["GID_2", "year", "no2_mean"]], on=["GID_2", "year"], how="inner")
        p["ln_no2"] = np.log(p["no2_mean"] + 0.01)
        p["no2_intensity"] = p["ln_no2"] - p["ln_ntl"]

    if add_pm25:
        pm25 = pd.read_parquet(PM25_PATH)
        pm25 = pm25.loc[valid_gid_mask(pm25["GID_2"])].dropna(subset=["pm25_mean"])
        pm25 = pm25[(pm25["year"].between(YEAR_LO, YEAR_HI)) &
                    (pm25["pm25_mean"] >= 0) & (pm25["pm25_mean"] < 1000)]
        p = p.merge(pm25[["GID_2", "year", "pm25_mean"]], on=["GID_2", "year"], how="inner")
        p["ln_pm25"] = np.log(p["pm25_mean"] + 0.01)
        p["pm25_intensity"] = p["ln_pm25"] - p["ln_ntl"]

    return p


def load_outcome_panels(plad: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ntl = pd.read_parquet(NTL_PATH)
    ntl = ntl.loc[valid_gid_mask(ntl["GID_2"])].dropna(subset=["ntl_mean"]).copy()
    ntl["ln_ntl"] = np.log(ntl["ntl_mean"] + 0.01)

    panels = {
        "ln_ntl": assemble_panel(ntl, plad),
        "no2_intensity": assemble_panel(ntl, plad, add_no2=True),
        "pm25_intensity": assemble_panel(ntl, plad, add_pm25=True),
    }
    return panels


# ---------------------------------------------------------------------------
# Event assignment
# ---------------------------------------------------------------------------

def build_event_assignments(plad: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of candidate event-study observations.

    Columns: GID_2, GID_0, year, event_label, sort_order, spell_idx.

    Only spells that are still active through t=+2 are eligible. Pre-period
    years must be conflict-free: no other spell can be active in that birth
    region during the lead years.
    """
    treated: dict[tuple, set] = {}
    for idx, row in plad.iterrows():
        for yr in range(int(row["startyear"]), int(row["endyear"]) + 1):
            if YEAR_LO <= yr <= YEAR_HI:
                treated.setdefault((row["gid_2"], yr), set()).add(idx)

    candidate: list[dict] = []
    for idx, row in plad.iterrows():
        start = int(row["startyear"])
        end = int(row["endyear"])
        gid2 = row["gid_2"]
        gid0 = row["gid_0"]

        if not (ENTRY_LO <= start <= ENTRY_HI):
            continue
        if end < start + 2:
            continue

        ok = True
        spell_rows: list[dict] = []
        for t in CORE_T:
            yr = start + t
            lbl = "et_ref" if t == REF_PERIOD else f"et_{t}"
            if not (YEAR_LO <= yr <= YEAR_HI):
                ok = False
                break
            if t < 0 and (treated.get((gid2, yr), set()) - {idx}):
                ok = False
                break
            spell_rows.append({
                "GID_2": gid2,
                "GID_0": gid0,
                "year": yr,
                "event_label": lbl,
                "sort_order": t,
                "spell_idx": idx,
            })
        if ok:
            candidate.extend(spell_rows)

    ev_all = pd.DataFrame(candidate)

    dupes = ev_all.duplicated(subset=["GID_2", "year"], keep=False)
    if dupes.sum():
        print(f"  Removed {dupes.sum()} rows with conflicting (GID_2, year) claims")
    ev_clean = ev_all[~dupes].copy()

    core_labels = {"et_ref"} | {f"et_{t}" for t in DUMMY_T}
    spell_has = (ev_clean.groupby("spell_idx")["event_label"].apply(set))
    good_spells = spell_has[spell_has.apply(lambda s: core_labels <= s)].index
    ev = ev_clean[ev_clean["spell_idx"].isin(good_spells)].copy()

    n_total = ev_all["spell_idx"].nunique()
    n_kept = len(good_spells)
    print(f"  Kept {n_kept} fully-balanced in-office spells out of {n_total} candidates")
    print(f"  Balanced entry window {ENTRY_LO}-{ENTRY_HI}")
    for lbl in ["et_-2", "et_ref", "et_0", "et_1", "et_2"]:
        sub = ev[ev["event_label"] == lbl]
        print(f"    {lbl:8s}  {len(sub):4d} obs  ({sub['spell_idx'].nunique()} spells)")
    return ev


def add_event_labels(panel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p = p.merge(ev[["GID_2", "year", "event_label", "sort_order", "spell_idx"]],
                on=["GID_2", "year"], how="left")
    return p


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

def run_event_study(panel: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, int]:
    core_labels = {"et_ref"} | {f"et_{t}" for t in DUMMY_T}

    event_obs = panel.loc[panel["event_label"].isin(core_labels) & panel[outcome].notna(),
                          ["spell_idx", "event_label"]].drop_duplicates()
    spell_has = event_obs.groupby("spell_idx")["event_label"].apply(set)
    good_spells = spell_has[spell_has.apply(lambda s: core_labels <= s)].index

    p = panel.copy()
    bad_event_rows = p["event_label"].isin(core_labels) & ~p["spell_idx"].isin(good_spells)
    p.loc[bad_event_rows, ["event_label", "sort_order", "spell_idx"]] = [pd.NA, pd.NA, pd.NA]

    dummy_labels = [f"et_{t}" for t in DUMMY_T]
    p = p.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    for lbl in dummy_labels:
        p[lbl] = (p["event_label"] == lbl).astype(float)
    p = p.set_index(["GID_2", "year"])
    rhs = " + ".join(f"`{lbl}`" for lbl in dummy_labels)
    res = PanelOLS.from_formula(
        f"{outcome} ~ {rhs} + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])

    rows = [{
        "event_label": "et_ref",
        "sort_order": REF_PERIOD,
        "coef": 0.0,
        "se": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
        "nobs": int(res.nobs),
    }]
    for lbl in dummy_labels:
        c = float(res.params[lbl])
        se = float(res.std_errors[lbl])
        so = int(lbl.split("_")[1])
        rows.append({
            "event_label": lbl,
            "sort_order": so,
            "coef": c,
            "se": se,
            "ci_lo": c - 1.96 * se,
            "ci_hi": c + 1.96 * se,
            "nobs": int(res.nobs),
        })
    df = pd.DataFrame(rows).sort_values("sort_order").reset_index(drop=True)
    df["outcome"] = outcome
    return df, int(len(good_spells))


def run_static(panel: pd.DataFrame, outcome: str) -> dict:
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    p = p.set_index(["GID_2", "year"])
    res = PanelOLS.from_formula(
        f"{outcome} ~ birth_region_leader + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])
    v = "birth_region_leader"
    return {
        "outcome": outcome,
        "coef": float(res.params[v]),
        "se": float(res.std_errors[v]),
        "pval": float(res.pvalues[v]),
        "nobs": int(res.nobs),
        "treated": int(p[v].sum()),
        "clusters": int(p["spell_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_event_study(results: dict[str, pd.DataFrame],
                     spells_kept: dict[str, int],
                     out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    color = "#1f77b4"
    x_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    x_ticks = [0, 1, 2, 3, 4]
    x_labels = ["-2", "-1", "0", "+1", "+2"]

    for ax, outcome in zip(axes, OUTCOMES):
        df = results[outcome].sort_values("sort_order")
        xs = np.array([x_map[int(v)] for v in df["sort_order"]])
        c = df["coef"].values
        lo = df["ci_lo"].values
        hi = df["ci_hi"].values

        ax.fill_between(xs, lo, hi, alpha=0.18, color=color)
        ax.plot(xs, c, color=color, lw=1.8, marker="o", ms=4.5, zorder=3)

        ref_i = df["event_label"].tolist().index("et_ref")
        ax.scatter([xs[ref_i]], [0], s=44, color="white",
                   edgecolors=color, linewidths=1.4, zorder=4)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(1.5, color="firebrick", lw=1.1, alpha=0.85)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_xlabel("Years relative to leader entry", fontsize=9)
        ax.set_ylabel("Coefficient estimate", fontsize=9)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

        n_spells = spells_kept[outcome]
        nobs = int(df["nobs"].iloc[0])
        ax.annotate(f"N={nobs:,}  ({n_spells} balanced spells)",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=7.5, va="top")

    fig.suptitle(
        "Event Study Around Leader Entry  |  Balanced in-office ±2 window  |  Reference: t = -1\n"
        "Outcome-specific 2005-2017 panels  |  Region FE + Country-Year FE  |  SE clustered by spell",
        fontsize=8.5, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(">> Loading PLAD...")
    plad = load_plad()
    print(f"   {len(plad)} spells")

    print(">> Building event assignments...")
    ev = build_event_assignments(plad)

    print(">> Loading outcome-specific panels...")
    panels = load_outcome_panels(plad)
    for outcome in OUTCOMES:
        print(f"   {outcome:15s} {len(panels[outcome]):,} obs")

    print(">> Adding event labels...")
    event_panels = {outcome: add_event_labels(panel, ev) for outcome, panel in panels.items()}

    print("\n>> Running event study regressions...")
    results: dict[str, pd.DataFrame] = {}
    spells_kept: dict[str, int] = {}
    for outcome in OUTCOMES:
        print(f"   {outcome}...", end=" ", flush=True)
        df, n_spells = run_event_study(event_panels[outcome], outcome)
        results[outcome] = df
        spells_kept[outcome] = n_spells
        print(f"N={int(df['nobs'].iloc[0]):,}  spells={n_spells}")

    pd.concat(results.values()).to_csv(DATA / "event_study_results.csv", index=False)

    print("\n-- Coefficients (ref = t=-1) ------------------------------------")
    all_labels = results[OUTCOMES[0]]["event_label"].tolist()
    print(f"{'label':9s}  " + "  ".join(f"{o:>24s}" for o in OUTCOMES))
    print("-" * 85)
    for lbl in all_labels:
        parts = []
        for outcome in OUTCOMES:
            rr = results[outcome][results[outcome]["event_label"] == lbl]
            c, s = float(rr["coef"]), float(rr["se"])
            if s == 0:
                parts.append(f"{'(ref)':>24s}")
            else:
                z = abs(c / s)
                star = "***" if z > 2.576 else "**" if z > 1.96 else "*" if z > 1.645 else ""
                parts.append(f"{c:+.4f}{star} ({s:.4f})")
        print(f"{lbl:9s}  " + "  ".join(f"{p:>24s}" for p in parts))

    print("\n>> Static regressions (birth_region_leader, matched samples)...")
    static_rows = []
    for outcome in OUTCOMES:
        r = run_static(panels[outcome], outcome)
        static_rows.append(r)
        z = r["coef"] / r["se"]
        star = "***" if abs(z) > 2.576 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.645 else ""
        print(f"   {outcome:22s}  {r['coef']:+.4f}{star}  ({r['se']:.4f})  "
              f"p={r['pval']:.3f}  N={r['nobs']:,}")
    pd.DataFrame(static_rows).to_csv(DATA / "regression_table.csv", index=False)

    print("\n>> Plotting...")
    plot_event_study(results, spells_kept, OUT_DIR / "event_study.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
