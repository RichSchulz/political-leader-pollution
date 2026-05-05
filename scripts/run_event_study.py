"""
Event study: birth-region treatment around leader spell entry.

Design
------
  Core window : t = -3, -2, -1 (pre), 0, +1, +2, +3 (post-entry). Ref = t=-1.
  Post-spell  : ps_1, ps_2 relative to each spell's own end-year.

  A spell is included only if it contributes a CONFLICT-FREE observation to
  every one of the 7 core bins.  This guarantees identical composition across
  all bins (no attrition, no FE-attenuation drift).

  Pre-period years must be clean (no other spell active for that birth region).
  Post-spell years excluded if another spell is already active or if ps_1
  points to the same calendar year as et_+3 (prevents double-claim for
  short-spell leaders).

Outputs
-------
  data/event_study_results.csv  -- coefficients
  data/regression_table.csv     -- static regression table
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
NTL_PATH  = DATA / "nightlights_adm2_green_favoritism_panel.parquet"
NO2_PATH  = DATA / "no2_adm2_acag_panel.parquet"
PM25_PATH = DATA / "pm25_adm2_acag_panel.parquet"

YEAR_LO, YEAR_HI = 2005, 2019
ENTRY_LO = YEAR_LO + 3   # 2008  (so t=-3 is always >= YEAR_LO)
ENTRY_HI = YEAR_HI - 3   # 2016  (so t=+3 is always <= YEAR_HI)
POST_SPELL_MAX = 2
REF_PERIOD = -1
CORE_T = list(range(-3, 4))           # -3 ... +3
DUMMY_T = [t for t in CORE_T if t != REF_PERIOD]

OUTCOMES = ["ln_ntl", "no2_intensity", "pm25_intensity"]
OUTCOME_LABELS = {
    "ln_ntl":         "Log Nightlights",
    "no2_intensity":  r"NO$_2$ Pollution Intensity",
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
    plad["endyear"]   = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].astype(str).str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)
    fixed = plad.copy()
    for _, grp in fixed.groupby("gid_0"):
        idxs = grp.index.tolist()
        for i in range(len(idxs) - 1):
            if fixed.loc[idxs[i], "endyear"] >= fixed.loc[idxs[i + 1], "startyear"]:
                fixed.loc[idxs[i], "endyear"] = fixed.loc[idxs[i + 1], "startyear"] - 1
    return fixed


def load_outcomes() -> pd.DataFrame:
    ntl  = pd.read_parquet(NTL_PATH)
    no2  = pd.read_parquet(NO2_PATH)
    pm25 = pd.read_parquet(PM25_PATH)

    ntl  = ntl.loc[valid_gid_mask(ntl["GID_2"])].dropna(subset=["ntl_mean"])
    no2  = no2.loc[valid_gid_mask(no2["GID_2"])].dropna(subset=["no2_mean"])
    pm25 = pm25.loc[valid_gid_mask(pm25["GID_2"])].dropna(subset=["pm25_mean"])
    no2  = no2[no2["no2_mean"] >= 0]
    pm25 = pm25[(pm25["pm25_mean"] >= 0) & (pm25["pm25_mean"] < 1000)]

    panel = ntl[ntl["year"].between(YEAR_LO, YEAR_HI)].copy()
    panel = panel.merge(
        no2[no2["year"].between(YEAR_LO, YEAR_HI)][["GID_2", "year", "no2_mean"]],
        on=["GID_2", "year"], how="inner")
    panel = panel.merge(
        pm25[pm25["year"].between(YEAR_LO, YEAR_HI)][["GID_2", "year", "pm25_mean"]],
        on=["GID_2", "year"], how="inner")

    panel["ln_ntl"]         = np.log(panel["ntl_mean"]  + 0.01)
    panel["ln_no2"]         = np.log(panel["no2_mean"]  + 0.01)
    panel["ln_pm25"]        = np.log(panel["pm25_mean"] + 0.01)
    panel["no2_intensity"]  = panel["ln_no2"]  - panel["ln_ntl"]
    panel["pm25_intensity"] = panel["ln_pm25"] - panel["ln_ntl"]
    panel["country_year"]   = panel["GID_0"] + "_" + panel["year"].astype(str)
    return panel


# ---------------------------------------------------------------------------
# Event assignment
# ---------------------------------------------------------------------------

def build_event_assignments(plad: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of event-study observations.

    Columns: GID_2, GID_0, year, event_label, sort_order, spell_idx.

    event_label values:
      et_ref            reference year (t = -1)
      et_{t}            core dummies t in {-3,-2,0,1,2,3}
      ps_{k}            post-spell k=1,2 relative to endyear

    Only spells that contribute conflict-free observations to ALL 7 core
    bins (et_ref + 6 dummies) are retained.
    """
    # (gid_2, year) -> set of spell indices that cover that year
    treated: dict[tuple, set] = {}
    for idx, row in plad.iterrows():
        for yr in range(int(row["startyear"]), int(row["endyear"]) + 1):
            if YEAR_LO <= yr <= YEAR_HI:
                treated.setdefault((row["gid_2"], yr), set()).add(idx)

    # Pass 1: collect all candidate rows
    candidate: list[dict] = []
    for idx, row in plad.iterrows():
        start = int(row["startyear"])
        end   = int(row["endyear"])
        gid2  = row["gid_2"]
        gid0  = row["gid_0"]

        if not (ENTRY_LO <= start <= ENTRY_HI):
            continue

        for t in CORE_T:
            yr  = start + t
            lbl = "et_ref" if t == REF_PERIOD else f"et_{t}"
            if not (YEAR_LO <= yr <= YEAR_HI):
                continue
            # Pre-period must not be treated by another spell
            if t < 0 and (treated.get((gid2, yr), set()) - {idx}):
                continue
            candidate.append(dict(GID_2=gid2, GID_0=gid0, year=yr,
                                  event_label=lbl, sort_order=t,
                                  spell_idx=idx))

        # Post-spell: skip if ps_1 clashes with et_3 (same calendar year)
        et3_yr = start + 3
        ps1_yr = end + 1
        if ps1_yr != et3_yr and end + POST_SPELL_MAX <= YEAR_HI:
            for k in range(1, POST_SPELL_MAX + 1):
                yr = end + k
                if yr > YEAR_HI:
                    continue
                if treated.get((gid2, yr), set()):
                    continue   # new leader active
                candidate.append(dict(GID_2=gid2, GID_0=gid0, year=yr,
                                      event_label=f"ps_{k}",
                                      sort_order=4 + k,
                                      spell_idx=idx))

    ev_all = pd.DataFrame(candidate)

    # Pass 2: remove any (GID_2, year) claimed by more than one spell
    dupes = ev_all.duplicated(subset=["GID_2", "year"], keep=False)
    if dupes.sum():
        print(f"  Removed {dupes.sum()} rows with conflicting (GID_2, year) claims")
    ev_clean = ev_all[~dupes].copy()

    # Pass 3: keep only spells that still have ALL 7 core bins intact
    core_labels = {"et_ref"} | {f"et_{t}" for t in DUMMY_T}
    spell_has = (ev_clean[ev_clean["event_label"].isin(core_labels)]
                 .groupby("spell_idx")["event_label"]
                 .apply(set))
    good_spells = spell_has[spell_has.apply(lambda s: core_labels <= s)].index
    ev = ev_clean[ev_clean["spell_idx"].isin(good_spells)].copy()

    n_total    = ev_all["spell_idx"].nunique()
    n_kept     = len(good_spells)
    print(f"  Kept {n_kept} fully-balanced spells out of {n_total} candidates")

    all_labels = sorted(
        ev["event_label"].unique(),
        key=lambda x: ev.loc[ev["event_label"] == x, "sort_order"].iloc[0]
    )
    print(f"  Balanced entry window {ENTRY_LO}-{ENTRY_HI}")
    for lbl in all_labels:
        sub = ev[ev["event_label"] == lbl]
        print(f"    {lbl:8s}  {len(sub):4d} obs  "
              f"({sub['spell_idx'].nunique()} spells)")
    return ev


def build_analysis_panel(panel: pd.DataFrame,
                         ev: pd.DataFrame,
                         plad: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p = p.merge(ev[["GID_2", "year", "event_label", "sort_order", "spell_idx"]],
                on=["GID_2", "year"], how="left")

    brl_rows: list[dict] = []
    for idx, row in plad.iterrows():
        for yr in range(max(int(row["startyear"]), YEAR_LO),
                        min(int(row["endyear"]),   YEAR_HI) + 1):
            brl_rows.append({"GID_2": row["gid_2"], "year": yr,
                             "birth_region_leader": 1, "spell_idx_brl": idx})
    brl = pd.DataFrame(brl_rows).drop_duplicates(["GID_2", "year"])
    p = p.merge(brl, on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)

    p["spell_id_num"] = p["spell_idx_brl"].combine_first(p["spell_idx"])
    p["spell_id"] = (p["spell_id_num"]
                     .apply(lambda x: f"spell_{int(x)}" if pd.notna(x) else None)
                     .fillna(p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"))

    p = (p.dropna(subset=["ln_ntl", "ln_no2", "ln_pm25"])
          .pipe(lambda d: d[np.isfinite(d["ln_ntl"]) &
                            np.isfinite(d["ln_no2"]) &
                            np.isfinite(d["ln_pm25"])]))
    return p


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

def run_event_study(panel: pd.DataFrame,
                    ev: pd.DataFrame,
                    outcome: str) -> pd.DataFrame:
    dummy_labels = sorted(
        [l for l in ev["event_label"].unique() if l != "et_ref"],
        key=lambda x: ev.loc[ev["event_label"] == x, "sort_order"].iloc[0]
    )
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    for lbl in dummy_labels:
        p[lbl] = (p["event_label"] == lbl).astype(float)
    p = p.set_index(["GID_2", "year"])
    rhs = " + ".join(f"`{l}`" for l in dummy_labels)
    res = PanelOLS.from_formula(
        f"{outcome} ~ {rhs} + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])

    rows = [{"event_label": "et_ref", "sort_order": REF_PERIOD,
             "coef": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
             "nobs": int(res.nobs)}]
    for lbl in dummy_labels:
        if lbl not in res.params.index:
            continue
        so = int(ev.loc[ev["event_label"] == lbl, "sort_order"].iloc[0])
        c  = float(res.params[lbl])
        se = float(res.std_errors[lbl])
        rows.append({"event_label": lbl, "sort_order": so,
                     "coef": c, "se": se,
                     "ci_lo": c - 1.96 * se, "ci_hi": c + 1.96 * se,
                     "nobs": int(res.nobs)})
    df = pd.DataFrame(rows).sort_values("sort_order").reset_index(drop=True)
    df["outcome"] = outcome
    return df


def run_static(panel: pd.DataFrame, outcome: str) -> dict:
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    p = p.set_index(["GID_2", "year"])
    res = PanelOLS.from_formula(
        f"{outcome} ~ birth_region_leader + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])
    v = "birth_region_leader"
    return {"outcome": outcome,
            "coef": float(res.params[v]),   "se":   float(res.std_errors[v]),
            "pval": float(res.pvalues[v]),   "nobs": int(res.nobs),
            "treated": int(p[v].sum()),      "clusters": int(p["spell_id"].nunique())}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _axis_layout(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    labels    = df["event_label"].tolist()
    sort_vals = df["sort_order"].tolist()
    xs: list[float] = []
    ticks: list[str] = []
    pos = 0.0
    gap_added = False
    for lbl, so in zip(labels, sort_vals):
        if lbl.startswith("ps_") and not gap_added:
            pos += 0.7   # visual gap before post-spell
            gap_added = True
        xs.append(pos)
        if lbl == "et_ref":
            ticks.append("-1")
        elif lbl.startswith("et_"):
            t = int(lbl.split("_")[1])
            ticks.append(f"+{t}" if t >= 0 else str(t))
        elif lbl.startswith("ps_"):
            k = lbl.split("_")[1]
            ticks.append(f"+{k}*")
        pos += 1.0
    return np.array(xs), ticks


def plot_event_study(results: dict[str, pd.DataFrame],
                     ev: pd.DataFrame,
                     out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    color = "#1f77b4"

    for ax, outcome in zip(axes, OUTCOMES):
        df = results[outcome].sort_values("sort_order")
        xs, ticks = _axis_layout(df)
        c  = df["coef"].values
        lo = df["ci_lo"].values
        hi = df["ci_hi"].values

        ax.fill_between(xs, lo, hi, alpha=0.18, color=color)
        ax.plot(xs, c, color=color, lw=1.8, marker="o", ms=4.5, zorder=3)

        # Reference: hollow dot
        ref_i = df["event_label"].tolist().index("et_ref")
        ax.scatter([xs[ref_i]], [0], s=44, color="white",
                   edgecolors=color, linewidths=1.4, zorder=4)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        # Entry line between t=-1 and t=0
        et0_i = df["event_label"].tolist().index("et_0")
        ax.axvline((xs[ref_i] + xs[et0_i]) / 2,
                   color="firebrick", lw=1.1, alpha=0.85)

        # Dashed exit line before post-spell segment (if present)
        labels = df["event_label"].tolist()
        if any(l.startswith("ps_") for l in labels):
            et3_i = next(i for i, l in enumerate(labels) if l == "et_3")
            ps1_i = next(i for i, l in enumerate(labels) if l == "ps_1")
            ax.axvline((xs[et3_i] + xs[ps1_i]) / 2,
                       color="firebrick", lw=1.1, alpha=0.85, ls="--")

        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlabel("Years relative to spell entry", fontsize=9)
        ax.set_ylabel("Coefficient estimate", fontsize=9)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

        n_spells = ev["spell_idx"].nunique()
        nobs = int(df["nobs"].iloc[0])
        ax.annotate(f"N={nobs:,}  ({n_spells} spells)",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=7.5, va="top")
        ax.annotate("* = years after spell end",
                    xy=(0.97, 0.03), xycoords="axes fraction",
                    fontsize=6.5, ha="right", va="bottom", color="gray")

    fig.suptitle(
        "Event Study: Birth Regions Around Spell Entry  |  "
        "Balanced entry window 2008-2016  |  Reference: t = -1\n"
        "Region FE + Country-Year FE  |  SE clustered by spell  |  "
        "Solid red = entry  |  Dashed red = exit",
        fontsize=8.5, y=1.01
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

    print(">> Loading outcomes...")
    panel = load_outcomes()
    print(f"   {len(panel):,} obs in joint NTL x NO2 x PM2.5 panel")

    print(">> Building analysis panel...")
    panel = build_analysis_panel(panel, ev, plad)
    print(f"   {len(panel):,} obs  |  "
          f"birth-region-leader: {panel['birth_region_leader'].sum():,}")

    # Event study
    print("\n>> Running event study regressions...")
    results: dict[str, pd.DataFrame] = {}
    for outcome in OUTCOMES:
        print(f"   {outcome}...", end=" ", flush=True)
        df = run_event_study(panel, ev, outcome)
        results[outcome] = df
        print(f"N={int(df['nobs'].iloc[0]):,}")

    pd.concat(results.values()).to_csv(DATA / "event_study_results.csv", index=False)

    # Print table
    print("\n-- Coefficients (ref = t=-1) ------------------------------------")
    all_labels = results[OUTCOMES[0]]["event_label"].tolist()
    print(f"{'label':9s}  " + "  ".join(f"{o:>24s}" for o in OUTCOMES))
    print("-" * 85)
    for lbl in all_labels:
        parts = []
        for o in OUTCOMES:
            rr = results[o][results[o]["event_label"] == lbl]
            if rr.empty:
                parts.append(f"{'--':>24s}")
                continue
            c, s = float(rr["coef"]), float(rr["se"])
            if s == 0:
                parts.append(f"{'(ref)':>24s}")
            else:
                z    = abs(c / s)
                star = "***" if z > 2.576 else "**" if z > 1.96 else "*" if z > 1.645 else ""
                parts.append(f"{c:+.4f}{star} ({s:.4f})")
        print(f"{lbl:9s}  " + "  ".join(f"{p:>24s}" for p in parts))

    # Static regressions
    print("\n>> Static regressions (birth_region_leader)...")
    static_rows = []
    for outcome in OUTCOMES:
        r    = run_static(panel, outcome)
        static_rows.append(r)
        z    = r["coef"] / r["se"]
        star = "***" if abs(z) > 2.576 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.645 else ""
        print(f"   {outcome:22s}  {r['coef']:+.4f}{star}  ({r['se']:.4f})  "
              f"p={r['pval']:.3f}  N={r['nobs']:,}")
    pd.DataFrame(static_rows).to_csv(DATA / "regression_table.csv", index=False)

    print("\n>> Plotting...")
    plot_event_study(results, ev, OUT_DIR / "event_study.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
