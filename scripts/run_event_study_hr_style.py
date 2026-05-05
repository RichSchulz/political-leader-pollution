"""
Hodler-Raschky-style event study on the paper's 2005--2017 headline samples.

Design
------
  Pre-entry bins   : pre_-3, pre_-2, pre_-1 (reference period)
  In-office bins   : in_0, in_1, in_2, in_3, in_4, in_5, in_6p
  Post-exit bins   : post_1, post_2, post_3p

  The figure is estimated from a single fixed-effects regression per outcome,
  using the full 2005--2017 panel with ADM2 and country-year fixed effects.
  Unlike the strict balanced-window check, this specification is intentionally
  unbalanced across event times, closer to Hodler and Raschky (2014).

  Critical timing rule: in-office bins are assigned only while the leader is
  still in office. Post-exit bins begin only after the spell ends.

Outputs
-------
  data/event_study_hr_style_2005_2017_results.csv
  data/event_study_hr_style_2005_2017_support.csv
  analysis/event_study_hr_style_2005_2017.pdf
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
PRE_PERIODS = [-3, -2, -1]
INDIVIDUAL_IN_YEARS = [0, 1, 2, 3, 4, 5]
POST_PERIODS = [1, 2]
REF_LABEL = "pre_-1"

OUTCOMES = ["ln_ntl", "no2_intensity", "pm25_intensity"]
OUTCOME_LABELS = {
    "ln_ntl": "Log Nightlights",
    "no2_intensity": r"NO$_2$ Pollution Intensity",
    "pm25_intensity": r"PM$_{2.5}$ Pollution Intensity",
}

PLOT_ORDER = [
    "pre_-3", "pre_-2", "pre_-1",
    "in_0", "in_1", "in_2", "in_3", "in_4", "in_5", "in_6p",
    "post_1", "post_2", "post_3p",
]
SORT_ORDER = {lbl: i for i, lbl in enumerate(PLOT_ORDER)}
REG_LABELS = [lbl for lbl in PLOT_ORDER if lbl != REF_LABEL]


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
    fixed = fixed[fixed["endyear"] >= fixed["startyear"]].copy()
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
    return {
        "ln_ntl": assemble_panel(ntl, plad),
        "no2_intensity": assemble_panel(ntl, plad, add_no2=True),
        "pm25_intensity": assemble_panel(ntl, plad, add_pm25=True),
    }


def build_event_rows(plad: pd.DataFrame) -> pd.DataFrame:
    treated: dict[tuple[str, int], set[int]] = {}
    for idx, row in plad.iterrows():
        for yr in range(int(row["startyear"]), int(row["endyear"]) + 1):
            if YEAR_LO <= yr <= YEAR_HI:
                treated.setdefault((row["gid_2"], yr), set()).add(idx)

    rows: list[dict] = []
    for idx, row in plad.iterrows():
        start = int(row["startyear"])
        end = int(row["endyear"])
        gid2 = row["gid_2"]
        gid0 = row["gid_0"]

        # Leads
        for k in [3, 2, 1]:
            yr = start - k
            if not (YEAR_LO <= yr <= YEAR_HI):
                continue
            if treated.get((gid2, yr), set()) - {idx}:
                continue
            rows.append({
                "GID_2": gid2, "GID_0": gid0, "year": yr, "spell_idx": idx,
                "event_label": f"pre_-{k}",
            })

        # In-office years only
        for yr in range(max(start, YEAR_LO), min(end, YEAR_HI) + 1):
            rel = yr - start
            lbl = f"in_{rel}" if rel in INDIVIDUAL_IN_YEARS else "in_6p"
            rows.append({
                "GID_2": gid2, "GID_0": gid0, "year": yr, "spell_idx": idx,
                "event_label": lbl,
            })

        # Post-exit years until another spell starts in same birth region
        post_year = end + 1
        while post_year <= YEAR_HI:
            active_other = treated.get((gid2, post_year), set())
            if active_other:
                break
            rel = post_year - end
            lbl = f"post_{rel}" if rel in POST_PERIODS else "post_3p"
            rows.append({
                "GID_2": gid2, "GID_0": gid0, "year": post_year, "spell_idx": idx,
                "event_label": lbl,
            })
            post_year += 1

    ev = pd.DataFrame(rows)
    dupes = ev.duplicated(subset=["GID_2", "year"], keep=False)
    if dupes.any():
        print(f"  Removed {int(dupes.sum())} conflicting event rows")
        ev = ev.loc[~dupes].copy()

    ev["sort_order"] = ev["event_label"].map(SORT_ORDER)
    return ev


def attach_events(panel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    p = panel.merge(
        ev[["GID_2", "year", "event_label", "sort_order", "spell_idx"]],
        on=["GID_2", "year"],
        how="left",
    )
    return p


def run_regression(panel: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    for lbl in REG_LABELS:
        p[lbl] = (p["event_label"] == lbl).astype(float)
    p = p.set_index(["GID_2", "year"])
    rhs = " + ".join(f"`{lbl}`" for lbl in REG_LABELS)
    res = PanelOLS.from_formula(
        f"{outcome} ~ {rhs} + EntityEffects",
        data=p,
        other_effects=p["country_year"],
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])

    rows = [{
        "event_label": REF_LABEL,
        "sort_order": SORT_ORDER[REF_LABEL],
        "coef": 0.0,
        "se": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
        "nobs": int(res.nobs),
        "outcome": outcome,
    }]
    for lbl in REG_LABELS:
        c = float(res.params[lbl])
        se = float(res.std_errors[lbl])
        rows.append({
            "event_label": lbl,
            "sort_order": SORT_ORDER[lbl],
            "coef": c,
            "se": se,
            "ci_lo": c - 1.96 * se,
            "ci_hi": c + 1.96 * se,
            "nobs": int(res.nobs),
            "outcome": outcome,
        })

    support = (
        panel.loc[panel["event_label"].isin(PLOT_ORDER) & panel[outcome].notna()]
        .groupby("event_label")
        .agg(obs=("event_label", "size"),
             spells=("spell_idx", pd.Series.nunique))
        .reset_index()
    )
    support["sort_order"] = support["event_label"].map(SORT_ORDER)
    support["outcome"] = outcome
    return pd.DataFrame(rows).sort_values("sort_order"), support.sort_values("sort_order")


def run_static(panel: pd.DataFrame, outcome: str) -> dict:
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    p = p.set_index(["GID_2", "year"])
    res = PanelOLS.from_formula(
        f"{outcome} ~ birth_region_leader + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=p["spell_id"])
    v = "birth_region_leader"
    return {
        "coef": float(res.params[v]),
        "se": float(res.std_errors[v]),
        "pval": float(res.pvalues[v]),
        "nobs": int(res.nobs),
    }


def plot_results(results: dict[str, pd.DataFrame],
                 support: dict[str, pd.DataFrame],
                 static_rows: dict[str, dict],
                 out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    color = "#233b73"
    ci_color = "#7f8ca8"
    x = np.arange(len(PLOT_ORDER))
    tick_labels = ["-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6+", "+1", "+2", "+3+"]

    for ax, outcome in zip(axes, OUTCOMES):
        df = results[outcome].set_index("event_label").loc[PLOT_ORDER].reset_index()
        s = support[outcome].set_index("event_label").reindex(PLOT_ORDER)

        ax.plot(x, df["coef"], color=color, lw=1.8, marker="o", ms=4.2, zorder=3)
        ax.plot(x, df["ci_lo"], color=ci_color, lw=1.0, alpha=0.9)
        ax.plot(x, df["ci_hi"], color=ci_color, lw=1.0, alpha=0.9)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(static_rows[outcome]["coef"], color="gray", lw=1.0, ls=(0, (5, 3)), alpha=0.9)
        ax.axvline(2.5, color="firebrick", lw=1.1, alpha=0.85)
        ax.axvline(9.5, color="firebrick", lw=1.1, alpha=0.85, ls="--")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.set_xlabel("Years relative to leader entry / exit", fontsize=9)
        ax.set_ylabel("Coefficient estimate", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

        last_support = s.dropna().iloc[-1]
        ax.annotate(
            f"N={int(df['nobs'].iloc[0]):,}\nentry bins: {int(s.loc['in_0','spells']) if 'in_0' in s.index and pd.notna(s.loc['in_0','spells']) else 0} spells\n6+ support: {int(s.loc['in_6p','spells']) if 'in_6p' in s.index and pd.notna(s.loc['in_6p','spells']) else 0}",
            xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.2
        )

    fig.suptitle(
        "HR-style Event Study on 2005-2017 Headline Samples\n"
        "Region FE + Country-Year FE | SE clustered by spell | Dashed horizontal line = main static coefficient",
        fontsize=9, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {out_path}")


def main() -> None:
    print(">> Loading PLAD...")
    plad = load_plad()
    print(f"   {len(plad)} cleaned spells")

    print(">> Building event rows...")
    ev = build_event_rows(plad)
    print(f"   {len(ev):,} event rows")
    print("   Support by label:")
    print(ev["event_label"].value_counts().reindex(PLOT_ORDER).fillna(0).astype(int).to_string())

    print(">> Loading outcome-specific panels...")
    panels = load_outcome_panels(plad)
    for outcome, panel in panels.items():
        print(f"   {outcome:15s} {len(panel):,} obs")

    print(">> Attaching event labels...")
    panels = {k: attach_events(v, ev) for k, v in panels.items()}

    print(">> Running regressions...")
    results: dict[str, pd.DataFrame] = {}
    supports: dict[str, pd.DataFrame] = {}
    static_rows: dict[str, dict] = {}
    for outcome in OUTCOMES:
        print(f"   {outcome}...")
        r, s = run_regression(panels[outcome], outcome)
        results[outcome] = r
        supports[outcome] = s
        static_rows[outcome] = run_static(panels[outcome], outcome)
        print(r[["event_label", "coef", "se"]].to_string(index=False))

    pd.concat(results.values(), ignore_index=True).to_csv(
        DATA / "event_study_hr_style_2005_2017_results.csv", index=False
    )
    pd.concat(supports.values(), ignore_index=True).to_csv(
        DATA / "event_study_hr_style_2005_2017_support.csv", index=False
    )

    print(">> Plotting...")
    plot_results(
        results,
        supports,
        static_rows,
        OUT_DIR / "event_study_hr_style_2005_2017.pdf",
    )
    print("Done.")


if __name__ == "__main__":
    main()
