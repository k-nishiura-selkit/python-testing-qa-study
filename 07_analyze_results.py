#!/usr/bin/env python3
"""
07_analyze_results.py

Generate the analyses reported in the paper from the already-created outputs of
repository collection and static analysis.

This script intentionally keeps the historical intermediate-file names produced
by the original pipeline. It does not clone repositories or call the GitHub API.

Inputs
------
data/processed/selected_repositories.csv
data/processed/03b_repository_history.csv
data/processed/04_repository_metrics.csv
data/processed/04_quality_metrics.csv

Outputs
-------
data/processed/07_results/
    tables/      CSV tables for all analyses reported in main(5).tex and the sensitivity analysis
    07_key_results.txt

Statistical reporting follows the final paper:
- binary outcomes: counts, proportions, percentage-point differences;
- RQ1 additionally reports a 95% CI for the percentage-point difference;
- skewed quantitative outcomes: median [Q1, Q3] and Cliff's delta;
- no risk ratios, null-hypothesis p-values, or multiple-testing corrections.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


SELECTED = Path("data/processed/selected_repositories.csv")
HISTORY = Path("data/processed/03b_repository_history.csv")
METRICS = Path("data/processed/04_repository_metrics.csv")
QUALITY = Path("data/processed/04_quality_metrics.csv")

OUT = Path("data/processed/07_results")
TBL = OUT / "tables"

COHORTS = ["pre_ai", "contemporary"]
LABEL = {"pre_ai": "Pre-AI", "contemporary": "Contemporary"}
CUTOFF = "2022-11-30"

TEST_DEFS = [
    ("Broad", "has_tests_broad"),
    ("Conventional", "has_tests_conventional"),
    ("Strict", "has_tests_strict"),
]

RQ2_QUANTITY = [
    ("Test artifact files", "test_artifact_file_count"),
    ("Test artifact LOC", "test_artifact_loc"),
    ("Test LOC ratio", "test_artifact_loc_ratio"),
]
RQ2_BINARY = [
    ("Structure", "Production-code reference", "has_production_reference_test"),
    ("Structure", "Explicit oracle", "has_explicit_oracle"),
    ("Testing practice", "pytest", "testcode_uses_pytest"),
    ("Testing practice", "unittest", "testcode_uses_unittest"),
    ("Testing practice", "Exception testing", "uses_exception_testing"),
    ("Testing practice", "Mocking", "uses_mocking"),
    ("Automation", "CI test execution", "has_ci_test_execution"),
    ("Automation", "Coverage tooling", "has_coverage_tooling"),
]

RQ4_QA = [
    ("Linting", "has_linting"),
    ("Static type checking", "has_type_checking"),
    ("Formatting", "has_formatting"),
    ("Any non-testing QA", "has_other_qa"),
    ("Automated non-testing QA", "has_other_qa_automated"),
]

LOC_BINS = [-np.inf, 109, 332, 1036, np.inf]
LOC_LABELS = ["<=109", "110-332", "333-1036", ">1036"]
COMMIT_BINS = [0, 1, 5, 20, np.inf]
COMMIT_LABELS = ["1", "2-5", "6-20", ">20"]
SPAN_BINS = [-np.inf, 1, 7, 30, np.inf]
SPAN_LABELS = ["<=1 day", ">1-7 days", ">7-30 days", ">30 days"]
CONTRIBUTOR_LABELS = ["1", "2", "3-5"]


def require(df: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns in {source}: " + ", ".join(missing))


def as_binary(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    """Cliff's delta for y relative to x: positive means y tends to be larger."""
    a = np.sort(np.asarray(pd.Series(x).dropna(), dtype=float))
    b = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    less = greater = 0
    for v in b:
        less += np.searchsorted(a, v, side="left")
        greater += len(a) - np.searchsorted(a, v, side="right")
    return (less - greater) / (len(a) * len(b))


def difference_ci(k_con: int, n_con: int, k_pre: int, n_pre: int) -> tuple[float, float, float]:
    """Unpooled normal-approximation 95% CI for a difference in proportions."""
    p_con = k_con / n_con
    p_pre = k_pre / n_pre
    diff = p_con - p_pre
    se = math.sqrt(p_con * (1 - p_con) / n_con + p_pre * (1 - p_pre) / n_pre)
    z = norm.ppf(0.975)
    return diff, diff - z * se, diff + z * se


def load_data() -> pd.DataFrame:
    for path in [SELECTED, HISTORY, METRICS, QUALITY]:
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    selected = pd.read_csv(SELECTED)
    history = pd.read_csv(HISTORY).drop_duplicates("repo_id", keep="last")
    metrics = pd.read_csv(METRICS)
    quality = pd.read_csv(QUALITY)

    require(selected, {"repo_id", "human_contributors"}, SELECTED)
    require(history, {
        "repo_id", "history_status", "commit_count", "development_span_days",
        "is_single_commit", "last_commit_date",
    }, HISTORY)
    require(metrics, {
        "repo_id", "cohort", "analysis_status", "python_loc",
        "has_tests_broad", "has_tests_conventional", "has_tests_strict",
        "test_artifact_file_count", "test_artifact_loc", "test_artifact_loc_ratio",
        "testcode_uses_pytest", "testcode_uses_unittest", "uses_exception_testing",
        "uses_mock", "uses_monkeypatch", "has_ci_test_execution", "has_coverage_tooling",
    }, METRICS)
    require(quality, {
        "repo_id", "quality_status", "has_production_reference_test", "has_explicit_oracle",
        "has_linting", "has_type_checking", "has_formatting", "has_other_qa",
        "has_other_qa_automated", "ai_explicit",
    }, QUALITY)

    metrics = metrics[metrics["analysis_status"] == "ok"].copy()
    quality = quality[quality["quality_status"] == "ok"].copy()
    selected = selected[["repo_id", "human_contributors"]].drop_duplicates("repo_id", keep="last")

    df = (
        metrics.merge(selected, on="repo_id", how="left", validate="one_to_one")
        .merge(
            history[["repo_id", "history_status", "commit_count", "development_span_days",
                     "is_single_commit", "last_commit_date"]],
            on="repo_id", how="left", validate="one_to_one",
        )
        .merge(
            quality[["repo_id", "has_production_reference_test", "has_explicit_oracle",
                     "has_linting", "has_type_checking", "has_formatting", "has_other_qa",
                     "has_other_qa_automated", "ai_explicit"]],
            on="repo_id", how="left", validate="one_to_one",
        )
    )

    if len(df) != 20000:
        raise SystemExit(f"Expected 20,000 analyzable repositories, found {len(df):,}.")
    for cohort in COHORTS:
        n = int((df["cohort"] == cohort).sum())
        if n != 10000:
            raise SystemExit(f"Expected 10,000 {cohort} repositories, found {n:,}.")

    binary_cols = [
        "has_tests_broad", "has_tests_conventional", "has_tests_strict",
        "testcode_uses_pytest", "testcode_uses_unittest", "uses_exception_testing",
        "uses_mock", "uses_monkeypatch", "has_ci_test_execution", "has_coverage_tooling",
        "has_production_reference_test", "has_explicit_oracle", "has_linting",
        "has_type_checking", "has_formatting", "has_other_qa", "has_other_qa_automated",
        "ai_explicit",
    ]
    for col in binary_cols:
        df[col] = as_binary(df[col])
    df["uses_mocking"] = ((df["uses_mock"] > 0) | (df["uses_monkeypatch"] > 0)).astype(int)

    return df


def binary_comparison(pre: pd.DataFrame, con: pd.DataFrame, defs, include_diff_ci=False) -> pd.DataFrame:
    rows = []
    for item in defs:
        if len(item) == 2:
            label, col = item
            section = None
        else:
            section, label, col = item
        kp, kc = int(pre[col].sum()), int(con[col].sum())
        np_, nc = len(pre), len(con)
        rp, rc = kp / np_, kc / nc
        row = {
            "metric": label, "column": col,
            "n_pre_ai": np_, "count_pre_ai": kp, "rate_pre_ai": rp,
            "n_contemporary": nc, "count_contemporary": kc, "rate_contemporary": rc,
            "difference": rc - rp,
        }
        if section is not None:
            row["section"] = section
        if include_diff_ci:
            diff, lo, hi = difference_ci(kc, nc, kp, np_)
            row.update(difference=diff, difference_ci_low=lo, difference_ci_high=hi)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_rq1(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df.cohort == "pre_ai"]
    con = df[df.cohort == "contemporary"]
    return binary_comparison(pre, con, TEST_DEFS, include_diff_ci=True)


def analyze_rq2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tested = df[df["has_tests_conventional"] == 1].copy()
    pre = tested[tested.cohort == "pre_ai"]
    con = tested[tested.cohort == "contemporary"]

    quantity_rows = []
    for label, col in RQ2_QUANTITY:
        x = pd.to_numeric(pre[col], errors="coerce").dropna()
        y = pd.to_numeric(con[col], errors="coerce").dropna()
        quantity_rows.append({
            "metric": label, "column": col,
            "n_pre_ai": len(x), "pre_ai_median": x.median(),
            "pre_ai_q1": x.quantile(.25), "pre_ai_q3": x.quantile(.75),
            "n_contemporary": len(y), "contemporary_median": y.median(),
            "contemporary_q1": y.quantile(.25), "contemporary_q3": y.quantile(.75),
            "cliffs_delta": cliffs_delta(x, y),
        })
    quantity = pd.DataFrame(quantity_rows)
    binary = binary_comparison(pre, con, RQ2_BINARY)

    overlap_rows = []
    for cohort in COHORTS:
        g = tested[tested.cohort == cohort]
        both = int(((g.has_ci_test_execution == 1) & (g.has_coverage_tooling == 1)).sum())
        overlap_rows.append({
            "cohort": LABEL[cohort], "repositories_with_conventional_tests": len(g),
            "ci_and_coverage_count": both, "ci_and_coverage_rate": both / len(g),
        })
    return quantity, binary, pd.DataFrame(overlap_rows)


def analyze_framework_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Joint pytest/unittest composition among Conventional test repositories."""
    tested = df[df["has_tests_conventional"] == 1].copy()
    p = tested["testcode_uses_pytest"] == 1
    u = tested["testcode_uses_unittest"] == 1
    tested["framework_group"] = np.select(
        [p & ~u, ~p & u, p & u],
        ["pytest only", "unittest only", "pytest + unittest"],
        default="neither detected",
    )
    order = ["pytest only", "unittest only", "pytest + unittest", "neither detected"]
    rows = []
    for cohort in COHORTS:
        g = tested[tested.cohort == cohort]
        for group in order:
            k = int((g.framework_group == group).sum())
            rows.append({
                "cohort": LABEL[cohort], "framework_group": group,
                "count": k, "test_repositories": len(g),
                "rate": k / len(g) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def prevalence_by_group(df: pd.DataFrame, group_col: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for group in order:
        vals = {}
        for cohort in COHORTS:
            g = df[(df.cohort == cohort) & (df[group_col].astype(str) == str(group))]
            n = len(g)
            k = int(g.has_tests_conventional.sum())
            vals[cohort] = (n, k, k / n if n else np.nan)
        diff = vals["contemporary"][2] - vals["pre_ai"][2]
        rows.append({
            "group": group,
            "n_pre_ai": vals["pre_ai"][0], "test_repositories_pre_ai": vals["pre_ai"][1],
            "test_prevalence_pre_ai": vals["pre_ai"][2],
            "n_contemporary": vals["contemporary"][0],
            "test_repositories_contemporary": vals["contemporary"][1],
            "test_prevalence_contemporary": vals["contemporary"][2],
            "difference": diff,
        })
    return pd.DataFrame(rows)


def analyze_ai_explicit(df: pd.DataFrame) -> pd.DataFrame:
    con = df[df.cohort == "contemporary"].copy()
    ai = con[con.ai_explicit == 1].copy()
    rows = []

    def binary(label, col, tested_only=False):
        base = con[con.has_tests_conventional == 1] if tested_only else con
        sub = ai[ai.has_tests_conventional == 1] if tested_only else ai
        rows.append({
            "metric": label,
            "contemporary_n": len(base), "contemporary_count": int(base[col].sum()),
            "contemporary_rate": base[col].mean() if len(base) else np.nan,
            "ai_explicit_n": len(sub), "ai_explicit_count": int(sub[col].sum()),
            "ai_explicit_rate": sub[col].mean() if len(sub) else np.nan,
        })

    binary("Conventional tests", "has_tests_conventional")
    binary("Strict tests", "has_tests_strict")
    binary("Other QA", "has_other_qa")
    binary("Production-code reference", "has_production_reference_test", tested_only=True)
    binary("Explicit oracle", "has_explicit_oracle", tested_only=True)

    for label, col in [
        ("Python LOC", "python_loc"), ("Commit count", "commit_count"),
        ("Development span days", "development_span_days"),
    ]:
        a = pd.to_numeric(con[col], errors="coerce").dropna()
        b = pd.to_numeric(ai[col], errors="coerce").dropna()
        rows.append({
            "metric": label, "contemporary_n": len(a), "contemporary_median": a.median(),
            "ai_explicit_n": len(b), "ai_explicit_median": b.median(),
        })
    return pd.DataFrame(rows)


def analyze_rq3(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.copy()
    work["loc_group"] = pd.cut(pd.to_numeric(work.python_loc, errors="coerce"),
                               bins=LOC_BINS, labels=LOC_LABELS, include_lowest=True).astype(str)

    hist_ok = work.history_status.isin(["ok", "single_commit"])
    history = work[hist_ok].copy()
    history["commit_group"] = pd.cut(pd.to_numeric(history.commit_count, errors="coerce"),
                                     bins=COMMIT_BINS, labels=COMMIT_LABELS,
                                     include_lowest=True).astype(str)
    history["span_group"] = pd.cut(pd.to_numeric(history.development_span_days, errors="coerce"),
                                   bins=SPAN_BINS, labels=SPAN_LABELS,
                                   include_lowest=True).astype(str)

    hc = pd.to_numeric(work.human_contributors, errors="coerce")
    if hc.isna().any() or not hc.dropna().isin([1, 2, 3, 4, 5]).all():
        raise SystemExit("human_contributors must be present and within 1-5 for all repositories.")
    work["contributor_group"] = np.select(
        [hc == 1, hc == 2, hc.between(3, 5)], CONTRIBUTOR_LABELS, default=""
    )

    dist_rows = []
    for cohort in COHORTS:
        g = work[work.cohort == cohort]
        for group in CONTRIBUTOR_LABELS:
            n = int((g.contributor_group == group).sum())
            dist_rows.append({"cohort": LABEL[cohort], "group": group,
                              "repositories": n, "share": n / len(g)})

    return {
        "loc": prevalence_by_group(work, "loc_group", LOC_LABELS),
        "commits": prevalence_by_group(history, "commit_group", COMMIT_LABELS),
        "span": prevalence_by_group(history, "span_group", SPAN_LABELS),
        "contributor_distribution": pd.DataFrame(dist_rows),
        "contributors": prevalence_by_group(work, "contributor_group", CONTRIBUTOR_LABELS),
        "ai_explicit": analyze_ai_explicit(work),
    }


def qa_combination(df: pd.DataFrame, qa_col: str, automated=False) -> pd.DataFrame:
    cats = ([
        ("Neither", 0, 0), ("Tests only", 1, 0),
        ("Automated QA only", 0, 1), ("Tests + automated QA", 1, 1),
    ] if automated else [
        ("Neither", 0, 0), ("Tests only", 1, 0),
        ("Other QA only", 0, 1), ("Tests + other QA", 1, 1),
    ])
    rows = []
    for cohort in COHORTS:
        g = df[df.cohort == cohort]
        for label, t, q in cats:
            k = int(((g.has_tests_conventional == t) & (g[qa_col] == q)).sum())
            rows.append({"cohort": LABEL[cohort], "category": label,
                         "count": k, "repositories": len(g), "rate": k / len(g)})
    return pd.DataFrame(rows)


def analyze_qa_absence_by_loc(df: pd.DataFrame) -> pd.DataFrame:
    """Repositories with neither Conventional tests nor non-testing QA, by RQ3 LOC stratum."""
    work = df.copy()
    work["loc_group"] = pd.cut(
        pd.to_numeric(work.python_loc, errors="coerce"),
        bins=LOC_BINS, labels=LOC_LABELS, include_lowest=True,
    ).astype(str)
    rows = []
    for group in LOC_LABELS:
        for cohort in COHORTS:
            g = work[(work.cohort == cohort) & (work.loc_group == group)]
            k = int(((g.has_tests_conventional == 0) & (g.has_other_qa == 0)).sum())
            rows.append({
                "cohort": LABEL[cohort], "python_loc_group": group,
                "neither_count": k, "repositories": len(g),
                "neither_rate": k / len(g) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def analyze_rq4(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre = df[df.cohort == "pre_ai"]
    con = df[df.cohort == "contemporary"]
    prevalence = binary_comparison(pre, con, RQ4_QA)
    return (
        prevalence,
        qa_combination(df, "has_other_qa", automated=False),
        qa_combination(df, "has_other_qa_automated", automated=True),
    )


def analyze_sensitivity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(CUTOFF, tz="UTC")
    last = pd.to_datetime(df.last_commit_date, errors="coerce", utc=True)
    valid = df.history_status.isin(["ok", "single_commit"]) & last.notna()
    pre = df[(df.cohort == "pre_ai") & valid & (last < cutoff)].copy()
    con = df[df.cohort == "contemporary"].copy()

    rq1 = binary_comparison(pre, con, TEST_DEFS, include_diff_ci=True)
    pretest = pre[pre.has_tests_conventional == 1]
    contest = con[con.has_tests_conventional == 1]

    qty_rows = []
    for label, col in RQ2_QUANTITY:
        x = pd.to_numeric(pretest[col], errors="coerce").dropna()
        y = pd.to_numeric(contest[col], errors="coerce").dropna()
        qty_rows.append({
            "metric": label,
            "pre_ai_median": x.median(), "pre_ai_q1": x.quantile(.25), "pre_ai_q3": x.quantile(.75),
            "contemporary_median": y.median(), "contemporary_q1": y.quantile(.25),
            "contemporary_q3": y.quantile(.75), "cliffs_delta": cliffs_delta(x, y),
        })

    rq2 = binary_comparison(pretest, contest, RQ2_BINARY)
    rq4 = binary_comparison(pre, con, RQ4_QA)
    population = pd.DataFrame([{
        "pre_ai_main": int((df.cohort == "pre_ai").sum()),
        "pre_ai_restricted": len(pre), "contemporary": len(con), "cutoff_exclusive": CUTOFF,
    }])
    return {
        "population": population, "rq1": rq1, "rq2_quantity": pd.DataFrame(qty_rows),
        "rq2_binary": rq2, "rq4": rq4,
        "rq4_combinations": qa_combination(pd.concat([pre, con]), "has_other_qa", False),
        "rq4_automated_combinations": qa_combination(pd.concat([pre, con]), "has_other_qa_automated", True),
    }


def fmt_binary(r) -> str:
    return (f"Pre-AI {int(r.count_pre_ai):,}/{int(r.n_pre_ai):,} ({100*r.rate_pre_ai:.2f}%)   "
            f"Contemporary {int(r.count_contemporary):,}/{int(r.n_contemporary):,} "
            f"({100*r.rate_contemporary:.2f}%)   Diff {100*r.difference:+.2f} pp")


def build_summary(rq1, rq2q, rq2b, rq2overlap, framework, rq3, rq4, combos, auto_combos, absence_loc, sensitivity) -> str:
    L = ["=" * 78, "PAPER RESULT ANALYSIS", "=" * 78, "", "RQ1: TEST PREVALENCE", "-" * 78]
    for _, r in rq1.iterrows():
        ci = f" [{100*r.difference_ci_low:+.2f}, {100*r.difference_ci_high:+.2f}]" if r.metric == "Conventional" else ""
        L.append(f"{r.metric:<14} {fmt_binary(r)}{ci}")

    L += ["", "RQ2: TESTING FOUNDATIONS", "-" * 78, "Quantity:"]
    for _, r in rq2q.iterrows():
        L.append(f"  {r.metric:<22} Pre-AI {r.pre_ai_median:.2f} [{r.pre_ai_q1:.2f}, {r.pre_ai_q3:.2f}]   "
                 f"Contemporary {r.contemporary_median:.2f} [{r.contemporary_q1:.2f}, {r.contemporary_q3:.2f}]   "
                 f"Cliff's delta {r.cliffs_delta:.3f}")
    for section in ["Structure", "Testing practice", "Automation"]:
        L += ["", section + ":"]
        for _, r in rq2b[rq2b.section == section].iterrows():
            L.append(f"  {r.metric:<28} {fmt_binary(r)}")
    L += ["", "CI + coverage overlap:"]
    for _, r in rq2overlap.iterrows():
        L.append(f"  {r.cohort:<14} {int(r.ci_and_coverage_count):,}/{int(r.repositories_with_conventional_tests):,} "
                 f"({100*r.ci_and_coverage_rate:.2f}%)")
    L += ["", "pytest / unittest composition:"]
    for group in ["pytest only", "unittest only", "pytest + unittest", "neither detected"]:
        a = framework[(framework.cohort == "Pre-AI") & (framework.framework_group == group)].iloc[0]
        b = framework[(framework.cohort == "Contemporary") & (framework.framework_group == group)].iloc[0]
        L.append(f"  {group:<24} Pre-AI {int(a['count']):,}/{int(a.test_repositories):,} ({100*a.rate:.2f}%)   "
                 f"Contemporary {int(b['count']):,}/{int(b.test_repositories):,} ({100*b.rate:.2f}%)")

    L += ["", "RQ3: REPOSITORY CHARACTERISTICS", "-" * 78]
    for title, key in [("Python LOC", "loc"), ("Commit count", "commits"),
                       ("Development span", "span"), ("Human contributors", "contributors")]:
        L += ["", title + ":"]
        for _, r in rq3[key].iterrows():
            L.append(f"  {r.group:<12} Pre-AI {100*r.test_prevalence_pre_ai:.2f}%   "
                     f"Contemporary {100*r.test_prevalence_contemporary:.2f}%   Diff {100*r.difference:+.2f} pp")
    L += ["", "Contributor distribution:"]
    for group in CONTRIBUTOR_LABELS:
        a = rq3["contributor_distribution"][(rq3["contributor_distribution"].cohort == "Pre-AI") &
                                             (rq3["contributor_distribution"].group == group)].iloc[0]
        b = rq3["contributor_distribution"][(rq3["contributor_distribution"].cohort == "Contemporary") &
                                             (rq3["contributor_distribution"].group == group)].iloc[0]
        L.append(f"  {group:<4} Pre-AI {int(a.repositories):,} ({100*a.share:.2f}%)   "
                 f"Contemporary {int(b.repositories):,} ({100*b.share:.2f}%)")

    L += ["", "AI-explicit subset (exploratory):"]
    ai = rq3["ai_explicit"].set_index("metric")
    for metric in ["Conventional tests", "Strict tests", "Other QA"]:
        r = ai.loc[metric]
        L.append(
            f"  {metric:<24} Contemporary {int(r.contemporary_count):,}/{int(r.contemporary_n):,} "
            f"({100*r.contemporary_rate:.2f}%)   "
            f"AI-explicit {int(r.ai_explicit_count):,}/{int(r.ai_explicit_n):,} "
            f"({100*r.ai_explicit_rate:.2f}%)"
        )
    for metric in ["Python LOC", "Commit count", "Development span days"]:
        r = ai.loc[metric]
        L.append(
            f"  {metric:<24} Contemporary median {r.contemporary_median:.2f} (n={int(r.contemporary_n):,})   "
            f"AI-explicit median {r.ai_explicit_median:.2f} (n={int(r.ai_explicit_n):,})"
        )

    L += ["", "RQ4: NON-TESTING QA", "-" * 78]
    for _, r in rq4.iterrows():
        L.append(f"{r.metric:<28} {fmt_binary(r)}")
    for title, raw in [("Tests x other QA", combos), ("Tests x automated QA", auto_combos)]:
        L += ["", title + ":"]
        for cat in raw.category.drop_duplicates():
            a = raw[(raw.cohort == "Pre-AI") & (raw.category == cat)].iloc[0]
            b = raw[(raw.cohort == "Contemporary") & (raw.category == cat)].iloc[0]
            L.append(f"  {cat:<24} Pre-AI {100*a.rate:.2f}%   Contemporary {100*b.rate:.2f}%")
    L += ["", "Neither tests nor non-testing QA by Python LOC:"]
    for group in LOC_LABELS:
        a = absence_loc[(absence_loc.cohort == "Pre-AI") & (absence_loc.python_loc_group == group)].iloc[0]
        b = absence_loc[(absence_loc.cohort == "Contemporary") & (absence_loc.python_loc_group == group)].iloc[0]
        L.append(f"  {group:<12} Pre-AI {int(a.neither_count):,}/{int(a.repositories):,} ({100*a.neither_rate:.2f}%)   "
                 f"Contemporary {int(b.neither_count):,}/{int(b.repositories):,} ({100*b.neither_rate:.2f}%)")

    L += ["", "SENSITIVITY: PRE-AI LAST COMMIT < " + CUTOFF, "-" * 78,
          f"Restricted Pre-AI repositories: {int(sensitivity['population'].iloc[0].pre_ai_restricted):,}"]
    for _, r in sensitivity["rq1"].iterrows():
        L.append(f"{r.metric:<14} {fmt_binary(r)}")
    L += ["", "Notes:", "- AI-explicit results are exploratory and do not estimate overall AI use.",
          "- The sensitivity cutoff uses default-branch last_commit_date.",
          "- No causal interpretation of period differences is made.", ""]
    return "\n".join(L)


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    df = load_data()

    rq1 = analyze_rq1(df)
    rq2q, rq2b, rq2overlap = analyze_rq2(df)
    framework = analyze_framework_composition(df)
    rq3 = analyze_rq3(df)
    rq4, combos, auto_combos = analyze_rq4(df)
    absence_loc = analyze_qa_absence_by_loc(df)
    sensitivity = analyze_sensitivity(df)

    rq1.to_csv(TBL / "rq1_test_prevalence.csv", index=False)
    rq2q.to_csv(TBL / "rq2_test_quantity.csv", index=False)
    rq2b.to_csv(TBL / "rq2_test_foundations.csv", index=False)
    rq2overlap.to_csv(TBL / "rq2_ci_coverage_overlap.csv", index=False)
    framework.to_csv(TBL / "rq2_framework_composition.csv", index=False)
    rq3["loc"].to_csv(TBL / "rq3_test_prevalence_by_loc.csv", index=False)
    rq3["commits"].to_csv(TBL / "rq3_test_prevalence_by_commits.csv", index=False)
    rq3["span"].to_csv(TBL / "rq3_test_prevalence_by_span.csv", index=False)
    rq3["contributor_distribution"].to_csv(TBL / "rq3_contributor_distribution.csv", index=False)
    rq3["contributors"].to_csv(TBL / "rq3_test_prevalence_by_contributors.csv", index=False)
    rq3["ai_explicit"].to_csv(TBL / "rq3_ai_explicit.csv", index=False)
    rq4.to_csv(TBL / "rq4_qa_prevalence.csv", index=False)
    combos.to_csv(TBL / "rq4_test_qa_combination.csv", index=False)
    auto_combos.to_csv(TBL / "rq4_test_automated_qa_combination.csv", index=False)
    absence_loc.to_csv(TBL / "rq4_neither_test_nor_qa_by_loc.csv", index=False)

    for name, table in sensitivity.items():
        table.to_csv(TBL / f"sensitivity_{name}.csv", index=False)

    text = build_summary(rq1, rq2q, rq2b, rq2overlap, framework, rq3, rq4, combos, auto_combos, absence_loc, sensitivity)
    (OUT / "07_key_results.txt").write_text(text, encoding="utf-8")
    print(text)
    print("=" * 78)
    print(f"Tables:  {TBL}")
    print(f"Summary: {OUT / '07_key_results.txt'}")


if __name__ == "__main__":
    main()
