"""Statistical analysis of optimisation experiments (for papers/reports).

Design choices
--------------
* Only runs with status ``Success`` that are feasible enter cost statistics
  (``Partial`` runs were stopped early and are not comparable).  Feasibility
  is analysed separately.
* Final costs are rarely normal and often tied (many runs end on the same
  discrete design), so comparisons use the rank-based **Mann-Whitney U**
  test (= Wilcoxon rank-sum) for independent runs, with the **Vargha-
  Delaney A12** effect size.  The Wilcoxon *signed-rank* test is for
  paired data (same problem instances) and is therefore not used for
  independent runs on one network.
* p-values never stand alone: effect sizes and confidence intervals are
  reported, and the minimum attainable p-value is flagged when too few
  runs make significance impossible.
* Bootstrap resampling uses a fixed seed, so reports are reproducible.
"""

from __future__ import annotations

import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 12345
_Z95 = 1.959963984540054


# =====================================================================
#  Small helpers
# =====================================================================

def success_runs(runs):
    return [r for r in runs if r.status == "Success"]


def feasible_costs(runs) -> np.ndarray:
    """Costs of finished, feasible runs."""
    return np.array(
        [r.score for r in runs
         if r.status == "Success" and r.feasible and np.isfinite(r.score)],
        dtype=float,
    )


def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilson_interval(k: int, n: int, z: float = _Z95):
    """Wilson score interval for a proportion, returned in percent."""
    if n <= 0:
        return float("nan"), float("nan")
    ph = k / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def _mean_ci(x):
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    sem = float(sps.sem(x))
    if sem == 0.0:
        return m, m
    lo, hi = sps.t.interval(0.95, n - 1, loc=m, scale=sem)
    return float(lo), float(hi)


def _boot_median_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def _boot_median_diff_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI of median(y) - median(x)."""
    if len(x) < 5 or len(y) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), size=(n_boot, len(x)))
    iy = rng.integers(0, len(y), size=(n_boot, len(y)))
    d = np.median(y[iy], axis=1) - np.median(x[ix], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def a12_lower(x, y) -> float:
    """Vargha-Delaney A12 for minimisation: P(x < y) + 0.5 P(x == y).

    > 0.5 means ``x`` tends to reach a LOWER (better) cost than ``y``.
    """
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    return float(np.mean((x < y) + 0.5 * (x == y)))


def magnitude_label(a12: float) -> str:
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def min_possible_p(nx: int, ny: int) -> float:
    """Smallest two-sided exact Mann-Whitney p-value (no ties)."""
    if nx < 1 or ny < 1:
        return 1.0
    return min(1.0, 2.0 / math.comb(nx + ny, nx))


# =====================================================================
#  Per-algorithm description
# =====================================================================

def describe_runs(runs, reference_cost=None, tol_pct=1.0) -> dict:
    succ = success_runs(runs)
    costs = feasible_costs(runs)
    n = len(costs)
    nan = float("nan")

    d = {
        "n_runs": len(runs),
        "n_success": len(succ),
        "n_partial": sum(r.status == "Partial" for r in runs),
        "n_feasible": n,
    }
    d["feas_rate"] = 100.0 * n / len(succ) if succ else nan
    d["feas_ci_low"], d["feas_ci_high"] = wilson_interval(n, len(succ))

    times = [r.execution_time for r in succ if np.isfinite(r.execution_time)]
    d["time_mean"] = float(np.mean(times)) if times else nan
    d["time_std"] = float(np.std(times, ddof=1)) if len(times) > 1 else nan
    d["restarts_mean"] = (
        float(np.mean([len(getattr(r, "restart_gens", [])) for r in succ]))
        if succ else nan
    )

    # Evaluation budget (runs from files saved before NFE existed have 0)
    nfes = [getattr(r, "nfe", 0) for r in succ if getattr(r, "nfe", 0) > 0]
    sims = [getattr(r, "n_sim", 0) for r in succ if getattr(r, "nfe", 0) > 0]
    tobest = [getattr(r, "nfe_to_best", 0) for r in succ
              if r.feasible and getattr(r, "nfe_to_best", 0) > 0]
    d["nfe_mean"] = float(np.mean(nfes)) if nfes else nan
    d["sim_mean"] = float(np.mean(sims)) if sims else nan
    d["sim_std"] = float(np.std(sims, ddof=1)) if len(sims) > 1 else nan
    d["cache_hit_pct"] = (100.0 * (1 - np.sum(sims) / np.sum(nfes))
                          if nfes else nan)
    d["nfe_to_best_median"] = float(np.median(tobest)) if tobest else nan

    gens = []
    for r in succ:
        if r.feasible and r.convergence is not None:
            c = np.asarray(r.convergence, dtype=float)
            c = c[np.isfinite(c)]
            if c.size:
                final = c.min()
                tol = 1e-9 * max(1.0, abs(final))
                gens.append(int(np.argmax(c <= final + tol)) + 1)
    d["gen_to_best_median"] = float(np.median(gens)) if gens else nan

    keys = ["best", "worst", "mean", "median", "std", "cv", "q1", "q3",
            "iqr", "mean_ci_low", "mean_ci_high", "median_ci_low",
            "median_ci_high", "shapiro_p", "gap_best", "gap_mean",
            "success_rate_ref"]
    for k in keys:
        d[k] = nan
    if n == 0:
        return d

    d["best"], d["worst"] = float(costs.min()), float(costs.max())
    d["mean"], d["median"] = float(costs.mean()), float(np.median(costs))
    d["std"] = float(costs.std(ddof=1)) if n > 1 else nan
    d["cv"] = 100.0 * d["std"] / d["mean"] if n > 1 and d["mean"] else nan
    d["q1"], d["q3"] = (float(v) for v in np.percentile(costs, [25, 75]))
    d["iqr"] = d["q3"] - d["q1"]
    d["mean_ci_low"], d["mean_ci_high"] = _mean_ci(costs)
    d["median_ci_low"], d["median_ci_high"] = _boot_median_ci(costs)
    if n >= 3 and np.ptp(costs) > 0:
        d["shapiro_p"] = float(sps.shapiro(costs).pvalue)

    if reference_cost and reference_cost > 0:
        ref = float(reference_cost)
        d["gap_best"] = 100.0 * (d["best"] - ref) / ref
        d["gap_mean"] = 100.0 * (d["mean"] - ref) / ref
        within = int(np.sum(costs <= ref * (1 + tol_pct / 100.0)))
        d["success_rate_ref"] = 100.0 * within / len(succ)
    return d


# =====================================================================
#  Two-sample comparison
# =====================================================================

def compare_samples(x, y, label_x="A", label_y="B", alpha=ALPHA) -> dict:
    """Mann-Whitney U + A12 + effect/CI summaries for two cost samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nan = float("nan")
    res = {
        "x": label_x, "y": label_y, "n_x": len(x), "n_y": len(y),
        "median_x": float(np.median(x)) if len(x) else nan,
        "median_y": float(np.median(y)) if len(y) else nan,
        "U": nan, "p": nan, "p_adj": nan, "a12": nan, "cliffs_delta": nan,
        "magnitude": "", "median_diff": nan, "diff_ci_low": nan,
        "diff_ci_high": nan, "welch_p": nan,
        "p_min": min_possible_p(len(x), len(y)), "valid": False,
        "can_be_significant": False, "note": "",
    }
    res["can_be_significant"] = res["p_min"] <= alpha
    if len(x) < 2 or len(y) < 2:
        res["note"] = "Need at least 2 feasible runs per group."
        return res

    res["valid"] = True
    res["a12"] = a12_lower(x, y)
    res["cliffs_delta"] = 2 * res["a12"] - 1
    res["magnitude"] = magnitude_label(res["a12"])
    res["median_diff"] = res["median_y"] - res["median_x"]
    res["diff_ci_low"], res["diff_ci_high"] = _boot_median_diff_ci(x, y)

    if np.ptp(np.concatenate([x, y])) == 0:      # every value identical
        res["U"], res["p"] = len(x) * len(y) / 2.0, 1.0
        res["note"] = "All values are identical in both groups."
    else:
        u1, p = sps.mannwhitneyu(x, y, alternative="two-sided",
                                 method="auto")
        res["U"] = float(min(u1, len(x) * len(y) - u1))
        res["p"] = float(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.ptp(x) > 0 or np.ptp(y) > 0:
                res["welch_p"] = float(
                    sps.ttest_ind(x, y, equal_var=False).pvalue)
    res["p_adj"] = res["p"]
    return res


def verdict(res: dict, alpha=ALPHA) -> str:
    """One-sentence interpretation (uses the adjusted p-value)."""
    if not res["valid"]:
        return f"{res['x']} vs {res['y']}: {res['note']}"
    p = res["p_adj"]
    pt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    a = res["a12"]
    if p < alpha and a != 0.5:
        win, lose = (res["x"], res["y"]) if a > 0.5 else (res["y"], res["x"])
        a_win = a if a > 0.5 else 1 - a
        return (f"{win} reached significantly lower cost than {lose} "
                f"({pt}, A12 = {a_win:.2f}, {res['magnitude']} effect).")
    s = (f"No significant difference between {res['x']} and {res['y']} "
         f"({pt}, A12 = {a:.2f}, {res['magnitude']} effect).")
    if not res["can_be_significant"]:
        s += (f" With {res['n_x']} vs {res['n_y']} runs the smallest "
              f"attainable p-value is {res['p_min']:.3f}, so significance "
              f"is impossible — use more runs.")
    return s


def compare_feasibility(runs_x, runs_y) -> dict:
    """Fisher exact test on feasible/infeasible counts of finished runs."""
    sx, sy = success_runs(runs_x), success_runs(runs_y)
    fx, fy = sum(r.feasible for r in sx), sum(r.feasible for r in sy)
    table = [[fx, len(sx) - fx], [fy, len(sy) - fy]]
    if len(sx) == 0 or len(sy) == 0:
        p = float("nan")
    elif fx + fy == len(sx) + len(sy) or fx + fy == 0:
        p = 1.0                                   # no variation at all
    else:
        p = float(sps.fisher_exact(table)[1])
    return {"table": table, "p": p,
            "rate_x": 100.0 * fx / len(sx) if sx else float("nan"),
            "rate_y": 100.0 * fy / len(sy) if sy else float("nan")}


def adjust_and_explain(rows: list, alpha=ALPHA) -> list:
    """Holm-adjust the p-values of a family of comparisons (in place) and
    attach the one-sentence verdict to each row."""
    valid = [r for r in rows if r["valid"]]
    for r, padj in zip(valid, holm_adjust([r["p"] for r in valid])):
        r["p_adj"] = padj
    for r in rows:
        r["verdict"] = verdict(r, alpha)
    return rows


def pairwise_comparison(groups: dict, alpha=ALPHA) -> list:
    """All pairs of ``{label: costs}``; Holm-adjusted across the pairs."""
    return adjust_and_explain(
        [compare_samples(groups[a], groups[b], a, b, alpha)
         for a, b in combinations(groups, 2)], alpha)


# =====================================================================
#  Whole experiment
# =====================================================================

def analyze_experiment(result, reference_cost=None, tol_pct=1.0,
                       alpha=ALPHA) -> dict:
    algos = list(result.runs)
    desc = {a: describe_runs(result.runs[a], reference_cost, tol_pct)
            for a in algos}
    groups = {a: feasible_costs(result.runs[a]) for a in algos}
    pairs = pairwise_comparison(groups, alpha) if len(algos) > 1 else []
    feas = []
    for a, b in combinations(algos, 2):
        f = compare_feasibility(result.runs[a], result.runs[b])
        f["x"], f["y"] = a, b
        feas.append(f)

    notes = []
    for a in algos:
        n = desc[a]["n_feasible"]
        if n < 5:
            notes.append(
                f"{a}: only {n} feasible run(s). Confidence intervals and "
                "tests are unreliable below ~5 runs; papers typically use "
                "20-30 independent runs.")
    if getattr(result, "cancelled", False):
        notes.append("The experiment was cancelled; statistics use only "
                     "the runs that finished.")
    return {"descriptive": desc, "pairwise": pairs, "feasibility": feas,
            "notes": notes, "alpha": alpha,
            "reference_cost": reference_cost, "tol_pct": tol_pct}


def sweep_analysis(groups: dict, reference_label, alpha=ALPHA) -> dict:
    """Kruskal-Wallis across sweep points + each point vs a reference
    point (Mann-Whitney, Holm-adjusted)."""
    labels = [k for k, v in groups.items() if len(v) >= 1]
    kw = {"H": float("nan"), "p": float("nan"), "note": ""}
    usable = [groups[k] for k in labels]
    if len(usable) < 2:
        kw["note"] = "Need at least 2 points with feasible runs."
    elif np.ptp(np.concatenate(usable)) == 0:
        kw["p"], kw["note"] = 1.0, "All values are identical."
    else:
        try:
            h, p = sps.kruskal(*usable)
            kw["H"], kw["p"] = float(h), float(p)
        except ValueError as e:
            kw["note"] = str(e)

    rows = []
    if reference_label in groups:
        for k in groups:
            if k == reference_label:
                continue
            rows.append(compare_samples(groups[k], groups[reference_label],
                                        k, reference_label, alpha))
        adjust_and_explain(rows, alpha)
    return {"kruskal": kw, "vs_reference": rows}


# =====================================================================
#  Presentation helpers (shared by the GUI and the report)
# =====================================================================

def fmt_cost(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt(v, spec=".2f") -> str:
    return "—" if v is None or not np.isfinite(v) else format(v, spec)


def descriptive_dataframe(analysis: dict, formatted=True) -> pd.DataFrame:
    rows = []
    has_ref = bool(analysis.get("reference_cost"))
    for algo, d in analysis["descriptive"].items():
        if formatted:
            row = {
                "Algorithm": algo,
                "Runs (finished)": f"{d['n_success']}/{d['n_runs']}",
                "Feasible": f"{d['n_feasible']} "
                            f"({_fmt(d['feas_rate'], '.0f')}%)",
                "Best": fmt_cost(d["best"]),
                "Worst": fmt_cost(d["worst"]),
                "Mean": fmt_cost(d["mean"]),
                "Median": fmt_cost(d["median"]),
                "Std": fmt_cost(d["std"]),
                "CV %": _fmt(d["cv"], ".2f"),
                "IQR": fmt_cost(d["iqr"]),
                "95% CI mean": (f"[{fmt_cost(d['mean_ci_low'])}, "
                                f"{fmt_cost(d['mean_ci_high'])}]"),
                "95% CI median (boot)": (
                    f"[{fmt_cost(d['median_ci_low'])}, "
                    f"{fmt_cost(d['median_ci_high'])}]"),
                "Shapiro p": fmt_p(d["shapiro_p"]),
                "Gen. to best (median)": _fmt(d["gen_to_best_median"], ".0f"),
                "NFE per run": _fmt(d["nfe_mean"], ",.0f"),
                "NFE to best (median)": _fmt(d["nfe_to_best_median"], ",.0f"),
                "Hydraulic simulations (mean ± std)": (
                    f"{_fmt(d['sim_mean'], ',.0f')} ± "
                    f"{_fmt(d['sim_std'], ',.0f')}"),
                "Cache hits %": _fmt(d["cache_hit_pct"], ".1f"),
                "Time (s) mean±std": (f"{_fmt(d['time_mean'], '.1f')} ± "
                                      f"{_fmt(d['time_std'], '.1f')}"),
            }
            if has_ref:
                row["Gap best %"] = _fmt(d["gap_best"], ".2f")
                row["Gap mean %"] = _fmt(d["gap_mean"], ".2f")
                row[f"Success ≤{analysis['tol_pct']:g}% of ref"] = (
                    _fmt(d["success_rate_ref"], ".0f") + "%")
        else:
            row = {"Algorithm": algo, **d}
        rows.append(row)
    return pd.DataFrame(rows)


def tests_dataframe(pairs: list, formatted=True) -> pd.DataFrame:
    rows = []
    for r in pairs:
        if formatted:
            rows.append({
                "Comparison": f"{r['x']} vs {r['y']}",
                "n": f"{r['n_x']} / {r['n_y']}",
                "Median x / y": (f"{fmt_cost(r['median_x'])} / "
                                 f"{fmt_cost(r['median_y'])}"),
                "U": _fmt(r["U"], ".1f"),
                "p (Mann-Whitney)": fmt_p(r["p"]),
                "p (Holm)": fmt_p(r["p_adj"]),
                "A12 (P x<y)": _fmt(r["a12"], ".2f"),
                "Cliff's δ": _fmt(r["cliffs_delta"], "+.2f"),
                "Effect": r["magnitude"] or "—",
                "Median diff (y−x) [95% CI]": (
                    f"{fmt_cost(r['median_diff'])} "
                    f"[{fmt_cost(r['diff_ci_low'])}, "
                    f"{fmt_cost(r['diff_ci_high'])}]"),
                "p (Welch t)": fmt_p(r["welch_p"]),
            })
        else:
            rows.append({k: v for k, v in r.items() if k != "valid"})
    return pd.DataFrame(rows)


REFERENCES = [
    "Mann, H. B., & Whitney, D. R. (1947). Ann. Math. Stat., 18(1), 50-60.",
    "Vargha, A., & Delaney, H. D. (2000). J. Educ. Behav. Stat., 25(2), "
    "101-132.",
    "Cliff, N. (1993). Psychol. Bull., 114(3), 494-509.",
    "Holm, S. (1979). Scand. J. Stat., 6(2), 65-70.",
    "Wilson, E. B. (1927). J. Am. Stat. Assoc., 22(158), 209-212.",
    "Shapiro, S. S., & Wilk, M. B. (1965). Biometrika, 52(3-4), 591-611.",
    "Derrac, J., Garcia, S., Molina, D., & Herrera, F. (2011). Swarm Evol. "
    "Comput., 1(1), 3-18.",
]


def methods_text(alpha=ALPHA) -> str:
    return (
        "Only finished runs that satisfy all constraints enter the cost "
        "statistics; feasibility rates are reported separately (Wilson 95% "
        "intervals; Fisher's exact test to compare them). Costs are "
        "summarised by best, worst, mean, median, standard deviation, "
        "coefficient of variation and interquartile range, with Student-t "
        "95% confidence intervals for the mean and percentile-bootstrap "
        f"intervals ({N_BOOT:,} resamples, seed {BOOT_SEED}) for the median "
        "and for the median difference. Normality is checked with the "
        "Shapiro-Wilk test; because final costs are generally non-normal "
        "and often tied, algorithms are compared with the two-sided "
        "Mann-Whitney U test (Wilcoxon rank-sum; exact p-values for small "
        "samples without ties, otherwise a normal approximation with tie "
        f"and continuity corrections) at alpha = {alpha}. With more than "
        "two groups, p-values are Holm-adjusted. The effect size is the "
        "Vargha-Delaney A12 (probability that a random run of the first "
        "algorithm has lower cost than one of the second, ties counted "
        "half; |A12-0.5| < 0.06 negligible, < 0.14 small, < 0.21 medium, "
        "otherwise large) together with Cliff's delta = 2*A12 - 1. Welch's "
        "t-test is shown for reference only. Parameter sweeps use the "
        "Kruskal-Wallis test across settings with Holm-adjusted "
        "Mann-Whitney comparisons against a reference setting. The "
        "Wilcoxon signed-rank and Friedman tests are not used: they "
        "require paired data (e.g. several networks), whereas the runs "
        "here are independent. The number of function evaluations (NFE) "
        "counts every candidate design the algorithm submitted for "
        "evaluation (population size x generations, plus the initial "
        "population for GA and the re-evaluation after each restart); "
        "'hydraulic simulations' counts the EPANET runs actually executed, "
        "because repeated designs are served from a cache. NFE to best is "
        "the NFE at which the final best design was first found."
    )