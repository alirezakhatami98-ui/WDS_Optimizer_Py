"""Command-line interface for WDS Optimizer.

Entry point: ``wds-optimize`` (installed via ``[project.scripts]``).
Also runnable as ``python -m wds_optimizer.cli`` — useful for CI and
for testing without installing the console script.

Subcommands
-----------
- ``run``       — run an optimisation on user-supplied files
- ``validate``  — check input files without running
- ``info``      — print environment metadata (versions, optional deps)
- ``benchmark`` — run on a built-in benchmark network

Exit codes
----------
0   success, feasible design found
1   success, but no feasible design was found
2   invalid input (missing file, bad value, constraint conflict)
3   hydraulics error (all runs failed)
4   cancelled (Ctrl+C)
10  unexpected error

Constraint units
----------------
By default, ``--pmin 30 --vmax 2.5`` means 30 in the network's own
pressure unit and 2.5 in its own velocity unit, exactly as the .inp
file defines.  The CLI detects those units automatically.  Pass
``--units si`` to interpret the values as metres and m/s instead.
This matches the API's "SI always" rule while keeping CLI users happy
(who typically type the numbers they see in the .inp file).
"""
from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from wds_optimizer.api import (
    OptimizationRequest,
    OptimizationResult,
    run_optimization,
)
from wds_optimizer.hydraulics.units import (
    detect_network_units,
    pressure_to_si,
    velocity_to_si,
)


# ---------------------------------------------------------------------------
#  Exit codes
# ---------------------------------------------------------------------------

EXIT_OK            = 0
EXIT_INFEASIBLE    = 1
EXIT_INVALID_INPUT = 2
EXIT_HYDRAULICS    = 3
EXIT_CANCELLED     = 4
EXIT_UNEXPECTED    = 10


# ---------------------------------------------------------------------------
#  SIGINT: graceful first, hard second
# ---------------------------------------------------------------------------

class _CancelFlag:
    def __init__(self):
        self.cancelled = False

    def __call__(self) -> bool:
        return self.cancelled


def _install_sigint_handler(flag: _CancelFlag) -> None:
    """First Ctrl+C: graceful stop.  Second Ctrl+C: immediate abort."""
    prev = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        if flag.cancelled:
            signal.signal(signal.SIGINT, prev)
            raise KeyboardInterrupt
        flag.cancelled = True
        print(
            "\n[wds-optimize] stopping gracefully "
            "(press Ctrl+C again to abort immediately)",
            file=sys.stderr, flush=True,
        )

    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
#  Progress reporting (no external dependency required)
# ---------------------------------------------------------------------------

class _Progress:
    """Minimal progress reporter; uses tqdm if available, else prints."""

    def __init__(self, *, quiet: bool):
        self.quiet = quiet
        self._tqdm_bar = None
        self._tqdm_mod = None
        self._last_bucket = -1
        self.algos: list[str] = []
        self.n_runs = 1
        self.max_gen = 1
        if not quiet:
            try:
                # tqdm is optional: if not installed, we fall back to
                # plain 10%-bucket printing below.  ``type: ignore``
                # silences Pylance's "could not be resolved from source"
                # warning in environments where the user has not
                # installed the [cli] extra; it does NOT affect runtime.
                from tqdm import tqdm   # type: ignore[import-untyped]
                self._tqdm_mod = tqdm
            except ImportError:
                self._tqdm_mod = None

    def start(self, algos, n_runs: int, max_gen: int) -> None:
        if self.quiet:
            return
        self.algos = list(algos)
        self.n_runs = n_runs
        self.max_gen = max_gen
        if self._tqdm_mod is not None:
            self._tqdm_bar = self._tqdm_mod(
                total=len(self.algos) * n_runs,
                desc="Optimising", unit="run", ncols=80,
            )

    def update(self, info: dict) -> None:
        if self.quiet:
            return
        algo = info.get("Algorithm", "")
        run_idx = int(info.get("RunIndex", 1))
        n_runs = int(info.get("NumberOfRuns", self.n_runs))
        gen = int(info.get("Generation", 0))
        max_gen = int(info.get("MaxGenerations", self.max_gen))
        a_idx = self.algos.index(algo) if algo in self.algos else 0

        frac = (a_idx * n_runs + (run_idx - 1)
                + gen / max(1, max_gen)) / max(1, len(self.algos) * n_runs)

        if self._tqdm_bar is not None:
            target_int = int(min(1.0, frac) * (len(self.algos) * n_runs))
            if target_int > self._tqdm_bar.n:
                self._tqdm_bar.update(target_int - self._tqdm_bar.n)
            return

        # Fallback: print at 10%-buckets only
        bucket = int(frac * 10)
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            best = info.get("BestCost", 0.0)
            print(
                f"  [{int(frac * 100):3d}%] {algo} run {run_idx}/{n_runs}  "
                f"gen {gen}/{max_gen}  best={best:,.0f}",
                flush=True,
            )

    def finish(self) -> None:
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _detect_display_units(inp_path: str) -> dict:
    from wds_optimizer.api import _parse_network
    wn = _parse_network(inp_path)
    return detect_network_units(wn)


def _constraints_from_args(args, units: dict) -> dict:
    """Build the SI constraint dict from CLI args + detected units."""
    use_si = (getattr(args, "units", "auto") == "si")

    def _val(raw, kind):
        if raw is None:
            return None
        if use_si:
            return float(raw)
        if kind == "pressure":
            return pressure_to_si(float(raw), units)
        return velocity_to_si(float(raw), units)

    pmin = _val(args.pmin, "pressure")
    pmax = _val(args.pmax, "pressure")
    vmin = _val(args.vmin, "velocity")
    vmax = _val(args.vmax, "velocity")

    if all(x is None for x in (pmin, pmax, vmin, vmax)):
        raise ValueError(
            "specify at least one constraint: "
            "--pmin, --pmax, --vmin, --vmax"
        )

    return {
        "Pmin": {"enabled": pmin is not None,
                 "value": float(pmin) if pmin is not None else 30.0},
        "Pmax": {"enabled": pmax is not None,
                 "value": float(pmax) if pmax is not None else 100.0},
        "Vmin": {"enabled": vmin is not None,
                 "value": float(vmin) if vmin is not None else 0.0},
        "Vmax": {"enabled": vmax is not None,
                 "value": float(vmax) if vmax is not None else 2.5},
    }


def _build_request_from_args(args) -> OptimizationRequest:
    units = _detect_display_units(args.inp)
    constraints = _constraints_from_args(args, units)
    return OptimizationRequest(
        inp_path=args.inp,
        d_path=args.d,
        cost_path=args.cost,
        diam_unit=args.diam_unit,
        algorithms=args.algo or ["GA"],
        n_runs=args.n_runs,
        seed=args.seed,
        ns=args.ns,
        max_gen=args.max_gen,
        constraints=constraints,
        fixed_pipes=args.fixed or "",
    )


def _print_summary(result: OptimizationResult) -> None:
    from wds_optimizer.experiments.runner import summarize_experiment

    print()
    print("=" * 62)
    status = "feasible" if result.best_feasible else "INFEASIBLE"
    print(f"  BEST RESULT ({status})")
    print("=" * 62)
    print(f"  Algorithm:    {result.best_algo} (run {result.best_run_index})")
    print(f"  Cost:         {result.best_score:,.2f}")
    print(f"  Hydraulics:   "
          f"{'converged' if result.hydraulics_converged else 'NOT converged'}")
    print(f"  Time:         {result.elapsed_seconds:.1f} s")
    print("=" * 62)

    print("\n  Per-algorithm summary:")
    summary = summarize_experiment(result.experiment)
    for algo, s in summary.items():
        n_ok = max(1, s["n_success"])
        feas_pct = 100 * s["n_feasible"] / n_ok
        best = f"{s['best']:,.0f}" if np.isfinite(s["best"]) else "—"
        mean = f"{s['mean']:,.0f}" if np.isfinite(s["mean"]) else "—"
        print(f"    {algo:6s}  feasible={feas_pct:3.0f}%  "
              f"best={best:>14s}  mean={mean:>14s}")
    print()


def _try_version() -> str:
    try:
        from wds_optimizer import __version__
        return __version__
    except (ImportError, AttributeError):
        return "0.1.0"


# ---------------------------------------------------------------------------
#  Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    try:
        request = _build_request_from_args(args)
        request.validate()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    cancel_flag = _CancelFlag()
    _install_sigint_handler(cancel_flag)

    progress = _Progress(quiet=args.quiet)
    progress.start(args.algo or ["GA"], args.n_runs, args.max_gen)

    result: OptimizationResult | None = None
    try:
        result = run_optimization(
            request,
            progress_callback=progress.update,
            cancel_callback=cancel_flag,
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return EXIT_CANCELLED
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_HYDRAULICS
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_UNEXPECTED
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)

    progress.finish()

    if cancel_flag.cancelled:
        print("\n[wds-optimize] stopped by user (Ctrl+C).",
              file=sys.stderr)

    _print_summary(result)

    if args.output:
        Path(args.output).write_bytes(result.to_json())
        print(f"Wrote {args.output}")
    if args.csv:
        Path(args.csv).write_bytes(result.to_csv())
        print(f"Wrote {args.csv}")
    if args.excel:
        Path(args.excel).write_bytes(result.to_excel())
        print(f"Wrote {args.excel}")
    if args.report:
        Path(args.report).write_bytes(result.to_report_zip(
            reference_cost=args.reference_cost,
            success_tol_pct=args.success_tol,
        ))
        print(f"Wrote {args.report}")

    if cancel_flag.cancelled:
        return EXIT_CANCELLED
    if not result.best_feasible:
        return EXIT_INFEASIBLE
    return EXIT_OK


# ---------------------------------------------------------------------------
#  Subcommand: validate
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    from wds_optimizer.api import _parse_network
    try:
        wn = _parse_network(args.inp)
    except Exception as e:
        print(f"Error reading .inp: {type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_INVALID_INPUT

    units = detect_network_units(wn)

    print(f"Network: {args.inp}")
    print(f"  Pipes:     {len(wn.pipe_name_list)}")
    print(f"  Junctions: {len(wn.junction_name_list)}")
    print(f"  Units:     {units['System']}  "
          f"(pressure {units['PressureUnit']}, "
          f"velocity {units['VelocityUnit']})")
    print(f"  Diameter file: {args.d}  (unit: {args.diam_unit})")
    print(f"  Cost file:     {args.cost}")

    try:
        d_lines = [l.strip()
                   for l in Path(args.d).read_text().splitlines()
                   if l.strip()]
        print(f"  Diameter options: {len(d_lines)}")
    except Exception as e:
        print(f"  Warning: cannot read D.txt: {e}", file=sys.stderr)

    try:
        c_lines = [l.strip()
                   for l in Path(args.cost).read_text().splitlines()
                   if l.strip()]
        print(f"  Cost entries:     {len(c_lines)}")
    except Exception as e:
        print(f"  Warning: cannot read Cost.txt: {e}", file=sys.stderr)

    print("\nAll checks passed.")
    return EXIT_OK


# ---------------------------------------------------------------------------
#  Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args) -> int:
    import platform
    import wntr
    print(f"wds-optimizer:  {_try_version()}")
    print(f"Python:         {platform.python_version()}")
    print(f"wntr:           {getattr(wntr, '__version__', 'unknown')}")
    print(f"Platform:       {platform.platform()}")
    try:
        import streamlit
        print(f"streamlit:      {streamlit.__version__} "
              f"(optional, GUI)")
    except ImportError:
        print("streamlit:      not installed (GUI not available)")
    try:
        import pandas
        print(f"pandas:         {pandas.__version__}")
    except ImportError:
        print("pandas:         not installed")
    return EXIT_OK


# ---------------------------------------------------------------------------
#  Subcommand: benchmark
# ---------------------------------------------------------------------------

_DEMO_NETWORK = """\
[TITLE]
demo 3-pipe line

[JUNCTIONS]
J1  0  5
J2  0  5
J3  0  5

[RESERVOIRS]
R1  50

[PIPES]
P1  R1  J1  1000  200  100  0  Open
P2  J1  J2   800  200  100  0  Open
P3  J2  J3  1200  200  100  0  Open

[OPTIONS]
 Units      LPS
 Headloss   H-W

[END]
"""

_DEMO_D    = "150\n200\n250\n"
_DEMO_COST = "100\n150\n200\n"

_BENCHMARKS = {
    "demo": {
        "inp": _DEMO_NETWORK,
        "d": _DEMO_D,
        "cost": _DEMO_COST,
        "diam_unit": "mm",
    },
}


def cmd_benchmark(args) -> int:
    if args.list:
        print("Available benchmarks:")
        for name in _BENCHMARKS:
            print(f"  {name}")
        print("\nReal-world benchmarks (Hanoi, NYTP, Anytown) "
              "will be added in a follow-up release.")
        return EXIT_OK

    if not args.network:
        print("Error: --network is required (or use --list).",
              file=sys.stderr)
        return EXIT_INVALID_INPUT

    if args.network not in _BENCHMARKS:
        print(
            f"Error: unknown benchmark {args.network!r}. "
            f"Available: {sorted(_BENCHMARKS)}",
            file=sys.stderr,
        )
        return EXIT_INVALID_INPUT

    data = _BENCHMARKS[args.network]
    tmpdir = Path(tempfile.mkdtemp(prefix="wds_benchmark_"))
    inp_p = tmpdir / "net.inp";   inp_p.write_text(data["inp"])
    d_p = tmpdir / "D.txt";       d_p.write_text(data["d"])
    c_p = tmpdir / "Cost.txt";    c_p.write_text(data["cost"])

    # Materialise into args and delegate to cmd_run
    args.inp = str(inp_p)
    args.d = str(d_p)
    args.cost = str(c_p)
    args.diam_unit = data["diam_unit"]
    if args.pmin is None and args.pmax is None \
            and args.vmin is None and args.vmax is None:
        # Sensible defaults so --network demo works without flags
        args.pmin = 30.0
        args.vmax = 2.5
    return cmd_run(args)


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------

def _add_common_algo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--algo", action="append",
        choices=["GA", "PSO", "DE", "SADE"],
        help="Algorithm(s) to run; repeat for multiple (default GA).",
    )
    p.add_argument("--n-runs", type=int, default=1,
                   help="Runs per algorithm (default 1).")
    p.add_argument("--seed", type=int, default=1,
                   help="Experiment seed (default 1).")
    p.add_argument("--ns", type=int, default=100,
                   help="Population / swarm size (default 100).")
    p.add_argument("--max-gen", type=int, default=500,
                   help="Max generations (default 500).")


def _add_constraint_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("constraints (at least one required)")
    g.add_argument("--pmin", type=float,
                   help="Enable Pmin with this value.")
    g.add_argument("--pmax", type=float,
                   help="Enable Pmax with this value.")
    g.add_argument("--vmin", type=float,
                   help="Enable Vmin with this value.")
    g.add_argument("--vmax", type=float,
                   help="Enable Vmax with this value.")
    p.add_argument(
        "--units", choices=["auto", "si"], default="auto",
        help="Units for constraint values: 'auto' (default) = the "
             "network's own units, auto-detected from the .inp; "
             "'si' = metres and m/s.",
    )
    p.add_argument("--fixed", default="",
                   help="Fixed pipe IDs, e.g. '1, 3'.")


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output",
                   help="Write snapshot JSON to this path.")
    p.add_argument("--csv",
                   help="Write per-run CSV to this path.")
    p.add_argument("--excel",
                   help="Write Excel workbook to this path.")
    p.add_argument("--report",
                   help="Write full report ZIP to this path.")
    p.add_argument("--reference-cost", type=float,
                   help="Reference cost for success statistics (optional).")
    p.add_argument("--success-tol", type=float, default=1.0,
                   help="Success tolerance %% above reference (default 1.0).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wds-optimize",
        description="Optimise water distribution system designs.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_try_version()}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="Run an optimisation on user-supplied files.",
    )
    p_run.add_argument("--inp", required=True, help="EPANET .inp file.")
    p_run.add_argument("--d", required=True,
                       help="Diameter-options file.")
    p_run.add_argument("--cost", required=True, help="Unit-cost file.")
    p_run.add_argument("--diam-unit", choices=["mm", "in"], default="mm",
                       help="Unit of values in the diameter file "
                            "(default mm).")
    _add_common_algo_args(p_run)
    _add_constraint_args(p_run)
    _add_output_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser(
        "validate", help="Check input files without running.",
    )
    p_val.add_argument("--inp", required=True)
    p_val.add_argument("--d", required=True)
    p_val.add_argument("--cost", required=True)
    p_val.add_argument("--diam-unit", choices=["mm", "in"], default="mm")
    p_val.set_defaults(func=cmd_validate)

    p_info = sub.add_parser("info", help="Print environment metadata.")
    p_info.set_defaults(func=cmd_info)

    p_bench = sub.add_parser(
        "benchmark", help="Run on a built-in benchmark network.",
    )
    p_bench.add_argument("--list", action="store_true",
                         help="List available benchmarks.")
    p_bench.add_argument("--network",
                         help="Benchmark name (see --list).")
    _add_common_algo_args(p_bench)
    _add_constraint_args(p_bench)
    _add_output_args(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as e:
        return int(e.code) if e.code is not None else EXIT_OK
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())