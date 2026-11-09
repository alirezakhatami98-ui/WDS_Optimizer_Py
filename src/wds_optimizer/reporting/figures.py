"""Static figure generation (matplotlib) for reproducibility reports."""

from __future__ import annotations

import io

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ..optimization.severity import two_sided_severity


# =====================================================================
#  Utilities
# =====================================================================

def _to_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def _sanitize_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")


# =====================================================================
#  Convergence
# =====================================================================

def make_convergence_png(result, colors=None) -> bytes:
    """Mean convergence + min-max band per algorithm."""
    fig = Figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    colors = colors or {"GA": "#d9480f", "PSO": "#1971c2", "DE": "#7048e8", "SADE": "#0b7285"}

    for algo, runs in result.runs.items():
        convs = [
            r.convergence for r in runs
            if r.status == "Success" and r.convergence is not None
        ]
        if not convs:
            continue
        maxlen = max(len(c) for c in convs)
        padded = np.full((len(convs), maxlen), np.nan)
        for i, c in enumerate(convs):
            padded[i, : len(c)] = c
        mean = np.nanmean(padded, axis=0)
        lo = np.nanmin(padded, axis=0)
        hi = np.nanmax(padded, axis=0)
        x = np.arange(1, maxlen + 1)
        c = colors.get(algo, "gray")
        ax.fill_between(x, lo, hi, color=c, alpha=0.15,
                        label=f"{algo} (min-max)")
        ax.plot(x, mean, color=c, linewidth=2,
                label=f"{algo} (mean)")

    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("Convergence: mean and min-max band")
    ax.legend(loc="upper right", fontsize=9)
    _sanitize_ax(ax)
    return _to_png(fig)


# =====================================================================
#  Cost distribution (boxplot)
# =====================================================================

def make_cost_distribution_png(result, colors=None) -> bytes:
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    colors = colors or {"GA": "#d9480f", "PSO": "#1971c2", "DE": "#7048e8", "SADE": "#0b7285"}

    data, labels = [], []
    for algo, runs in result.runs.items():
        cs = [r.score for r in runs if r.status == "Success" and r.feasible]
        if cs:
            data.append(cs)
            labels.append(algo)

    if not data:
        ax.text(0.5, 0.5, "No feasible runs", ha="center", va="center",
                transform=ax.transAxes)
        return _to_png(fig)

    # `labels` was removed from Axes.boxplot() in matplotlib 3.10.
    # Use set_xticks/set_xticklabels — portable across all versions.
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(label, "gray"))
        patch.set_alpha(0.55)
    ax.set_ylabel("Cost")
    ax.set_title("Final cost distribution (feasible runs)")
    _sanitize_ax(ax)
    return _to_png(fig)


# =====================================================================
#  Pressures / velocities bar charts
# =====================================================================

def make_pressure_png(junction_names, pressures_si, pmin_si=None,
                      units=None, pmax_si=None) -> bytes:
    """Pressure bar chart with optional Pmin / Pmax reference lines.

    ``pmin_si`` / ``pmax_si`` are in metres; pass ``None`` to omit that
    constraint line and exclude it from violation colouring.
    """
    from ..hydraulics.units import pressure_from_si

    units = units or {"PressureUnit": "m"}
    p_unit = units.get("PressureUnit", "m")
    pressures = np.array([pressure_from_si(p, units) for p in pressures_si])
    pmin = pressure_from_si(pmin_si, units) if pmin_si is not None else None
    pmax = pressure_from_si(pmax_si, units) if pmax_si is not None else None

    def _viol(p):
        return ((pmin is not None and p < pmin)
                or (pmax is not None and p > pmax))

    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    colors = ["#e74c3c" if _viol(p) else "#2ecc71" for p in pressures]
    x = np.arange(len(pressures))
    ax.bar(x, pressures, color=colors)
    if pmin is not None:
        ax.axhline(pmin, color="red", linestyle="--",
                   label=f"Pmin = {pmin:.2f} {p_unit}")
    if pmax is not None:
        ax.axhline(pmax, color="red", linestyle="--",
                   label=f"Pmax = {pmax:.2f} {p_unit}")
    ax.set_xticks(x)
    ax.set_xticklabels(junction_names, rotation=45)
    ax.set_ylabel(f"Pressure ({p_unit})")
    ax.set_title("Node pressures (best design)")
    if pmin is not None or pmax is not None:
        ax.legend()
    _sanitize_ax(ax)
    return _to_png(fig)


def make_velocity_png(pipe_names, velocities_si, vmax_si=None,
                      units=None, vmin_si=None) -> bytes:
    """Velocity bar chart with optional Vmin / Vmax reference lines.

    ``vmin_si`` / ``vmax_si`` are in m/s; pass ``None`` to omit.
    """
    from ..hydraulics.units import velocity_from_si

    units = units or {"VelocityUnit": "m/s"}
    v_unit = units.get("VelocityUnit", "m/s")
    velocities = np.array([velocity_from_si(v, units) for v in velocities_si])
    vmin = velocity_from_si(vmin_si, units) if vmin_si is not None else None
    vmax = velocity_from_si(vmax_si, units) if vmax_si is not None else None

    def _viol(v):
        return ((vmin is not None and v < vmin)
                or (vmax is not None and v > vmax))

    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    colors = ["#e74c3c" if _viol(v) else "#3498db" for v in velocities]
    x = np.arange(len(velocities))
    ax.bar(x, velocities, color=colors)
    if vmin is not None:
        ax.axhline(vmin, color="red", linestyle="--",
                   label=f"Vmin = {vmin:.2f} {v_unit}")
    if vmax is not None:
        ax.axhline(vmax, color="red", linestyle="--",
                   label=f"Vmax = {vmax:.2f} {v_unit}")
    ax.set_xticks(x)
    ax.set_xticklabels(pipe_names, rotation=45)
    ax.set_ylabel(f"Velocity ({v_unit})")
    ax.set_title("Pipe velocities (best design)")
    if vmin is not None or vmax is not None:
        ax.legend()
    _sanitize_ax(ax)
    return _to_png(fig)


# =====================================================================
#  Network map (static)
# =====================================================================

def make_network_map_png(
    wn, pipe_names, junction_names, position_m,
    pipe_velocities, node_pressures,
    mode="velocity", fixed_pipe_indices=None,
    pmin=None, pmax=None, vmin=None, vmax=None, units=None,
) -> bytes:
    """Static network map for the reproducibility report.

    Mirrors the interactive map (``wds_optimizer.app.network_map``) so
    the PNG embedded in the ZIP shows the same information as the app.
    All four constraints are optional; pass ``None`` for any that are
    disabled, and the figure omits its colouring influence.

    Colour semantics (matching build_network_figure):

    Pressure (nodes)
        Pmin only  → RdYlGn   (red = low pressure = violation)
        Pmax only  → RdYlGn_r (red = high pressure = violation)
        both       → RdYlGn_r on a severity scale (0 = OK, 1 = severe)
        neither    → gray

    Velocity (pipes)
        Vmax only  → RdYlGn_r (red = high velocity = violation)
        Vmin only  → RdYlGn   (red = low velocity = violation)
        both       → RdYlGn_r on a severity scale
        neither    → gray

    Backward compatibility: with only Pmin and Vmax supplied (the
    pre-Pmax/Vmin defaults), the colour range and colormaps reproduce
    the old implementation exactly, so archived reports remain
    visually comparable.
    """
    from ..hydraulics.units import (
        diameter_to_display, pressure_from_si, velocity_from_si,
    )

    units = units or {}
    p_unit = units.get("PressureUnit", "m")
    v_unit = units.get("VelocityUnit", "m/s")
    d_unit = units.get("DiameterUnit", "m")

    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    fixed_set = set(fixed_pipe_indices or [])
    D_si = np.asarray(position_m, dtype=float)
    D_disp = np.array([diameter_to_display(d, units) for d in D_si])
    dmin, dmax = D_disp.min(), D_disp.max()

    # --------------------------------------------------------------
    # Decide pipe colormap/norm
    # --------------------------------------------------------------
    pipe_cmap = None
    pipe_norm = None
    pipe_color_vals = None
    pipe_cbar_label = ""

    if mode == "velocity":
        vals_si = np.asarray(pipe_velocities, dtype=float)
        vals = np.array([velocity_from_si(v, units) for v in vals_si])
        vmin_disp = velocity_from_si(vmin, units) if vmin is not None else None
        vmax_disp = velocity_from_si(vmax, units) if vmax is not None else None

        if vmin_disp is not None and vmax_disp is not None:
            pipe_cmap = colormaps["RdYlGn_r"]
            pipe_color_vals = two_sided_severity(vals, vmin_disp, vmax_disp)
            pipe_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            pipe_cbar_label = (f"Velocity severity ({v_unit}) — "
                               f"green=OK, red=violation")
        elif vmin_disp is not None:
            pipe_cmap = colormaps["RdYlGn"]
            pipe_color_vals = vals
            lo = min(float(vals.min()) if len(vals) else vmin_disp,
                     vmin_disp)
            hi = float(vals.max()) if len(vals) else vmin_disp * 2
            if hi <= lo:
                hi = lo + 1.0
            pipe_norm = mcolors.Normalize(vmin=lo, vmax=hi)
            pipe_cbar_label = f"Velocity ({v_unit})"
        elif vmax_disp is not None:
            pipe_cmap = colormaps["RdYlGn_r"]
            pipe_color_vals = vals
            # Reproduces the previous behaviour exactly:
            #   vmax_val = max(vmax_disp, vals.max() * 1.05)
            vmax_val = max(vmax_disp,
                           float(vals.max()) * 1.05 if len(vals) else 1.0)
            pipe_norm = mcolors.Normalize(vmin=0.0, vmax=vmax_val)
            pipe_cbar_label = f"Velocity ({v_unit})"
        # else: leave all None → gray preview

    elif mode == "diameter":
        pipe_cmap = colormaps["viridis"]
        pipe_color_vals = D_disp
        if len(D_disp):
            dmin_v = float(D_disp.min())
            dmax_v = float(D_disp.max())
            if dmax_v <= dmin_v:
                dmax_v = dmin_v + 1e-9
        else:
            dmin_v, dmax_v = 0.0, 1.0
        pipe_norm = mcolors.Normalize(vmin=dmin_v, vmax=dmax_v)
        pipe_cbar_label = f"Diameter ({d_unit})"

    # --------------------------------------------------------------
    # Draw pipes
    # --------------------------------------------------------------
    for i, pname in enumerate(pipe_names):
        link = wn.get_link(pname)
        n0, n1 = link.start_node_name, link.end_node_name
        x = [wn.get_node(n0).coordinates[0],
             wn.get_node(n1).coordinates[0]]
        y = [wn.get_node(n0).coordinates[1],
             wn.get_node(n1).coordinates[1]]

        if i in fixed_set:
            ax.plot(x, y, color="orange", linewidth=3,
                    linestyle="--", zorder=2)
            continue

        if pipe_cmap is not None and pipe_color_vals is not None:
            # NaN → neutral gray (matplotlib's default "bad" colour
            # is transparent, which hides the pipe entirely — fixing
            # the pre-refactor bug where NaN pipes were invisible).
            val = pipe_color_vals[i]
            color = (pipe_cmap(pipe_norm(val))
                     if np.isfinite(val) else "#b4b4b4")
        else:
            color = "#999999"
        lw = 1 + 4 * (D_disp[i] - dmin) / max(dmax - dmin, 1e-9)
        ax.plot(x, y, color=color, linewidth=lw, zorder=2)

    # --------------------------------------------------------------
    # Decide junction colormap/norm
    # --------------------------------------------------------------
    jx, jy, jp_si, jnames = [], [], [], []
    for i, n in enumerate(junction_names):
        x, y = wn.get_node(n).coordinates
        jx.append(x)
        jy.append(y)
        jnames.append(n)
        jp_si.append(node_pressures[i])

    junc_cmap = None
    junc_norm = None
    junc_color_vals = None
    junc_cbar_label = ""
    pmin_disp = pmax_disp = None       # for the title

    if mode == "pressure":
        jp_disp = np.array([pressure_from_si(p, units) for p in jp_si])
        pmin_disp = pressure_from_si(pmin, units) if pmin is not None else None
        pmax_disp = pressure_from_si(pmax, units) if pmax is not None else None

        if pmin_disp is not None and pmax_disp is not None:
            junc_cmap = colormaps["RdYlGn_r"]
            junc_color_vals = two_sided_severity(
                jp_disp, pmin_disp, pmax_disp,
            )
            junc_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            junc_cbar_label = (f"Pressure severity ({p_unit}) — "
                               f"green=OK, red=violation")
        elif pmin_disp is not None:
            junc_cmap = colormaps["RdYlGn"]
            junc_color_vals = jp_disp
            # Reproduces the previous behaviour exactly:
            #   lo = min(jp_disp.min(), pmin_disp)
            #   hi = max(jp_disp.max(), pmin_disp)
            lo = min(float(jp_disp.min()) if len(jp_disp) else pmin_disp,
                     pmin_disp)
            hi = max(float(jp_disp.max()) if len(jp_disp) else pmin_disp,
                     pmin_disp)
            if hi <= lo:
                hi = lo + 1.0
            junc_norm = mcolors.Normalize(vmin=lo, vmax=hi)
            junc_cbar_label = f"Pressure ({p_unit})"
        elif pmax_disp is not None:
            junc_cmap = colormaps["RdYlGn_r"]
            junc_color_vals = jp_disp
            lo = float(jp_disp.min()) if len(jp_disp) else 0.0
            hi = max(float(jp_disp.max()) if len(jp_disp) else pmax_disp,
                     pmax_disp)
            if hi <= lo:
                hi = lo + 1.0
            junc_norm = mcolors.Normalize(vmin=lo, vmax=hi)
            junc_cbar_label = f"Pressure ({p_unit})"

    if junc_cmap is not None and junc_color_vals is not None:
        node_colors = [
            (junc_cmap(junc_norm(v)) if np.isfinite(v) else "#b4b4b4")
            for v in junc_color_vals
        ]
    else:
        node_colors = ["#dddddd"] * len(jx)

    ax.scatter(jx, jy, c=node_colors, s=180, zorder=3,
               edgecolors="black", linewidths=1)
    for x, y, name in zip(jx, jy, jnames):
        ax.annotate(name, (x, y), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=8)

    # Reservoirs / tanks
    for n in wn.reservoir_name_list + wn.tank_name_list:
        x, y = wn.get_node(n).coordinates
        ax.scatter([x], [y], c="green", marker="s", s=220, zorder=3,
                   edgecolors="black", linewidths=1)
        ax.annotate(n, (x, y), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title: append the active constraint values (the PNG has no
    # hover, so the values must live in the title). ----
    title_parts = [f"Network Map ({mode})"]
    if mode == "pressure":
        if pmin_disp is not None:
            title_parts.append(f"Pmin={pmin_disp:.2f} {p_unit}")
        if pmax_disp is not None:
            title_parts.append(f"Pmax={pmax_disp:.2f} {p_unit}")
    elif mode == "velocity":
        if vmin is not None:
            title_parts.append(
                f"Vmin={velocity_from_si(vmin, units):.2f} {v_unit}")
        if vmax is not None:
            title_parts.append(
                f"Vmax={velocity_from_si(vmax, units):.2f} {v_unit}")
    ax.set_title(" — ".join(title_parts))

    # ---- Colorbar ----
    if pipe_cmap is not None and mode in ("velocity", "diameter"):
        sm = ScalarMappable(cmap=pipe_cmap, norm=pipe_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=pipe_cbar_label, shrink=0.7)
    elif mode == "pressure" and junc_cmap is not None:
        sm = ScalarMappable(cmap=junc_cmap, norm=junc_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=junc_cbar_label, shrink=0.7)

    return _to_png(fig)