"""Static figure generation (matplotlib) for reproducibility reports."""

from __future__ import annotations

import io

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable          # ← این خط
from matplotlib.figure import Figure


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
    colors = colors or {"GA": "#d9480f", "PSO": "#1971c2"}

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
    colors = colors or {"GA": "#d9480f", "PSO": "#1971c2"}

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

def make_pressure_png(junction_names, pressures_si, pmin_si, units=None) -> bytes:
    """Pressure bar chart. ``pressures_si``/``pmin_si`` are in meters."""
    from ..hydraulics.units import pressure_from_si

    units = units or {"PressureUnit": "m"}
    p_unit = units.get("PressureUnit", "m")
    pressures = np.array([pressure_from_si(p, units) for p in pressures_si])
    pmin = pressure_from_si(pmin_si, units)

    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    colors = ["#e74c3c" if p < pmin else "#2ecc71" for p in pressures]
    x = np.arange(len(pressures))
    ax.bar(x, pressures, color=colors)
    ax.axhline(pmin, color="red", linestyle="--",
               label=f"Pmin = {pmin:.2f} {p_unit}")
    ax.set_xticks(x)
    ax.set_xticklabels(junction_names, rotation=45)
    ax.set_ylabel(f"Pressure ({p_unit})")
    ax.set_title("Node pressures (best design)")
    ax.legend()
    _sanitize_ax(ax)
    return _to_png(fig)


def make_velocity_png(pipe_names, velocities_si, vmax_si, units=None) -> bytes:
    """Velocity bar chart. ``velocities_si``/``vmax_si`` are in m/s."""
    from ..hydraulics.units import velocity_from_si

    units = units or {"VelocityUnit": "m/s"}
    v_unit = units.get("VelocityUnit", "m/s")
    velocities = np.array([velocity_from_si(v, units) for v in velocities_si])
    vmax = velocity_from_si(vmax_si, units)

    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    colors = ["#e74c3c" if v > vmax else "#3498db" for v in velocities]
    x = np.arange(len(velocities))
    ax.bar(x, velocities, color=colors)
    ax.axhline(vmax, color="red", linestyle="--",
               label=f"Vmax = {vmax:.2f} {v_unit}")
    ax.set_xticks(x)
    ax.set_xticklabels(pipe_names, rotation=45)
    ax.set_ylabel(f"Velocity ({v_unit})")
    ax.set_title("Pipe velocities (best design)")
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
    pmin=None, vmax=None, units=None,
) -> bytes:
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

    # Choose coloring
    if mode == "velocity":
        cmap = colormaps["RdYlGn_r"]
        vals_si = np.asarray(pipe_velocities, dtype=float)
        vals = np.array([velocity_from_si(v, units) for v in vals_si])
        vmax_disp = velocity_from_si(vmax or 0.0, units) if vmax else None
        vmax_val = max(vmax_disp or 0.0,
                       vals.max() * 1.05 if len(vals) else 1.0)
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax_val)
        cbar_label = f"Velocity ({v_unit})"
    elif mode == "diameter":
        cmap = colormaps["viridis"]
        vals = D_disp
        norm = mcolors.Normalize(
            vmin=vals.min(),
            vmax=vals.max() if vals.max() > vals.min() else vals.min() + 1e-9,
        )
        cbar_label = f"Diameter ({d_unit})"
    else:
        cmap = None
        norm = None
        vals = None
        cbar_label = ""

    # Pipes
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

        if mode in ("velocity", "diameter"):
            color = cmap(norm(vals[i]))
        else:
            color = "#999999"
        lw = 1 + 4 * (D_disp[i] - dmin) / max(dmax - dmin, 1e-9)
        ax.plot(x, y, color=color, linewidth=lw, zorder=2)

    # Junctions
    jx, jy, jp_si, jnames = [], [], [], []
    for i, n in enumerate(junction_names):
        x, y = wn.get_node(n).coordinates
        jx.append(x)
        jy.append(y)
        jnames.append(n)
        jp_si.append(node_pressures[i])

    if mode == "pressure":
        jp_disp = [pressure_from_si(p, units) for p in jp_si]
        pmin_disp = pressure_from_si(pmin or 0.0, units) if pmin else 0.0
        cmap_p = colormaps["RdYlGn"]
        lo = min(min(jp_disp), pmin_disp)
        hi = max(max(jp_disp), pmin_disp)
        if hi <= lo:
            hi = lo + 1.0
        norm_p = mcolors.Normalize(vmin=lo, vmax=hi)
        node_colors = [cmap_p(norm_p(v)) for v in jp_disp]
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
    ax.set_title(f"Network Map ({mode})")

    # Colorbar
    if mode in ("velocity", "diameter"):
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=cbar_label, shrink=0.7)
    elif mode == "pressure":
        sm = ScalarMappable(cmap=cmap_p, norm=norm_p)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"Pressure ({p_unit})", shrink=0.7)

    return _to_png(fig)