"""Interactive network map using Plotly."""

import numpy as np
import plotly.graph_objects as go
import wntr


def build_network_figure(wn, mode="preview",
                         pressures=None, velocities=None,
                         diameters=None, pmin=None, vmax=None,
                         fixed_pipe_indices=None, show_labels=False):
    """Return a Plotly Figure.

    Parameters
    ----------
    wn : WaterNetworkModel
    mode : {'preview', 'velocity', 'pressure', 'diameter'}
    pressures : dict[node_name -> value], optional
    velocities : dict[link_name -> value], optional
    diameters : dict[link_name -> value_meters], optional
    pmin, vmax : float, optional
    fixed_pipe_indices : list[int], optional
        1-based indices into ``wn.pipe_name_list``.
    """
    # Node coordinates
    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}

    pipe_names = list(wn.pipe_name_list)
    fixed_set = set(fixed_pipe_indices or [])

    # ---- Decide link coloring ----
    if mode == "velocity" and velocities is not None:
        vals = [velocities.get(p, np.nan) for p in pipe_names]
        colorscale = "RdYlGn_r"
        cbar_title = "Velocity (m/s)"
        cmin = 0
        cmax = max(vmax or 2.5, max([v for v in vals if np.isfinite(v)] or [1]))
    elif mode == "diameter" and diameters is not None:
        vals = [diameters.get(p, np.nan) for p in pipe_names]
        colorscale = "Viridis"
        cbar_title = "Diameter (m)"
        cmin = min([v for v in vals if np.isfinite(v)] or [0])
        cmax = max([v for v in vals if np.isfinite(v)] or [1])
    else:
        vals = [1.0] * len(pipe_names)
        colorscale = None
        cbar_title = ""
        cmin, cmax = 0, 1

    # ---- Build pipe traces (one per pipe to allow individual styling) ----
    fig = go.Figure()
    for i, pname in enumerate(pipe_names):
        link = wn.get_link(pname)
        n0, n1 = link.start_node_name, link.end_node_name
        x = [coords[n0][0], coords[n1][0]]
        y = [coords[n0][1], coords[n1][1]]

        if (i + 1) in fixed_set:
            # Fixed pipe: dashed orange overlay
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="rgba(230, 126, 34, 1.0)", width=4, dash="dash"),
                hovertemplate=f"<b>Pipe {pname}</b><br>FIXED<extra></extra>",
                showlegend=False,
            ))
        else:
            color = vals[i]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=_val_to_color(color, cmin, cmax, colorscale)
                          if colorscale else "rgba(120,120,120,0.85)",
                          width=3),
                customdata=[[pname, color]],
                hovertemplate=(
                    f"<b>Pipe {pname}</b><br>"
                    f"{cbar_title}: %{{customdata[1]:.3f}}<extra></extra>"
                ),
                showlegend=False,
            ))

    # ---- Nodes ----
    junc_x, junc_y, junc_names = [], [], []
    res_x, res_y, res_names = [], [], []
    color_by_pressure = (mode == "pressure" and pressures is not None)
    junc_vals = []

    for n in wn.junction_name_list:
        x, y = coords[n]
        junc_x.append(x); junc_y.append(y); junc_names.append(n)
        junc_vals.append(pressures.get(n, np.nan) if color_by_pressure else np.nan)
    for n in wn.reservoir_name_list + wn.tank_name_list:
        x, y = coords[n]
        res_x.append(x); res_y.append(y); res_names.append(n)

    if color_by_pressure:
        lo = min([v for v in junc_vals if np.isfinite(v)] + [pmin or 0])
        hi = max([v for v in junc_vals if np.isfinite(v)] + [pmin or 1])
        fig.add_trace(go.Scatter(
            x=junc_x, y=junc_y, mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=12, color=junc_vals, colorscale="RdYlGn",
                cmin=lo, cmax=hi, showscale=True,
                colorbar=dict(title="Pressure (m)"),
                line=dict(width=1, color="black"),
            ),
            text=junc_names if show_labels else None,
            textposition="top center",
            textfont=dict(size=8),
            customdata=list(zip(junc_names, junc_vals)),
            hovertemplate=(
                "<b>Junction %{customdata[0]}</b><br>"
                "Pressure: %{customdata[1]:.2f} m<extra></extra>"
            ),
            name="Junctions",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=junc_x, y=junc_y, mode="markers+text" if show_labels else "markers",
            marker=dict(size=11, color="lightgray",
                        line=dict(width=1, color="black")),
            text=junc_names if show_labels else None,
            textposition="top center", textfont=dict(size=8),
            customdata=junc_names,
            hovertemplate="<b>Junction %{customdata}</b><extra></extra>",
            name="Junctions",
        ))

    fig.add_trace(go.Scatter(
        x=res_x, y=res_y, mode="markers",
        marker=dict(size=16, symbol="square", color="green",
                    line=dict(width=1, color="black")),
        customdata=res_names,
        hovertemplate="<b>%{customdata}</b><extra></extra>",
        name="Reservoir/Tank",
    ))

    # ---- Colorbar for pipe-based colorings (velocity, diameter) ----
    # Per-pipe traces use discrete line colors, so Plotly does not build
    # a colorbar automatically. This invisible marker trace only exists
    # to trigger the colorbar (size≈0, opacity=0).
    if colorscale is not None and mode in ("velocity", "diameter"):
        ref_x = res_x[0] if res_x else (junc_x[0] if junc_x else 0.0)
        ref_y = res_y[0] if res_y else (junc_y[0] if junc_y else 0.0)
        fig.add_trace(go.Scatter(
            x=[ref_x, ref_x],
            y=[ref_y, ref_y],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                color=[cmin, cmax],
                showscale=True,
                colorbar=dict(title=cbar_title),
                size=1,
                opacity=0,
                line=dict(width=0),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- Layout ----
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        hovermode="closest",
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.7)"),
        title=f"Network ({mode})",
    )
    return fig


def _val_to_color(v, cmin, cmax, colorscale):
    """Map a scalar to an approximate RGBA color from a Plotly colorscale."""
    if not np.isfinite(v):
        return "rgba(180,180,180,0.8)"
    if cmax <= cmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (v - cmin) / (cmax - cmin)))

    # Sample from Plotly colorscale via simple interpolation
    import plotly.colors as pc
    try:
        rgb = pc.sample_colorscale(colorscale, t)[0]
    except Exception:
        rgb = "rgb(120,120,120)"
    # sample_colorscale returns 'rgb(r, g, b)' string
    r, g, b = [int(x) for x in rgb.split("(")[1].rstrip(")").split(",")]
    return f"rgba({r},{g},{b},0.9)"