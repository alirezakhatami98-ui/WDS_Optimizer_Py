"""Interactive network map using Plotly."""

import numpy as np
import plotly.graph_objects as go
import wntr


def build_network_figure(wn, mode="preview",
                         pressures=None, velocities=None,
                         diameters=None,
                         pmin=None, pmax=None, vmin=None, vmax=None,
                         fixed_pipe_indices=None, show_labels=False,
                         units=None):
    """Return a Plotly Figure.

    Parameters
    ----------
    wn : WaterNetworkModel
    mode : {'preview', 'velocity', 'pressure', 'diameter'}
    pressures : dict[node_name -> value], optional
    velocities : dict[link_name -> value], optional
    diameters : dict[link_name -> value_meters], optional
    pmin, pmax, vmin, vmax : float, optional
        Constraint values (SI, pressure in m, velocity in m/s).
        Pass ``None`` for a constraint that is disabled — it will not
        influence colouring, hover or the colour bar.  Pass a value
        to enable it.
    fixed_pipe_indices : list[int], optional
        1-based indices into ``wn.pipe_name_list``.

    Colouring semantics
    -------------------
    * No relevant constraint enabled  → gray (preview style).
    * Exactly one pressure/velocity constraint enabled → value-based
      colouring with the colormap chosen so red = the "bad" side:
        - Pmin only: RdYlGn   (red at low pressure)
        - Pmax only: RdYlGn_r (red at high pressure)
        - Vmin only: RdYlGn   (red at low velocity)
        - Vmax only: RdYlGn_r (red at high velocity)
    * Both constraints of one family enabled → severity colouring
      (green = OK, red = violation), scaled by how far the value is
      from the boundary.

    Hover shows every supplied constraint (including disabled ones with
    an ``off`` marker) so the user can tell what is being enforced.
    """
    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    pipe_names = list(wn.pipe_name_list)
    fixed_set = set(fixed_pipe_indices or [])
    u = units or {}
    p_unit = u.get("PressureUnit", "m")
    v_unit = u.get("VelocityUnit", "m/s")
    d_unit = u.get("DiameterUnit", "mm")

    # ---- Helpers ----
    def _two_sided_severity(value, low, high):
        """0 = inside [low, high], up to 1 = severe violation."""
        if not np.isfinite(value):
            return 1.0
        if low is not None and value < low:
            span = max(abs(low), 1e-9)
            return min(1.0, (low - value) / span)
        if high is not None and value > high:
            span = max(abs(high), 1e-9)
            return min(1.0, (value - high) / span)
        return 0.0

    def _pressure_hint():
        parts = []
        if pmin is not None:
            parts.append(f"Pmin={pmin:.2f}")
        if pmax is not None:
            parts.append(f"Pmax={pmax:.2f}")
        return f" ({', '.join(parts)} {p_unit})" if parts else ""

    def _velocity_hint():
        parts = []
        if vmin is not None:
            parts.append(f"Vmin={vmin:.2f}")
        if vmax is not None:
            parts.append(f"Vmax={vmax:.2f}")
        return f" ({', '.join(parts)} {v_unit})" if parts else ""

    # ---- Decide pipe coloring (velocity / diameter / preview) ----
    # color_vals[i]      -- value fed to the colormap for pipe i
    # hover_val[i]       -- actual physical value shown in hover
    # colorscale / cmin / cmax / cbar_title describe the coloring
    if mode == "velocity" and velocities is not None:
        hover_val = [velocities.get(p, np.nan) for p in pipe_names]
        vals_finite = [v for v in hover_val if np.isfinite(v)]

        if vmin is not None and vmax is not None:
            # Two-sided: severity colouring (green=OK, red=violation)
            color_vals = [_two_sided_severity(v, vmin, vmax)
                          for v in hover_val]
            colorscale = "RdYlGn_r"
            cmin, cmax = 0.0, 1.0
            cbar_title = (f"Velocity severity ({v_unit}) — "
                          f"green=OK, red=violation")
        elif vmin is not None:
            # Vmin only: red at low velocity
            color_vals = hover_val
            colorscale = "RdYlGn"
            cmin = min([vmin] + vals_finite) if vals_finite else vmin
            cmax = max(vals_finite) if vals_finite else vmin * 2
            if cmax <= cmin:
                cmax = cmin + 1.0
            cbar_title = f"Velocity ({v_unit})"
        elif vmax is not None:
            # Vmax only: red at high velocity (current behaviour)
            color_vals = hover_val
            colorscale = "RdYlGn_r"
            cmin = 0.0
            cmax = max(vmax, (max(vals_finite) * 1.05)
                       if vals_finite else vmax)
            cbar_title = f"Velocity ({v_unit})"
        else:
            color_vals = [1.0] * len(pipe_names)
            colorscale = None
            cmin, cmax = 0, 1
            cbar_title = ""

    elif mode == "diameter" and diameters is not None:
        hover_val = [diameters.get(p, np.nan) for p in pipe_names]
        vals_finite = [v for v in hover_val if np.isfinite(v)]
        color_vals = hover_val
        colorscale = "Viridis"
        cmin = min(vals_finite) if vals_finite else 0
        cmax = max(vals_finite) if vals_finite else 1
        cbar_title = f"Diameter ({d_unit})"

    else:
        hover_val = [np.nan] * len(pipe_names)
        color_vals = [1.0] * len(pipe_names)
        colorscale = None
        cmin, cmax = 0, 1
        cbar_title = ""

    # ---- Pipe traces ----
    fig = go.Figure()
    for i, pname in enumerate(pipe_names):
        link = wn.get_link(pname)
        n0, n1 = link.start_node_name, link.end_node_name
        x = [coords[n0][0], coords[n1][0]]
        y = [coords[n0][1], coords[n1][1]]

        if (i + 1) in fixed_set:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="rgba(230, 126, 34, 1.0)",
                          width=4, dash="dash"),
                hovertemplate=f"<b>Pipe {pname}</b><br>FIXED"
                              "<extra></extra>",
                showlegend=False,
            ))
        else:
            line_color = (_val_to_color(color_vals[i], cmin, cmax,
                                        colorscale)
                          if colorscale else "rgba(120,120,120,0.85)")
            if mode == "velocity":
                hv = (f"{hover_val[i]:.3f} {v_unit}{_velocity_hint()}"
                      if np.isfinite(hover_val[i]) else "n/a")
                hovertmpl = (f"<b>Pipe {pname}</b><br>"
                             f"Velocity: {hv}<extra></extra>")
            elif mode == "diameter":
                hv = (f"{hover_val[i]:.3f} {d_unit}"
                      if np.isfinite(hover_val[i]) else "n/a")
                hovertmpl = (f"<b>Pipe {pname}</b><br>"
                             f"Diameter: {hv}<extra></extra>")
            else:
                hovertmpl = f"<b>Pipe {pname}</b><extra></extra>"

            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=line_color, width=3),
                hovertemplate=hovertmpl,
                showlegend=False,
            ))

    # ---- Decide node coloring (pressure) ----
    color_by_pressure = (mode == "pressure" and pressures is not None)
    junc_x, junc_y, junc_names, junc_vals = [], [], [], []
    res_x, res_y, res_names = [], [], []

    for n in wn.junction_name_list:
        x, y = coords[n]
        junc_x.append(x); junc_y.append(y); junc_names.append(n)
        junc_vals.append(pressures.get(n, np.nan)
                         if color_by_pressure else np.nan)
    for n in wn.reservoir_name_list + wn.tank_name_list:
        x, y = coords[n]
        res_x.append(x); res_y.append(y); res_names.append(n)

    if color_by_pressure:
        vals_finite = [v for v in junc_vals if np.isfinite(v)]

        if pmin is not None and pmax is not None:
            # Two-sided: severity colouring
            node_color_vals = [_two_sided_severity(v, pmin, pmax)
                               for v in junc_vals]
            node_colorscale = "RdYlGn_r"
            cmin_p, cmax_p = 0.0, 1.0
            cbar_title_p = (f"Pressure severity ({p_unit}) — "
                            f"green=OK, red=violation")
        elif pmin is not None:
            # Pmin only (current behaviour): red at low pressure
            node_color_vals = junc_vals
            node_colorscale = "RdYlGn"
            cmin_p = min([pmin] + vals_finite) if vals_finite else pmin
            cmax_p = max(vals_finite) if vals_finite else pmin * 2
            if cmax_p <= cmin_p:
                cmax_p = cmin_p + 1.0
            cbar_title_p = f"Pressure ({p_unit})"
        elif pmax is not None:
            # Pmax only: red at high pressure
            node_color_vals = junc_vals
            node_colorscale = "RdYlGn_r"
            cmin_p = min(vals_finite) if vals_finite else 0.0
            cmax_p = max([pmax] + vals_finite) if vals_finite else pmax
            if cmax_p <= cmin_p:
                cmax_p = cmin_p + 1.0
            cbar_title_p = f"Pressure ({p_unit})"
        else:
            node_colorscale = None
            cmin_p, cmax_p = 0, 1
            cbar_title_p = ""
            node_color_vals = junc_vals

        hint = _pressure_hint()
        customdata = [
            [n, v] if np.isfinite(v) else [n, None]
            for n, v in zip(junc_names, junc_vals)
        ]
        fig.add_trace(go.Scatter(
            x=junc_x, y=junc_y,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=12, color=node_color_vals,
                colorscale=node_colorscale if node_colorscale else "Greys",
                cmin=cmin_p, cmax=cmax_p,
                showscale=bool(node_colorscale),
                colorbar=dict(title=cbar_title_p) if node_colorscale else None,
                line=dict(width=1, color="black"),
            ),
            text=junc_names if show_labels else None,
            textposition="top center",
            textfont=dict(size=8),
            customdata=customdata,
            hovertemplate=(
                f"<b>Junction %{{customdata[0]}}</b><br>"
                f"Pressure: %{{customdata[1]:.2f}} {p_unit}"
                f"{hint}<extra></extra>"
            ),
            name="Junctions",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=junc_x, y=junc_y,
            mode="markers+text" if show_labels else "markers",
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
    if colorscale is not None and mode in ("velocity", "diameter"):
        ref_x = res_x[0] if res_x else (junc_x[0] if junc_x else 0.0)
        ref_y = res_y[0] if res_y else (junc_y[0] if junc_y else 0.0)
        fig.add_trace(go.Scatter(
            x=[ref_x, ref_x], y=[ref_y, ref_y],
            mode="markers",
            marker=dict(
                colorscale=colorscale, cmin=cmin, cmax=cmax,
                color=[cmin, cmax], showscale=True,
                colorbar=dict(title=cbar_title),
                size=1, opacity=0, line=dict(width=0),
            ),
            showlegend=False, hoverinfo="skip",
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