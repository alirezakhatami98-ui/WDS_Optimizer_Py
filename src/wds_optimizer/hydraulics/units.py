"""Network unit-system detection for WNTR networks."""

_UNIT_MAP = {
    # flow_unit : (system, flow, diameter, length, pressure, velocity)
    "LPS":  ("SI", "LPS",  "mm", "m",  "m",   "m/s"),
    "LPM":  ("SI", "LPM",  "mm", "m",  "m",   "m/s"),
    "MLD":  ("SI", "MLD",  "mm", "m",  "m",   "m/s"),
    "CMH":  ("SI", "CMH",  "mm", "m",  "m",   "m/s"),
    "CMD":  ("SI", "CMD",  "mm", "m",  "m",   "m/s"),
    "CMS":  ("SI", "CMS",  "mm", "m",  "m",   "m/s"),
    "CFS":  ("US", "CFS",  "in", "ft", "psi", "ft/s"),
    "GPM":  ("US", "GPM",  "in", "ft", "psi", "ft/s"),
    "MGD":  ("US", "MGD",  "in", "ft", "psi", "ft/s"),
    "IMGD": ("US", "IMGD", "in", "ft", "psi", "ft/s"),
    "AFD":  ("US", "AFD",  "in", "ft", "psi", "ft/s"),
}


def detect_network_units(wn):
    """Return a dict describing the network's unit system.

    Mirrors the MATLAB `detectNetworkUnits` output. WNTR always stores
    internal quantities in SI (meters), so these units are used only
    for display and for converting the user's D.txt file.
    """
    flow = str(wn.options.hydraulic.inpfile_units).strip().upper()
    if flow not in _UNIT_MAP:
        raise ValueError(f"Unsupported network unit system: {flow!r}")

    system, flow_u, diam_u, len_u, pres_u, vel_u = _UNIT_MAP[flow]
    return {
        "FlowUnit": flow_u,
        "System": system,
        "DiameterUnit": diam_u,
        "LengthUnit": len_u,
        "PressureUnit": pres_u,
        "VelocityUnit": vel_u,
    }