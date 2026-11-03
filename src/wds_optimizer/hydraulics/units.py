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

# ---------------------------------------------------------------------
#  Unit conversions
# ---------------------------------------------------------------------
# WNTR always returns pressure in meters of water head and velocity in
# m/s, regardless of the network's `inpfile_units`. These helpers convert
# between WNTR's SI and the network's display units.

_PRESSURE_TO_M = {
    "m":   1.0,
    "psi": 0.70307,   # psi -> m water head (at ~4°C)
    "ft":  0.3048,
    "kPa": 0.101972,
    "bar": 10.1972,
}

_VELOCITY_TO_MS = {
    "m/s":  1.0,
    "ft/s": 0.3048,
    "mph":  0.44704,
}

_DIAMETER_TO_M = {
    "m":  1.0,
    "mm": 1e-3,
    "in": 0.0254,
    "ft": 0.3048,
}


def pressure_to_si(value, units):
    """Convert a pressure value from network units to meters."""
    unit = units.get("PressureUnit", "m")
    return float(value) * _PRESSURE_TO_M.get(unit, 1.0)


def pressure_from_si(value_m, units):
    """Convert a pressure value from meters to network units."""
    unit = units.get("PressureUnit", "m")
    return float(value_m) / _PRESSURE_TO_M.get(unit, 1.0)


def velocity_to_si(value, units):
    """Convert a velocity value from network units to m/s."""
    unit = units.get("VelocityUnit", "m/s")
    return float(value) * _VELOCITY_TO_MS.get(unit, 1.0)


def velocity_from_si(value_ms, units):
    """Convert a velocity value from m/s to network units."""
    unit = units.get("VelocityUnit", "m/s")
    return float(value_ms) / _VELOCITY_TO_MS.get(unit, 1.0)


def diameter_to_display(value_m, units):
    """Convert a diameter in meters to the network's display unit."""
    unit = units.get("DiameterUnit", "mm")
    return float(value_m) / _DIAMETER_TO_M.get(unit, 1e-3)