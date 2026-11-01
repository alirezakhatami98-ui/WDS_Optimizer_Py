"""Verify that node coordinates are present in the .inp file."""

import wntr
from pathlib import Path

net = Path(__file__).resolve().parent.parent / "networks" / "2loops" / "2Loops.inp"
wn = wntr.network.WaterNetworkModel(str(net))

print("Node coordinates:")
any_nonzero = False
for n in wn.node_name_list:
    xy = wn.get_node(n).coordinates
    print(f"  {n:>5s} : {xy}")
    if xy != (0.0, 0.0) and xy is not None:
        any_nonzero = True

print()
if any_nonzero:
    print("[OK] Coordinates are defined. Plotly map will look correct.")
else:
    print("[WARN] All coordinates are (0, 0). Map will be useless without layout.")
    print("       Fix: ensure [COORDINATES] section exists in 2Loops.inp.")