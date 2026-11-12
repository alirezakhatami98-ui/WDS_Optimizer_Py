# NYTP — reserved for v0.2

The New York Tunnels Problem (NYTP) is one of the two canonical WDS
design benchmarks, but its Dandy et al. (1996) formulation requires
support for **parallel pipe duplication**, which the current tool
does not implement.

## The formulation difference

Dandy's NYTP formulation: each existing pipe can be duplicated with a
parallel pipe of a chosen diameter.  The combined capacity is the
Hazen-Williams parallel-pipe equivalent

    D_eq = (D_existing^2.63 + D_new^2.63)^(1/2.63)

and the cost is only for the new pipe.  The optimum design duplicates
pipe 15 (existing 204 in) with a 120-in pipe → D_eq ≈ 221 in.

Our tool implements a *replacement* formulation (choose one diameter,
pipe's diameter becomes that value), which cannot express
"existing + new = combined".

## Status

- v0.1.0: not implemented.  Users who need NYTP should use another
  tool or wait for v0.2.
- v0.2: planned feature.  Will land with `formulation="duplicate"`
  support in `build_problem`, `evaluate.py`, and a `formulation`
  parameter in `OptimizationRequest`.

## Reference

Dandy, G. C., Simpson, A. R., and Murphy, L. J. (1996). "An improved
genetic algorithm for pipe network optimization." *Water Resources
Research*, 32(2), 449-457.

Best-known cost: **$38.64M** (Maier et al. 2003; improved from
Dandy's own $38.80M by subsequent methods).