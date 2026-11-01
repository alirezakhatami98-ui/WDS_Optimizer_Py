# WDS Optimizer (Python port)

Water distribution network diameter optimization using
Genetic Algorithm and Particle Swarm Optimization, powered by WNTR/EPANET.

## Origin

Python port of the MATLAB **WDS Optimization Toolkit** originally built in
MATLAB App Designer.

## Roadmap

- [x] **Phase 1** — Core algorithm engine (GA + PSO), WNTR hydraulics
- [ ] **Phase 2** — Streamlit GUI with interactive network map
- [ ] **Phase 3** — Multi-run experiments + parameter sweep + persistence
- [ ] **Phase 4** — Reproducibility report + Excel export
- [ ] **Phase 5** — Packaging (Tauri / PyInstaller) + release

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"

python scripts/run_2loops.py
```

## Benchmark validation

Two-loops network (Alperovits & Shamir 1973):

| Algorithm | Cost |
|---|---|
| GA (tournament) | 419,000 |
| PSO | 441,000 |
| Literature optimum | 419,000 |

## Folder structure

```
src/wds_optimizer/
├── algorithms/      # GA, PSO, selection, comparison
├── hydraulics/      # WNTR simulation, units
├── data/            # File loaders
└── optimization/    # Problem, cost, constraints, cache

networks/
└── 2loops/          # Benchmark network

scripts/
└── run_2loops.py    # Smoke test
```

## License

```
Copyright (c) 2026 Alireza Khatami

All rights reserved.

This repository and its contents, including source code, documentation, algorithms, and related materials, are proprietary to the copyright holder.

No license or permission is granted to any person or entity to use, copy, modify, distribute, sublicense, publish, or commercially exploit any part of this project without prior written permission from the copyright holder.

Access to this repository does not, by itself, grant any rights to use, reproduce, distribute, or commercialize its contents.

Third-party components remain subject to their respective licenses.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW.
```