"""Sand, Solvers, and World Models: a quadruped in Newton, from solver bake-off to Cosmos 3 critic.

Modules:

* :mod:`ssw.device`   - ``--device auto`` selection (cuda:0 if present, else cpu) and CUDA-graph checks
* :mod:`ssw.metrics`  - pure NumPy metrics (energy, settling, timing, judge scoring, control fidelity)
* :mod:`ssw.bakeoff`  - Phase 0 solver bake-off on ``quadruped.urdf`` (GPU target with CUDA graphs; CPU run executed)
* :mod:`ssw.terrain`  - Phase 1 ANYmal C on rigid vs MPM terrain (GPU target; CPU smoke tests run, GPU sweep not yet)
* :mod:`ssw.judge`    - Phase 2 Cosmos 3 Reasoner critic (dry-run and parsing tested; server not run)
* :mod:`ssw.transfer` - Phase 2 Cosmos 3 transfer requests (dry-run tested; generation not run)

Newton conventions apply throughout: Z-up, quaternions in XYZW order.
"""

__version__ = "0.1.0"
