Reference files vendored from `unitreerobotics/unitree_rl_mjlab`
(Apache-2.0), commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`, cloned
2026-08-17.

**Not installed as a dependency** — see `docs/mjlab_migration.md` §0 for why
(unitree's fork is a near-verbatim copy of `mjlab`'s own tracking task,
pinned to older `mjlab`/`mujoco-warp` versions, and not pip-installable as a
library). These three files are kept as read-only reference for the
migration, not imported by any code in this repo:

- `csv_to_npz.py.reference` — the CSV/pkl → mjlab-NPZ motion converter our
  own `legged_gym/scripts/process_reference_motion_mjlab.py` is ported from
  (Phase 1 of the migration).
- `State_Mimic.cpp.reference` — Unitree's real-robot C++ implementation of
  the 154-dim tracking observation vector. Reference only for anyone who
  later builds the real-robot obs bridge (`deploy_real/`) — explicitly out
  of scope for this migration, see `docs/mjlab_migration.md` R6.
- `g1_mimic_deploy.yaml.reference` — term-by-term written spec of that same
  obs vector (stiffness/damping/default_joint_pos/action-scale arrays),
  cross-checked against `mjlab`'s own observation manager output in Phase 0.

`resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz` (a
sibling of this directory) is also vendored from the same commit/license —
a real, working 29-DoF G1 motion (6574 frames @ 50fps) used as the Phase 0
smoke-test fixture and the Phase 4 validation motion for Javier Villalba's
`javier_mjlab_dance1_subject2` checkpoint.
