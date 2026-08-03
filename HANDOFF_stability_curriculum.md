# HANDOFF — Stability curriculum for `g1` (entropy runaway fix + Clone-from restructure)

> **Read this first.** Fresh session, no memory. This file + the repo state (this worktree,
> branch `policy-context-target-ui`) is your full continuity for this sub-thread.
> Companion doc: `HANDOFF_control_web.md` (the broader control-web architecture — read that
> too if you need context on the panel/service/supervisor layers this builds on).

## TL;DR (español)

Estábamos armando un "curriculum" de entrenamiento en pasos chicos para el policy `g1` en vez
de un salto agresivo desde cero — porque una corrida así (`crouch`, `base_height_target=0.45`,
sin overrides) **divergió**: el ruido de exploración de PPO (`Mean action noise std`) subió sin
parar toda la corrida (0.80→2.83) en vez de bajar, y el robot terminó moviendo las patas al
azar. Causa raíz: `entropy_coef` (peso del bonus de entropía en PPO) no tiene techo ni
decaimiento, y con el reward débil de esta task, gana la entropía y el std se dispara.

Arreglamos eso agregando `--entropy_coef` como flag configurable (CLI + UI), y de paso
arreglamos que `web_train.py` estaba tirando el override de `train_cfg` al piso (ver §2).
También reorganizamos cómo se guardan las policies entrenadas por la UI (`policies/<name>/`
con checkpoint + crudo juntos) para que Clone-from siempre funcione de acá en más (§3).

Corrimos dos pasos del curriculum (`stable_step_one`, `stable_step_two`) con `entropy_coef`
bajo y sin empujones — el runaway NO volvió a pasar (std sigue bajando, reward sube), **pero
el episode length se estancó en ~73 pasos de 1000 posibles** (se sigue cayendo a los ~1.5s de
20s posibles) entre las dos corridas. Ese estancamiento es el problema abierto — ver §6.

Todo el código está commiteado y pusheado en la rama `policy-context-target-ui` / PR #4. El
server vivo (worktree) tiene el código nuevo cargado. Detalles de cómo seguir abajo.

---

## 1. What was broken and how we found it

A training job (`crouch`, task `g1`, `--base_height_target 0.45`, no `--from_checkpoint`, no
entropy override — task default `entropy_coef=0.01`) ran 40 min / 3680 iterations and produced
a policy whose legs moved "ridiculously fast" and erratically once loaded. Diagnosis, from the
job's own log (`logs/_web_training/f3d2d365.log`):

| iteration | `Mean action noise std` | `Mean episode length` |
|---|---|---|
| 0 | 0.80 | ~17 |
| ~600–800 (best point) | 1.10–1.17 | ~50–57 |
| 2000 | 1.70 | ~30 |
| 3679 (final) | **2.83** | **~18** |

`std` climbed **monotonically, the entire run, never once decreasing** — textbook PPO
exploration-noise runaway: the entropy bonus (`algorithm.entropy_coef`) dominates a weak reward
gradient and keeps pushing exploration noise up instead of letting it anneal down as the policy
converges. There is no upper bound or decay on it anywhere in this codebase. Confirmed via the
actual reward function (`legged_gym/envs/base/legged_robot.py:722-728`,
`_reward_base_height` — squared-error tracking around a fixed setpoint, small magnitude,
easily dominated by the entropy term at `entropy_coef=0.01`,
`legged_gym/envs/g1/g1_config.py:59`).

**The fix is not "train less" — it's "give this a ceiling."** Watch `Mean action noise std` in
any job's log: it should trend down, never up. If it climbs the whole run, the policy gets
*more* erratic the longer it trains, not better.

## 2. Code changes made (all in this worktree, branch `policy-context-target-ui`)

### 2.1 `--entropy_coef` exposed end-to-end

- **`legged_gym/scripts/web_train.py`**: new `--entropy_coef` CLI arg; overrides
  `train_cfg.algorithm.entropy_coef` when given.
  - **Real bug fixed along the way**: `env_cfg, train_cfg = task_registry.get_cfgs(...)` was
    called, `env_cfg` was correctly re-passed into `make_env()`, but `train_cfg` was **never**
    passed into `make_alg_runner()` — that call re-fetched a fresh, unmodified `train_cfg` by
    task name, silently discarding any override to it (not just entropy_coef — this would've
    swallowed ANY future `train_cfg`-level flag too). Fixed by passing `train_cfg=train_cfg`
    explicitly.
- **`legged_gym/control/training.py`**: `TrainingManager.start()` gained an `entropy_coef`
  param, appended to the subprocess argv as `--entropy_coef <value>` when set; validated
  non-negative.
- **`legged_gym/control/service.py`**: `ControlService.start_training()` passes it through.
- **`web/index.html` + `web/app.js`**: new "Exploration noise (entropy_coef)" field in the
  Create Policy panel, right after Push disturbances. Full explanatory copy is in the HTML
  (`#train-entropy-coef`'s sibling `<p class="field-hint">`) — read it in-app rather than here,
  it's the single source of truth and was written to be self-explanatory to a non-ML person.

No `transport.py` changes needed — `start_training` was already whitelisted and dispatches via
`**params`, so a new optional kwarg needed no allowlist update.

### 2.2 `policies/<name>/` restructuring (Clone-from architecture fix)

Unrelated to the entropy bug but done in the same session, because Clone-from kept showing
everything greyed out — root cause was that pre-existing policies (`stable`, `crouch`,
`scratch_wobbly`) were exported checkpoints copied into a flat `policies/` folder, disconnected
from the `logs/<run>/` directory that had their raw `model_N.pt` (rsl_rl's own resumable
checkpoint format — NOT the same file as the deployable exported one; see
`TrainingManager._train_checkpoint_from_export()`'s docstring for why those are different
files and why passing the wrong one crashes `ppo_runner.load()`).

- **`legged_gym/scripts/web_train.py`**: after export, also reports the exact raw checkpoint it
  just wrote — `os.path.join(log_dir, f'model_{ppo_runner.current_learning_iteration}.pt')` —
  as `train_checkpoint_path` in `result.json`. No more guessing.
- **`legged_gym/control/training.py`**: new `TrainingManager.finalize_policy(name, task,
  checkpoint, train_checkpoint)` — copies both files into `POLICIES_DIR / name /
  {checkpoint.pt, train_checkpoint.pt, meta.json}` (copies, doesn't move — the original
  `logs/<run>/` tree with all its `model_N.pt` snapshots and TensorBoard events is untouched).
  `register_source()` now accepts an explicit `train_checkpoint` (skips the old
  directory-guessing fallback, `_train_checkpoint_from_export()`, which is kept only for
  `--policy name:path` CLI-registered sources this UI never produced — e.g. `stable`, which
  correctly stays un-fine-tunable forever, it has no raw training history anywhere).
  `forget_source()` now `shutil.rmtree`s the whole `policies/<name>/` folder when it detects
  that layout (checkpoint's parent dir == `POLICIES_DIR/name`), single-file removal otherwise
  (backward compat with CLI-registered sources).
- **`legged_gym/scripts/swap_experiment.py`**: `drain_finished_training()` now calls
  `finalize_policy()` BEFORE `load_policy()`, and loads from the returned (finalized) path —
  what's running always matches what's registered in the catalog.

**Gotcha for the next session**: `POLICIES_DIR = REPO_ROOT / "policies"` where `REPO_ROOT` is
derived from `training.py`'s own `__file__` — i.e. **wherever the live process's `legged_gym`
package was actually imported from**. The server has been running from inside THIS WORKTREE
(`.claude/worktrees/sim-tab-pause/`, see §4), so every policy finalized during this session
lives at `.claude/worktrees/sim-tab-pause/policies/<name>/`, **not** the main checkout's
`policies/` (`/Users/josetabuyo/Development/GIAR/LeggedGym-Ex/policies/`). Those are two
different folders. If this worktree gets removed/merged before those policies matter
elsewhere, copy them out first.

Existing tests (`tests/test_delete_policy.py`, `tests/test_training_estimate.py`,
`tests/test_control_transport.py`) all still pass unchanged.

### 2.3 Also in this branch, unrelated to training (earlier in the same session)

Drag-to-reorder the Policies list with shortcuts fixed to position (top=`0`, then `9`↓`1`) —
see the git log / PR description for that piece, it's orthogonal to everything in this file.

## 3. Deleted policies (dead ends, confirmed unrecoverable)

Searched **by content hash**, not just filename, across this entire repo, the worktree's own
`logs/`, `unitree_rl_gym`'s clone, and common stray locations (Downloads/Desktop) before
deleting anything:

| Policy | Verdict |
|---|---|
| `stable` | External `unitree_rl_gym` pretrained checkpoint (`deploy/pre_train/g1/motion.pt` — hash-verified identical). No raw training history anywhere, ever will. **Kept** — it's the one thing that actually works well; treat as reference, not as a Clone-from base. |
| `cautious`, `crouch` (original), `scratch_wobbly`, `undertrained_dummy`, `g1_crouch_stability` (original) | Deleted. Some had no raw checkpoint anywhere (hash search came up empty); others did have one recoverable at the time (`undertrained_dummy`, `g1_crouch_stability` — fixed and briefly usable) but were deleted anyway per explicit user instruction ("vamos a borrar todo menos estable... quiero que todas las que tengamos, tengan su crudo, porque las creamos nosotros"). |

Current catalog going into next session: `stable` (external reference) +
`stable_step_one` + `stable_step_two` (this session's curriculum attempts, both have proper
`policies/<name>/` folders with real `train_checkpoint.pt`).

## 4. Live server state (as of end of this session)

```
PID 39248 (may have exited/been restarted since — check `ps aux | grep swap_experiment`)
cwd / LEGGED_GYM_ROOT_DIR: /Users/josetabuyo/Development/GIAR/LeggedGym-Ex/.claude/worktrees/sim-tab-pause
launched via: SIMULATOR=genesis GENESIS_BACKEND=cpu PYTHONPATH=<worktree root> \
  /Users/josetabuyo/Development/GIAR/LeggedGym-Ex/.venv/bin/python3 -u legged_gym/scripts/swap_experiment.py \
  --policy stable:/Users/josetabuyo/Development/GIAR/LeggedGym-Ex/policies/stable.pt \
  --active stable --viser_port 9014 --control_port 9013 --speed 0.5 --ball
log: /Users/josetabuyo/.claude/jobs/3564e413/tmp/swap_experiment_current.log
```

**The venv matters**: `torch`/`genesis` are only importable via
`/Users/josetabuyo/Development/GIAR/LeggedGym-Ex/.venv/bin/python3` — the bare system Python at
the same-looking path (Cellar python 3.12) does NOT have them on its default sys.path. Also
needs `SIMULATOR=genesis` set (raises `ValueError` otherwise — see `legged_gym/__init__.py`).

To relaunch after code changes (process doesn't hot-reload): `kill <pid>`, then re-run the
launch command above with a fresh `nohup ... > <logfile> 2>&1 &`. Wait ~15-20s for Genesis env
setup before it's ready (`curl -s http://localhost:9013/config` returns real JSON once up).

To query it headlessly instead of the browser (handy for checking job status without
navigating the UI):
```python
import asyncio, json, websockets
async def main():
    async with websockets.connect("ws://localhost:9013/ws") as ws:
        for _ in range(3):
            msg = json.loads(await ws.recv())
            r = msg.get("result")
            if isinstance(r, dict) and "active" in r:
                print(json.dumps(r, indent=2)); break
asyncio.run(main())
```

## 5. Curriculum progress so far

Plan: nail down basic balance/stability WITHOUT pushes or a height target first, in short
supervised steps, before reintroducing disturbances or asking for a crouch — rather than one
long aggressive run (which is what diverged in §1).

| Job | From | `push_robots` | `entropy_coef` | Time | Final std | Final reward | **Final episode length** (/1000 max) |
|---|---|---|---|---|---|---|---|
| `stable_step_one` | scratch | off | 0.002 | 20 min / 1860 iters | 0.31 | 0.63 | **73.28** |
| `stable_step_two` | `stable_step_one`'s train_checkpoint | off | 0.001 | 20 min / 1870 iters (cumulative 3730) | 0.23 | 0.81 | **72.88** |

Episode length = 20s episode at 50Hz (`legged_robot_config.py:16`,
`env.episode_length_s=20`, `dt=0.02`) → **1000 steps is "survived the whole episode."** Both
runs fall after ~1.5s. `noise std` shrinking and `reward` rising are good signs the entropy fix
holds — but neither of those means "it stops falling less." That's the metric to check first,
always: last few lines of the job's `.log` file, `Mean episode length`.

## 6. Open problem — episode length plateaued (73.28 → 72.88), not yet diagnosed

40 minutes of training (two stages, entropy fix confirmed working) produced **zero measurable
improvement** in how long the robot stays upright. Reward went up, noise went down, survival
time didn't move. Untested hypotheses, roughly in the order I'd check them:

1. **`entropy_coef` may now be too low.** We kept lowering it (0.002 → 0.001) chasing a
   shrinking std, but if it's now too low to explore past whatever specific failure mode causes
   the fall at ~73 steps, the policy could be stuck in a local optimum instead of converging on
   a genuinely better gait. Try bumping it back up slightly (0.003–0.005) for the next
   iteration and see if episode length starts moving, even if std ticks up a little (that's
   fine — the failure mode we're avoiding is UNBOUNDED growth, not "any growth at all").
2. **Verify `--push_robots off` actually took effect** in both jobs — re-grep
   `logs/_web_training/*.log`'s command line (it's echoed at the top of the file, or check
   `training_jobs[].command` in a `status()` call) to confirm `env_cfg.domain_rand.push_robots`
   was really `False`, not silently still `True` (the default). Should be fine — the CLI flag
   plumbing was verified — but worth a first-principles check before spending more compute.
3. **`rew_alive` weight.** `G1RoughCfg.rewards.scales.alive = 0.15` (`g1_config.py`) — small
   relative to the penalty terms. Check whether it's actually pulling hard enough toward
   "survive longer" as an objective, or whether the policy is finding a local optimum that
   scores fine per-step without prioritizing episode length. Compare the per-term
   `Mean episode rew_*` breakdown between the two jobs' final iterations (both logs have it) —
   look for which term dominates and whether it correlates with early termination.
4. **Just needs more raw iterations.** `G1RoughCfgPPO.runner.max_iterations = 10000` is the
   task's own full-training target; we're at 3730 (37%). Could be that nothing is actually
   wrong and this specific plateau is just an early-training plateau that breaks given enough
   more time at the same settings — cheapest hypothesis to test, just expensive in wall-clock.
5. Consider comparing against training the task **from scratch with the DEFAULT
   entropy_coef=0.01** but capped at a similarly short budget, just to have a clean baseline of
   "what does normal (buggy-but-not-yet-diverged) short training look like" — useful signal for
   telling apart "entropy_coef too low" from "just needs more time" from hypothesis 1 vs 4.

Recommended next step for the following session: try hypothesis 1 first (cheap, one 20-min
run, `--from_checkpoint` = `stable_step_two`'s `train_checkpoint.pt`,
`--entropy_coef 0.003`ish, still `--push_robots off`), and actually read the full per-term
reward breakdown (hypothesis 3) from the two existing logs before spending any more compute —
that's free, the data already exists.

## 7. How the user wants to keep working (preferences, stated explicitly this session)

- Curriculum style: **short steps, one lesson at a time**, review results together between
  each step before deciding the next — not long unsupervised runs. "como lo haríamos con un ser
  al que le enseñamos, primero una lección y luego otra."
- Wants concrete, checkable numbers, not vibes — asked "cómo sabemos si mejoró o no mejoró la
  estabilidad" and wanted the actual before/after metric, not a qualitative answer. Keep
  reporting the literal `Mean episode length` comparison, not just "looks better."
- Reviews every proposed CLI command line-by-line before running it and catches missing flags
  himself (caught a missing `--push_robots off` and a missing `--entropy_coef` on his own,
  twice) — keep composing full exact commands for him to review/paste rather than describing
  parameters abstractly.
- Wants any reusable knob to live in the UI, not be a one-off CLI-only thing (this is why
  `entropy_coef` became a real form field with a full plain-language explanation instead of
  just documented as a flag).
