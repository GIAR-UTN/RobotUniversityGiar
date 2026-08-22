# mjlab training backend — implementation contract (S4)

Frozen contract for **S5** (`legged_gym/scripts/mjlab_train.py`, the new
subprocess entrypoint) and **S6** (`TrainingManager.start()` dispatch).
Everything below was verified against the code actually on disk on this
machine (`.venv-mjlab` = rsl-rl-lib **5.4.2**, mjlab from the same venv) and,
where marked ✅ EMPIRICAL, by running it. Implement it as written; do not
re-derive.

Background/scope: item 4 of `HANDOFF_mimic_motion_library_ux.md`. Local
training only — the Kaggle backend is explicitly out of scope
(`backend="kaggle"` + an mjlab task must be rejected in `start()`, see §5).

---

## 1. Runner/registry resolution

The entrypoint reproduces `mjlab/scripts/train.py::run_train()` minus tyro,
wandb and torchrunx. Exact calls, in this order:

```python
import mjlab_tasks                       # side effect: registers "Rugiar-G1-Mimic"
from mjlab.tasks import registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from dataclasses import asdict

env_cfg   = registry.load_env_cfg(task)        # NOT play=True — play cfg is for the driver
agent_cfg = registry.load_rl_cfg(task)         # RslRlOnPolicyRunnerCfg (deep copy, safe to mutate)
runner_cls = registry.load_runner_cls(task) or MjlabOnPolicyRunner
```

`registry.load_runner_cls("Rugiar-G1-Mimic")` returns
`mjlab.tasks.tracking.rl.runner.MotionTrackingOnPolicyRunner` ✅ EMPIRICAL —
it is registered that way in `mjlab_tasks/tracking/__init__.py`. **Never
hardcode `MjlabOnPolicyRunner` as the runner class**; use `load_runner_cls`
with the `or MjlabOnPolicyRunner` fallback exactly as above.

Mandatory cfg mutations before constructing anything:

| what | exact assignment | why |
|---|---|---|
| env count | `env_cfg.scene.num_envs = cli.num_envs` | registered default is 1 ✅ EMPIRICAL |
| motion clip | `env_cfg.commands["motion"].motion_file = <abs path>` | registered default is `""`; mjlab raises without it |
| seeds | `agent_cfg.seed = cli.seed; env_cfg.seed = cli.seed` | mirrors `run_train()` |
| logger | `agent_cfg.logger = "tensorboard"` | **cfg default is `"wandb"`** ✅ EMPIRICAL — leaving it would require a wandb login |
| model upload | `agent_cfg.upload_model = False` | `MjlabOnPolicyRunner.save()` calls `self.logger.save_model()` when true |
| run label | `agent_cfg.run_name = cli.name` | run dir suffix |
| iterations | `agent_cfg.max_iterations = <effective cap>` | cosmetic (ETA line); the real budget is §7 |

`torch.utils.tensorboard` imports fine in `.venv-mjlab` ✅ EMPIRICAL. This
matters: rsl-rl gates **printing AND checkpoint saving** on
`logger.writer is not None` (`rsl_rl/runners/on_policy_runner.py:132-134`),
which requires `log_dir is not None` and a constructible writer.

Construction:

```python
configure_torch_backends()
env = ManagerBasedRlEnv(cfg=env_cfg, device=cli.device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = runner_cls(env, asdict(agent_cfg), str(log_dir), cli.device)
```

Do **not** pass `registry_name=` — `MotionTrackingOnPolicyRunner.__init__`
defaults it to `None`, which is what we want (no wandb registry).

`log_dir` (mirrors mjlab's own layout so nothing else in the repo has to
learn a second convention):

```python
log_dir = REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{cli.name}")
```

`experiment_name` for `Rugiar-G1-Mimic` is `g1_mjlab_mimic` ✅ EMPIRICAL.

Resume (`--from_checkpoint`):

```python
runner.load(cli.from_checkpoint, load_cfg={
    "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": True})
```

`optimizer: False` mirrors `web_train.py`'s "optimizer state NOT carried
over"; `iteration: False` keeps `runner.current_learning_iteration == 0` so
this job's budget accounting starts at zero (see `rsl_rl/algorithms/ppo.py:372-395`).

---

## 2. CLI surface — `legged_gym/scripts/mjlab_train.py`

**File header requirement (non-negotiable):** before any
`mjlab`/`rsl_rl`/`legged_gym` import, reproduce
`rugiar_driver_mjlab.py:53-66` verbatim:

```python
REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path = [p for p in sys.path if p not in ("", ".", REPO_ROOT)] + [REPO_ROOT]
os.environ.setdefault("SIMULATOR", "mjlab")
```

Without the reorder, this repo's vendored top-level `rsl_rl/` shadows PyPI
`rsl-rl-lib` and the run dies (docs/mjlab_migration.md R1). This is also what
makes an inherited `PYTHONPATH=<repo root>` harmless (§5).

### Accepted flags

| flag | type | required | effect |
|---|---|---|---|
| `--task` | str | yes | mjlab registry task id |
| `--name` | str | yes | display/policy name; also `agent_cfg.run_name` and the log-dir suffix |
| `--max_iterations` | int | one of the two | §7 |
| `--max_minutes` | float | one of the two | §7 |
| `--num_envs` | int, default 64 | no | `env_cfg.scene.num_envs` |
| `--motion_file` | str | **yes for any task whose `env_cfg.commands` contains `"motion"`** | repo-root-relative or absolute path to a `.npz`; resolved to absolute before assignment |
| `--device` | str, default `"cpu"` | no | passed to `ManagerBasedRlEnv` and the runner (`"cpu"` / `"cuda:0"`) |
| `--from_checkpoint` | str | no | §1 resume |
| `--reward_scale NAME VALUE` | 2 strings, `action="append"` | no | §6 |
| `--entropy_coef` | float | no | `agent_cfg.algorithm.entropy_coef` |
| `--seed` | int, default `agent_cfg.seed` (42) | no | §1 |
| `--headless` | `store_true`, default True | no | **accepted and ignored** — training never opens a viewer; exists only so the shared flag list in `TrainingManager` need not special-case it |
| `--result_path` | str | yes | §3 |
| `--progress_path` | str | no | §3 |

`--max_iterations`/`--max_minutes`: if both are `None`,
`parser.error("give at least one of --max_iterations / --max_minutes")`
(same wording as `web_train.py:139`).

Missing motion file for a tracking task:

```
parser.error(f"--motion_file is required for tracking task '{task}' "
             f"(its command term has no default clip)")
```

Unknown reward term (validated against `env_cfg.rewards`, a **dict**):

```
parser.error(f"unknown reward term '{name}' for mjlab task '{task}' — "
             f"valid terms: {', '.join(sorted(env_cfg.rewards))}")
```

### Rejected flags (accepted by argparse, then errored)

These exist on `web_train.py` and have **no mjlab analogue**. Declare each
one with `default=None` / `action="store_true"` so a stale caller gets a
clear message instead of `unrecognized arguments`:

`--cmd_vx_range`, `--cmd_vy_range`, `--cmd_yaw_range`, `--base_height_target`,
`--lin_vel_z_target`, `--ang_vel_xy_target`, `--orientation_tilt_target`,
`--push_robots`, `--max_push_vel_xy`, `--push_interval_s`, `--push_dir`,
`--cpu`, `--gpu`.

Single check after parsing, one message listing everything supplied:

```python
INAPPLICABLE = ("cmd_vx_range", "cmd_vy_range", "cmd_yaw_range", "base_height_target",
                "lin_vel_z_target", "ang_vel_xy_target", "orientation_tilt_target",
                "push_robots", "max_push_vel_xy", "push_interval_s", "push_dir")
supplied = [f"--{n}" for n in INAPPLICABLE if getattr(cli, n) not in (None, False)]
if supplied:
    parser.error(f"{', '.join(supplied)} do not apply to mjlab task '{cli.task}' — "
                 f"a motion-tracking task has no velocity-command envelope, no "
                 f"base-height/tilt reward targets and no push domain randomization "
                 f"(see legged_gym/control/mjlab_adapter.py). Use --motion_file and "
                 f"--reward_scale instead.")
```

`--cpu`/`--gpu` are handled separately: `--cpu` is a silent no-op (device
already defaults to cpu), `--gpu` errors with
`"--gpu is not supported by mjlab_train.py; pass --device cuda:0"`.
Exit code is argparse's `2` in every case; `TrainingManager` surfaces it as
`"mjlab_train.py exited with code 2 — see <log>"`.

---

## 3. Progress / result / log contract

### 3a. `--progress_path` JSON (extended, never renamed)

`TrainingManager._refresh_progress()` (`training.py:1481-1501`) reads exactly
one key: `iterations_done`. The existing three keys keep their exact names
and meanings; the rest are additive and safe to ignore.

```json
{
  "iterations_done": 120,
  "elapsed_s": 93.4,
  "updated_at": 1787150380.12,

  "iteration": 119,
  "reward": -1.92,
  "episode_length": 12.33,
  "noise_std": 1.0,
  "reward_terms": {"motion_body_pos": 0.0257, "action_rate_l2": -0.1152, "...": 0.0},
  "backend": "mjlab"
}
```

- `iterations_done` — iterations **this job actually ran** (our own counter,
  §7), not `runner.current_learning_iteration`.
- `iteration` — absolute rsl-rl iteration index (`it` as printed).
- `reward` / `episode_length` — `statistics.mean(runner.logger.rewbuffer)` /
  `lenbuffer`, omitted (key absent) while those deques are empty.
- `noise_std` — `action_std.mean().item()`, i.e. the same number printed as
  `Mean action std:`.
- `reward_terms` — `Episode_Reward/<term>` means with the prefix stripped.
- Written best-effort inside `try/except OSError: pass`, exactly like
  `web_train.py::write_progress()`.

**How to get `reward_terms`:** `rsl_rl/utils/logger.py:288` calls
`self.ep_extras.clear()` at the end of every `log()` ✅ EMPIRICAL (the list is
empty between `learn()` chunks — reading it after `learn()` returns yields
nothing). So wrap the logger's `log()` once, right after runner construction:

```python
_orig_log = runner.logger.log

def _log_and_report(**kwargs):          # rsl-rl calls log() with keyword args only
    terms = {}
    for entry in runner.logger.ep_extras:          # snapshot BEFORE log() clears it
        for key, value in entry.items():
            if key.startswith("Episode_Reward/"):
                terms.setdefault(key[len("Episode_Reward/"):], []).append(float(value.mean()))
    _orig_log(**kwargs)                            # prints + tensorboard + clears ep_extras
    state["iteration"] = kwargs["it"]
    state["noise_std"] = float(kwargs["action_std"].mean().item())
    state["reward_terms"] = {k: sum(v) / len(v) for k, v in terms.items()}
    write_progress()

runner.logger.log = _log_and_report
```

This yields one progress write per **iteration** (not per chunk) and the
values match the printed block exactly, because both read the same window.

### 3b. `--result_path` JSON

Byte-for-byte the same keys `poll()` reads (`training.py:1562-1571`), plus
two additive ones:

```json
{
  "policy_path": "<log_dir>/exported/policy.onnx",
  "task": "Rugiar-G1-Mimic",
  "name": "my_mimic_policy",
  "iterations_done": 500,
  "stopped_reason": "max_minutes",
  "train_checkpoint_path": "<log_dir>/model_499.pt",
  "motion_file": "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz",
  "simulator": "mjlab"
}
```

`stopped_reason` ∈ `{"max_iterations", "max_minutes"}`.
`train_checkpoint_path` is `os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")`,
or `None` if that file does not exist. It **does** exist: `learn()` saves
`model_{current_learning_iteration}.pt` at the end of every call ✅ EMPIRICAL
(`model_0.pt`, `model_1.pt`, `model_2.pt` observed in a 2-chunk probe run).

### 3c. Log lines the entrypoint itself must print

rsl-rl prints its own per-iteration block; on top of that print exactly
(stdout, the process runs under `python -u`, stdout+stderr are the job log):

```
[mjlab_train] task='Rugiar-G1-Mimic' num_envs=64 device=cpu motion=<abs path> log_dir=<log_dir>
Fine-tuning from: <path> (optimizer state NOT carried over)      # only with --from_checkpoint
Stopped after <N> iterations (<stopped_reason>).
Done. Exported to <policy_path>
```

The last two are the same strings `web_train.py:247,300` prints; keep them
identical so any future log tooling stays backend-agnostic.

### 3d. `parse_training_log()` changes (`legged_gym/control/training.py:90-96`)

rsl-rl 5.x prints `Mean action std:` (not `Mean action noise std:`) and
`Episode_Reward/<term>:` (not `Mean episode rew_<term>:`) ✅ EMPIRICAL. Two
regex edits, no structural change — `_ITER_RE`, `Mean reward:` and
`Mean episode length:` are already identical in both stacks:

```python
_STAT_RES = {
    # "noise " is optional: this repo's vendored rsl_rl prints "Mean action noise std:",
    # rsl-rl-lib 5.x (.venv-mjlab) prints "Mean action std:".
    "noise_std": re.compile(r"Mean action (?:noise )?std:\s*([-\d.]+)"),
    "reward": re.compile(r"Mean reward:\s*([-\d.]+)"),
    "episode_length": re.compile(r"Mean episode length:\s*([-\d.]+)"),
}
# Genesis: "Mean episode rew_<term>: <v>"   mjlab/rsl-rl 5.x: "Episode_Reward/<term>: <v>"
_TERM_RE = re.compile(r"(?:Mean episode rew_(\w+)|Episode_Reward/(\w+)):\s*([-\d.]+)")
```

and at the single call site (`training.py:153-155`):

```python
m = _TERM_RE.search(line)
if m:
    current["terms"][m.group(1) or m.group(2)] = float(m.group(3))
```

Nothing else changes: `parse_training_log()`'s return shape, the `rew_<term>`
series keys, `final`, `final_reward_terms`, `TrainingJob.to_dict()` and
`web/app.js` all stay as-is. `Episode_Termination/*` and `Metrics/*` lines are
correctly ignored (they don't match the literal `Episode_Reward/` prefix).

---

## 4. ONNX export

`MotionTrackingOnPolicyRunner.save()` auto-exports a 2-input/7-output ONNX
(`obs`,`time_step` → `actions` + 6 motion tensors) into
`<log_dir>/<log_dir_name>.onnx`. That file **must not** be used as
`policy_path`: `load_onnx_backend()` (`policy.py:186-188`) routes any 2+-input
graph to `OnnxExplicitStatePolicy`, which would feed a body-position output
back in as `time_step` — silently wrong, no crash.

Export a stateless graph instead, by calling the **parent class's** method
explicitly (this bypasses the tracking override, which is the whole point):

```python
from mjlab.rl import MjlabOnPolicyRunner
export_dir = os.path.join(log_dir, "exported")
MjlabOnPolicyRunner.export_policy_to_onnx(runner, export_dir, "policy.onnx")
policy_path = os.path.join(export_dir, "policy.onnx")
```

✅ EMPIRICAL on a real 2-chunk `Rugiar-G1-Mimic` run:
inputs `[('obs', [1, 154])]`, outputs `[('actions', [1, 29])]`. Observation
normalization is baked into the exported graph (`_OnnxMLPModel` deep-copies
`obs_normalizer`), and 154 is exactly `MjlabAdapter.OBS_GROUP == "actor"`,
the tensor `rugiar_driver_mjlab.py` feeds at inference.

**`legged_gym/control/policy.py` needs ZERO changes** — one input ⇒
`OnnxStatelessPolicy`, the same path Javier's checkpoints already use.
`OnnxPhaseConditionedPolicy` stays deferred future work; do not write it here.

`<log_dir>/exported/` is also the directory convention
`TrainingManager._train_checkpoint_from_export()` and `finalize_policy()`'s
`source_log_dir` already assume — keep the name exactly `exported`.

---

## 5. Dispatch logic (`TrainingManager.start()`)

### 5a. Which backend a task belongs to

Neither venv can import the other's registry ✅ EMPIRICAL
(`.venv-mjlab` → `from legged_gym.utils import task_registry` raises
`ImportError: cannot import name 'ActorCriticTSDepth'`; `.venv` →
`import mjlab` raises `ModuleNotFoundError`). Add **one module-level helper**
to `legged_gym/control/training.py` (that module already imports cleanly in
both venvs — the mjlab driver constructs `TrainingManager()` today):

```python
def _mjlab_registered_tasks():
    """Task ids in mjlab's registry, or None if mjlab isn't importable here."""
    try:
        import mjlab_tasks  # noqa: F401 - registers this repo's own tasks
        from mjlab.tasks import registry
    except ImportError:
        return None
    tasks = registry.list_tasks()
    return set(tasks) if tasks else None      # CORRECTION 2b, see below


def _legged_gym_registered_tasks():
    """Genesis/Isaac task names, or None if legged_gym's registry isn't importable
    here (the normal case under .venv-mjlab — same try/except ImportError shape as
    ControlService._registered_task_names(), service.py:473-486)."""
    try:
        import legged_gym.envs  # noqa: F401 - CORRECTION 2a: REQUIRED side effect
        from legged_gym.utils import task_registry
    except ImportError:
        return None
    tasks = getattr(task_registry, "task_classes", None)
    if tasks is None:                          # CORRECTION 2c
        tasks = getattr(getattr(task_registry, "task_registry", None), "task_classes", None)
    return set(tasks) if tasks else None       # CORRECTION 2b


def training_backend_for_task(task: str) -> str:
    """'mjlab' | 'genesis' — which training entrypoint/interpreter `task` needs.
    Mirrors rugiar_driver_mjlab.py's _script_for_task() asymmetry."""
    mjlab_tasks_ = _mjlab_registered_tasks()
    if mjlab_tasks_ is not None:                 # we ARE in .venv-mjlab: authoritative
        return "mjlab" if task in mjlab_tasks_ else "genesis"
    genesis_tasks = _legged_gym_registered_tasks()
    if genesis_tasks is not None:                # we ARE in .venv: authoritative by exclusion
        return "genesis" if task in genesis_tasks else "mjlab"
    raise ValueError(f"cannot determine a training backend for task '{task}': neither "
                     f"mjlab's nor legged_gym's task registry is importable here")
```

Decision table (no other branches, no `meta.json` sniffing needed — the
policy catalog is only what makes an mjlab task *visible* from a Genesis
session, via `ControlService._switchable_families()`, service.py:488-506):

| running under | task in mjlab registry | task in legged_gym registry | result |
|---|---|---|---|
| `.venv-mjlab` | yes | (unimportable) | `mjlab` |
| `.venv-mjlab` | no | (unimportable) | `genesis` |
| `.venv` | (unimportable) | yes | `genesis` |
| `.venv` | (unimportable) | no | `mjlab` |
| neither importable | — | — | `ValueError` |

> **CORRECTION 2 (S6)** — three defects in the helpers above, all found by
> the S6 tests, all capable of misrouting a *Genesis* job to the mjlab
> entrypoint:
>
> **2a — `import legged_gym.envs` is mandatory, not optional.**
> `task_registry` starts empty; every `task_registry.register()` call lives
> in `legged_gym/envs/__init__.py`. A bare
> `from legged_gym.utils import task_registry` in a process that hasn't
> imported `envs` yields an **empty** dict, so `training_backend_for_task()`
> answered `"mjlab"` by exclusion for *every* Genesis task — `g1` included.
> The long-running drivers import `envs` before ever calling this, which is
> exactly what would have hidden it in manual testing.
>
> **2b — empty means "unreadable", never "no such task".** Both probes must
> return `None` (not an empty set) when the registry reads empty, or 2a-style
> misrouting returns through a different door.
>
> **2c — the re-export resolves to either a module or an instance.**
> `from legged_gym.utils import task_registry` normally yields the
> `TaskRegistry` *instance* re-exported by `legged_gym/utils/__init__.py`,
> but resolves to the *submodule* of the same name when that `__init__`
> hasn't run its re-export (e.g. a partially-stubbed `legged_gym`, as several
> tests in `tests/` install). Accept both shapes.
>
> **CORRECTION 3 (§5b).** The "is `self.python_exe` inside `.venv-mjlab`?"
> check must compare the **unresolved** path. A venv's `bin/python` is a
> symlink to the base interpreter, so `Path(interp).resolve()` throws away
> the very `.venv-mjlab` marker the check looks for, and a Genesis job
> launched from an mjlab session kept the mjlab interpreter.

### 5b. Interpreter, script, env

```python
MJLAB_TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "mjlab_train.py"
MJLAB_PYTHON       = REPO_ROOT / ".venv-mjlab" / "bin" / "python"
GENESIS_PYTHON     = REPO_ROOT / ".venv" / "bin" / "python"
```

| | mjlab job | genesis job (unchanged behavior) |
|---|---|---|
| script | `MJLAB_TRAIN_SCRIPT` | `TRAIN_SCRIPT` (`web_train.py`) |
| interpreter | `MJLAB_PYTHON`; if missing → `ValueError(f"no mjlab venv at {MJLAB_PYTHON} — mjlab training isn't set up on this machine")` | `self.python_exe`, **except** when `self.python_exe` resolves inside `.venv-mjlab`, in which case `GENESIS_PYTHON` (mirrors `rugiar_driver_mjlab.py:169-177`, including its "venv missing → refuse" behavior) |
| argv prefix | `[interp, "-u", str(script), "--result_path", …, "--progress_path", …]` (no `-I`) | `[interp, "-u", str(TRAIN_SCRIPT), "--headless", "--cpu", "--result_path", …, "--progress_path", …]` |
| `cwd` | `str(REPO_ROOT)` | `str(REPO_ROOT)` |
| `SIMULATOR` | `"mjlab"` | `"genesis"` (set explicitly — an mjlab session's inherited `SIMULATOR=mjlab` would otherwise make `legged_gym/__init__.py` skip the Genesis import in the child) |
| `PYTHONPATH` | **do not prepend `REPO_ROOT`**; strip it from any inherited value (`os.pathsep`-split filter). The entrypoint's own `sys.path` reorder re-appends it *last* so PyPI `rsl-rl-lib` wins over the vendored `rsl_rl/` (R1) | `str(REPO_ROOT) + (os.pathsep + existing if existing else "")` — exactly as today (`training.py:1346-1348`) |
| `CUDA_VISIBLE_DEVICES` | `""` when the job is CPU (the default; matches `rugiar_driver_mjlab.py`'s documented invocation and mjlab's own `run_train()` cpu detection). Only leave the inherited value when a CUDA `--device` was requested | untouched |

`job.command` (the copy-pasteable rugiar string) keeps its current shape;
only the flag list differs (§2's accepted set). `job.simulator = "mjlab"`
for these jobs — extend `TrainingJob.simulator`'s docstring to
`"genesis" | "isaacgym" | "mjlab"`.

### 5c. Required guards/edits in `TrainingManager.start()`

1. `backend = training_backend_for_task(task)` computed **after** the
   name/budget validation, **before** the reward-scale validation.
2. `if backend == "mjlab" and backend_arg == "kaggle": raise ValueError(
   "the Kaggle backend doesn't support mjlab tasks (its bootstrap is IsaacGym-specific)")`.
   (Rename the existing `backend` parameter or the new local — do not shadow.)
3. Inapplicable knobs: if the caller passed any of `cmd_vx`, `cmd_vy`,
   `cmd_yaw`, `base_height_target`, `lin_vel_z_target`, `ang_vel_xy_target`,
   `orientation_tilt_target`, `push_robots`, `max_push_vel_xy`,
   `push_interval_s`, `push_dir` for an mjlab task →
   `raise ValueError(f"{', '.join(names)} don't apply to mjlab task '{task}' "
   f"(motion-tracking task: no velocity command, no stability targets, no pushes)")`.
   Fail here, not in the subprocess, so the panel shows it immediately (same
   reasoning as the existing reward-scale pre-validation, `training.py:1186-1194`).
4. `motion_file` becomes **required** for an mjlab tracking task:
   `raise ValueError(f"task '{task}' needs a --motion_file (reference-motion clip)")`.
   `TrainingManager.start()` already accepts `motion_file` and forwards it as
   `--motion_file` (`training.py:1274-1275`) — reuse that, unchanged.
   (`ControlService.start_training()` does **not** forward `motion_file`
   today, `service.py:318-377`; adding that parameter is part of S6.)
5. Reward-scale validation must not import legged_gym's registry under
   `.venv-mjlab`. In `task_defaults()` (`training.py:1109-1142`) wrap the
   `from legged_gym.utils import task_registry` in `try/except ImportError`,
   and add an mjlab branch:
   ```python
   if training_backend_for_task(task) == "mjlab" and _mjlab_registered_tasks() is not None:
       import mjlab_tasks  # noqa: F401
       from mjlab.tasks import registry
       cfg = registry.load_env_cfg(task)
       reward_scales = {name: term.weight for name, term in cfg.rewards.items()}
       variables = {key: {"reference": None} for key in self.VARIABLE_REGISTRY}
   ```
   and in `start()` change `unknown = [...]` to be skipped when `known` is
   empty (empty means "this process can't validate", not "all names invalid").
6. Same `try/except ImportError` treatment for `catalog()`'s bare
   `from legged_gym.utils import task_registry` (`training.py:1050`) — under
   `.venv-mjlab` it raises today, so the Create Policy panel is currently
   broken on an mjlab session. `all_tasks` falls back to
   `sorted(_mjlab_registered_tasks() or [])`.
7. `finalize_policy()` (`training.py:~660`) hardcodes
   `dest_checkpoint = dest_dir / "checkpoint.pt"`. Our export is `.onnx`, and
   `load_policy_backend()` dispatches on the file suffix — copying it to
   `checkpoint.pt` would break loading. Minimal fix:
   ```python
   suffix = ".onnx" if str(checkpoint).endswith(".onnx") else ".pt"
   dest_checkpoint = dest_dir / f"checkpoint{suffix}"
   ```
   `discover_local_policies()` already accepts `checkpoint.onnx`
   (`training.py:~905`), so nothing else changes.

Reward-term names for `Rugiar-G1-Mimic` (for the panel / validation)
✅ EMPIRICAL: `motion_global_root_pos` 0.5, `motion_global_root_ori` 0.5,
`motion_body_pos` 1.0, `motion_body_ori` 1.0, `motion_body_lin_vel` 1.0,
`motion_body_ang_vel` 1.0, `action_rate_l2` −0.1, `joint_limit` −10.0,
`self_collisions` −10.0.

---

## 6. Reward-scale override mapping

Genesis: `env_cfg.rewards.scales.<term>` is a class attribute, validated with
`hasattr` (`web_train.py:168-173`).
mjlab: `env_cfg.rewards` is a **plain `dict`** of term name →
`RewardTermCfg` with a `.weight` float ✅ EMPIRICAL. So:

```python
for name, value in (cli.reward_scale or []):
    if name not in env_cfg.rewards:
        parser.error(f"unknown reward term '{name}' for mjlab task '{cli.task}' — "
                     f"valid terms: {', '.join(sorted(env_cfg.rewards))}")
    env_cfg.rewards[name].weight = float(value)
```

Applied **before** `ManagerBasedRlEnv(cfg=env_cfg, …)` (the `RewardManager`
reads weights at construction). Sign convention is identical to Genesis
(positive rewards, negative penalizes); setting a weight to exactly `0.0`
makes `RewardManager.compute()` skip the term entirely
(`mjlab/managers/reward_manager.py`), and its `Episode_Reward/<term>` line
still prints as `0.0000` — so a zeroed term keeps its chart series.

---

## 7. Budget / chunking

Confirmed resume semantics ✅ EMPIRICAL: `OnPolicyRunner.learn(n)` loops
`range(start_it, start_it + n)` where `start_it = self.current_learning_iteration`,
and sets `self.current_learning_iteration = it` (the **last index**, not
`it + 1`). So the counter advances by `n - 1` per chunk after the first
(observed: `learn(2)` → 1, then `learn(2)` → 2). **Never** use
`current_learning_iteration` differences as the iteration budget.

> **CORRECTION 1 (S5).** An earlier draft of this section said repeated
> `learn()` calls "never repeat work". That is wrong, and the two halves of
> the sentence contradicted each other: because `start_it` is the last index
> *already executed*, each chunk after the first **re-executes that index**.
> Every `it` still runs a fresh rollout + PPO update, so no compute is
> wasted and the budget accounting is exact — but `model_{it}.pt` for the
> boundary index is overwritten and the printed iteration index repeats once
> per boundary. Visible in the acceptance run: a 20-iteration job
> (2 × `learn(10)`) ends at `current_learning_iteration == 18`, so
> `train_checkpoint_path` is `model_18.pt`, not `model_19/20.pt`. Deriving
> that path from `runner.current_learning_iteration` (as §3b already says)
> is what keeps this correct — do not compute it from the budget.

Keep our own counter:

```python
TIME_BUDGET_CHUNK_ITERS = 10          # same constant/rationale as web_train.py:59

start_time = time.time()
deadline = start_time + cli.max_minutes * 60 if cli.max_minutes is not None else None
iterations_done = 0
stopped_reason = "max_iterations" if cli.max_minutes is None else "max_minutes"

while True:
    if deadline is not None and time.time() >= deadline:
        stopped_reason = "max_minutes"; break
    if cli.max_iterations is not None and iterations_done >= cli.max_iterations:
        stopped_reason = "max_iterations"; break
    chunk = TIME_BUDGET_CHUNK_ITERS
    if cli.max_iterations is not None:
        chunk = min(chunk, cli.max_iterations - iterations_done)
    runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(iterations_done == 0))
    iterations_done += chunk
    write_progress()
```

`init_at_random_ep_len` only on the first chunk (re-randomizing episode
lengths mid-run would corrupt the tracking phase of in-flight episodes; the
Genesis path passes `True` every chunk, deliberately **not** copied).
`iterations_done` is what goes into both JSON files and into
`TrainingManager`'s history/estimate model.

Each `learn()` call ends with a `save()` → `model_{current_learning_iteration}.pt`
plus (from the tracking runner's own override) a discardable
`<log_dir>/<run_dir_name>.onnx`. Ignore the latter everywhere.

---

## 8. Open risks

1. **`rugiar_driver_mjlab.py` has no `drain_finished_training()`.** Its
   `control_tick()` never calls `training.poll()`/`finalize_policy()`
   (contrast `rugiar_driver.py:481-511`), so a job started from an mjlab
   session will run to completion and write its result file, but the new
   policy is not hot-loaded into the live supervisor until the session is
   restarted (`discover_local_policies()` then finds it). Not resolvable
   inside S5/S6 as scoped here — flagged for whichever step owns the driver.
2. **No GPU run was performed.** Everything above was validated on CPU
   (macOS, `CUDA_VISIBLE_DEVICES=""`). The `--device cuda:0` path is
   specified by mirroring mjlab's own `run_train()`, not by execution.
3. **Throughput unknown at real `num_envs`.** The probe ran `num_envs=2` at
   ~52 steps/s; nothing here calibrates `TrainingManager.estimate()` for
   mjlab jobs, and its history bucket does not distinguish mjlab from
   Genesis runs (`backend` in the history entries is `"local"` for both).
   ETA numbers shown in the panel for the first mjlab jobs will be wrong.
