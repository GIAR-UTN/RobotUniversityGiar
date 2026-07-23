# HANDOFF — Unified Control Web for LeggedGym-Ex

> **Read this first.** You are a fresh session with no memory. This document + the repo state is your full
> continuity. Everything in §1 is real, committed, and verified working unless explicitly marked otherwise.
> Repo: `github.com/josetabuyo/LeggedGym-Ex` (public fork, owner: josetabuyo).
> Local checkout: `/Users/josetabuyo/Development/GIAR/LeggedGym-Ex`.

---

## 1. Current state (verified)

### 1.1 Lineage & purpose

legged_gym (ETH Zürich) → unitree_rl_gym (Unitree; G1/Go2/H1 in Isaac Gym, MuJoCo sim2sim, real DDS sim2real)
→ LeggedGym-Ex (lupinjia; same framework ported to the **Genesis** physics engine, which runs on Apple Silicon
via Metal) → **this fork**. Everything here was trained and run on an M1 Pro Mac, CPU only, no NVIDIA GPU.
The `README.md` is written as course material (this is genuinely for a course/students); the original upstream
README is preserved as `UPSTREAM_README.md`.

### 1.2 Trained policies

| Name | What it is | Path |
|---|---|---|
| `stable` | unitree_rl_gym's shipped pretrained G1 checkpoint. Verified drop-in compatible with this fork's Genesis env (same URDF, joint order, PD gains). Dramatically more stable than our from-scratch run; treated as the reference policy. | `/Users/josetabuyo/Development/GIAR/unitree_rl_gym/deploy/pre_train/g1/motion.pt` (separate clone of the ORIGINAL unitree_rl_gym repo) |
| `cautious` | Fine-tuned by **resuming from the stable checkpoint's weights** under a reward that heavily penalizes torque/joint-velocity. Config: `G1CautiousCfg` in `legged_gym/envs/g1/g1_config.py`, task name `g1_cautious`. | `logs/g1_cautious/<run>/exported/policy_lstm_1.pt` |
| (g1 from-scratch) | 1800 PPO iterations from scratch on Genesis/CPU (task `g1`). Works but wobblier; superseded by `stable` as reference. | `logs/g1/...` |

### 1.3 Control architecture — `legged_gym/control/`

Backend-agnostic, sim/real-symmetric. All paths relative to repo root.

- **`legged_gym/control/adapter.py`** — `Lifecycle` enum (INACTIVE/READY/ACTIVE/FAULT; named to match ROS2
  `ros2_control` vocabulary on purpose), `RobotState` dataclass (dof_pos/vel, base_quat, base_ang_vel,
  base_lin_vel, projected_gravity, commands, action_scale, lifecycle), `RobotAdapter` Protocol
  (reset/get_state/send_action/record), `SimAdapter` (wraps a Genesis/MuJoCo legged_gym env — working, tested).
- **`legged_gym/control/policy.py`** — `ExplicitStatePolicy` (forward(obs,h,c)→(action,h,c), this fork's
  export convention) and `InternalStatePolicy` (forward(obs)→action, unitree_rl_gym's convention);
  `load_policy()` auto-detects. `ObsSpec` dataclass + enforcement (warns on mismatch). `damping_policy()` —
  zero-action emergency fallback skill.
- **`legged_gym/control/supervisor.py`** — `PolicySupervisor`. `request_switch(name)` only records intent;
  `confirm_pending_switch()` (called ONLY by SafetyGovernor) begins a linear cross-fade of actions over
  `ramp_ticks` (default 15) — never a hard cut.
- **`legged_gym/control/safety.py`** — `SafetyGovernor`, the ONLY component that decides "is this instant safe
  to switch," using `projected_gravity` (same upright/fallen signal `legged_robot.py` uses for RL episode
  termination). Forces hand-off to the `damping` fallback on fall / NaN / `estop()`, and **keeps forcing it
  every tick until `safety.reset()` is called explicitly** (a real bug found and fixed in review — estop was
  once just a one-time boolean).
- **`legged_gym/control/selector.py`** — `Selector` Protocol (`propose(state) -> Optional[name]`),
  `TiltRecoverySelector` rule-based impl. KNOWN LIMITATION: no hysteresis; re-proposes every tick; would fight
  a human's manual switch if wired in live.
- **`legged_gym/control/service.py`** — **`ControlService`, THE single call surface**:
  `request_switch(name) -> bool`, `status() -> dict`, `pause()`, `resume()`, `estop()`,
  `tick(obs) -> Optional[Tensor]`. Human (today: viser GUI buttons, in-process) and autonomous Selector call
  the exact same methods. The file's own docstring says it: wrap this same class with a tiny JSON-RPC-ish
  layer over WebSocket when a networked front-end is needed. **That is Stage A of this handoff.**
  README §6 states the same for the eventual LLM tool-calling layer (Stage D).

### 1.4 Real-robot adapter — `deploy_real/real_adapter.py`

`RealAdapter` implements the same `RobotAdapter` protocol against unitree_rl_gym's real `deploy_real.py`
conventions (DDS LowCmd/LowState via unitree_sdk2py, motor index mapping, IMU frame transform for
torso-mounted IMUs). **UNTESTED** — built with no physical G1 and no unitree_sdk2py installed. The physical
button-gated state machine (zero_torque_state → move_to_default_pos → default_pos_state) and the CRC/publish
step are `NotImplementedError` with exact porting instructions inline. Anything depending on real hardware
is gated on this (see Stage C).

### 1.5 Live demo — `legged_gym/scripts/swap_experiment.py`

Loads N policies via repeatable `--policy name:/path/to/file.pt`, builds
SimAdapter + PolicySupervisor + SafetyGovernor + ControlService, drives a **viser** web viewer
(nerfstudio-project's Python 3D web viz library — a websocket-based server, NOT a general static file server):

- Live markdown label with active policy (🟢 active / 🟡 ramping / 🔴 tripped)
- "Restart", "Pause"/"Resume" buttons; one `Switch to: <name>` button per policy under a "Policies" GUI folder
  — clicks call `service.request_switch(name)` in-process
- `--headless` scripted smoke test (no browser), used for CI/sanity
- `--docs_port N` adds a `[📖 Read the docs]` markdown link in the viser panel pointing at a **separate**
  `python -m http.server` process serving `docs/index.html` (viser cannot serve arbitrary files) — the most
  recent addition, in response to the user asking for an access point to the docs from the viser web

Current invocation:

```bash
python legged_gym/scripts/swap_experiment.py \
    --policy stable:/Users/josetabuyo/Development/GIAR/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
    --policy cautious:logs/g1_cautious/<run>/exported/policy_lstm_1.pt \
    --active stable --viser_port 9006 --docs_port 9007
```

### 1.6 Docs & README

- `docs/index.html` — long self-contained static didactic page (no build step): Unitree/RL-locomotion explainer,
  interactive PD-control canvas demo, collapsible real code excerpts, glossary. Fork-specific sections:
  "Running on Genesis instead" and "Switching policies live".
- `README.md` — course-style, explains the architecture rationale, explicitly why NOT full ROS2/ros2_control
  (cites `legubiao/quadruped_ros2_control` as prior art for the pattern borrowed without the dependency),
  honest limitations (§5), and **§6 "Roadmap: LLM interfacing"** which anticipates Stage D exactly.

### 1.7 Local conventions (mandatory)

- **Ports**: the user runs a "Local Agent Society" (`las`) CLI with a port registry. Before starting ANY
  server: `las ports audit`, then `las ports claim "<description>" --port <N>` (and `las ports free` when
  done). **9006 (viser) and 9007 (docs http.server) are currently claimed.** Claim a new port for any new
  server you add — do not squat.
- **Review checkpoints**: this project was built with a "Fable" model as independent architecture reviewer at
  key checkpoints (it caught the estop bug). The user values plan review before big implementation pushes —
  offer a Fable review of the Stage A/B design before writing lots of code.

---

## 2. The goal (user's intent, faithfully)

An **integrated control web** with:

1. A **Help/Docs view** (the existing `docs/index.html` content)
2. A **Simulator view** (viser running against the Genesis sim)
3. A persistent **Controls panel**: clickable policy list + **configurable keyboard shortcuts**
   (e.g. keys to switch policy; later arrow-key velocity commands like "forward")
4. Later, a **Real robot view** (G1 sensor status + camera) that replaces the simulator view but keeps the
   SAME controls in the same place and form, graying out / removing anything the real robot can't do
5. Later still, a **chat/LLM layer** translating natural language into ControlService calls

The controls must be **common to sim and real** — one UI, two backends — because `ControlService` +
`RobotAdapter` were designed exactly for that. The web layer is a new *caller* of the existing service, never
a new control path.

---

## 3. Staged roadmap

### Stage A — Networked transport wrapping `ControlService` ★ next session

The piece `service.py`'s docstring and README already call "the natural next piece". Everything downstream
(browser UI, real-robot remote control, LLM tools) needs it.

**What to build:** a small server that exposes ControlService's five methods + a status/telemetry stream over
the network, embedded in (or importable from) `swap_experiment.py`'s process — the service object lives where
the sim loop lives, so the transport must too.

**Protocol options:**

| Option | Pros | Cons |
|---|---|---|
| **WebSocket, JSON messages (recommended)** — commands as `{"method": "request_switch", "params": {"name": "cautious"}, "id": 1}` (JSON-RPC-ish), server pushes `status()` snapshots every N ticks on the same socket | One socket = commands + live telemetry push; trivial from browser JS; matches the docstring's own plan; symmetric for Stage C/D | Slightly more plumbing than plain HTTP |
| HTTP REST (FastAPI/Flask) | Dead simple, curl-able | Telemetry needs polling or a second SSE channel; two mechanisms where one would do |
| Full JSON-RPC 2.0 spec + library | Standard | Overkill for 5 methods; a hand-rolled dispatch dict is ~30 lines |

**Recommendation:** hand-rolled JSON-over-WebSocket using the `websockets` package (or FastAPI's WebSocket
support if you want HTTP static serving from the same process — see Stage B). Keep the message schema
JSON-RPC-*shaped* (method/params/id/result/error) without pulling in a spec library, so a future strict
JSON-RPC or MCP wrapper is a rename, not a rewrite.

**Threading reality (design this first):** `swap_experiment.py` runs a synchronous sim loop that calls
`service.tick(obs)`. A WebSocket server is async and runs on another thread/event loop. Do NOT call
ControlService methods directly from the socket thread. Pattern: socket handler puts commands on a
`queue.Queue`; the sim loop drains the queue once per tick and executes commands there (viser button
callbacks already effectively work this way — viser calls them from its own thread, and
`request_switch`/`pause`/`estop` are cheap flag-setting operations, which is why it's been safe so far —
but the networked layer should make the single-threaded-command-execution rule explicit, not accidental).
Telemetry direction: sim loop deposits latest `status()` dict; server task broadcasts it at ~10 Hz.

**Concrete first steps:**
1. New file `legged_gym/control/transport.py` *(new path — flagged as new)*: `ControlServer` class taking a
   `ControlService`, a command queue, and a port. Message dispatch for the 5 methods + a `status` push.
2. Wire into `swap_experiment.py` behind a new `--control_port` flag (claim the port via `las ports claim`).
   Queue-drain call inside the existing sim loop.
3. Headless test mirroring the existing `--headless` smoke test: connect a client, switch policies, estop,
   assert status transitions. Put it under `tests/` alongside whatever exists there.
4. Decide auth posture: none for localhost now, but log a warning if bound to non-localhost. estop must always
   be accepted even when paused/tripped (it already is at the service level — don't break that in dispatch).

**Scope: half a session**, including the test. The design questions above are the work; the code is small.

### Stage B — Unified single-page control web ★ next session (after A)

One page, one new lightweight server, three regions:

- **View area** (tabbed/switchable): **Docs** view · **Simulator** view · (later) **Real robot** view
- **Persistent Controls panel** (always visible regardless of active view): active-policy indicator
  (mirroring the 🟢/🟡/🔴 label), clickable policy list from `status()`, Pause/Resume, Restart, a visually
  distinct E-STOP, and the keyboard-shortcut system

**Server options:**

| Option | Pros | Cons |
|---|---|---|
| **Extend viser's GUI further** | No new server | viser's GUI is a control panel DSL, not a layout engine; cannot host tabs/iframes/custom keyboard UX. Dead end for this goal — reject. |
| **FastAPI in the swap_experiment process, serving static files + the Stage A WebSocket (recommended)** | One process, one new port claim; docs served from the same origin (kills the separate `--docs_port` http.server); WebSocket and page share origin — no CORS | Couples web serving to the sim process (fine at this scale, and Stage C wants the transport near the adapter anyway) |
| Separate static server + Stage A socket on its own port | Decoupled | Another port, CORS config, two processes to babysit for zero current benefit |

**Recommendation:** FastAPI (uvicorn on its own thread) inside swap_experiment: serves `web/index.html`
*(new dir at repo root — flagged as new)*, mounts `docs/` as static, exposes the Stage A WebSocket at `/ws`.
The frontend is plain HTML/JS/CSS, **no build step** — same philosophy as `docs/index.html`, and appropriate
for a course repo students clone and run.

**The three views:**
- **Docs**: `<iframe src="/docs/index.html">`. Same-origin once FastAPI mounts it — no problem. Retire the
  `--docs_port` side-server (keep the flag as a deprecated no-op or remove it; free port 9007 via
  `las ports free`).
- **Simulator**: `<iframe src="http://localhost:9006/">` (viser). **This is the top open question — verify
  it in the first 15 minutes of Stage B** (see §4, Q1). Fallback if iframing fights viser: a "pop out
  simulator" button opening viser in a new tab, with the unified page keeping only telemetry. Do NOT build a
  from-scratch 3D view — not worth it while viser works.
- **Real robot**: render the tab now as a disabled/"hardware required" placeholder driven by a
  `backend: "sim" | "real"` field added to `status()`. Builds the UI shape without waiting on hardware.

**Keyboard shortcuts:**
- A JSON keymap, default checked into `web/keymap.json` *(new — flagged)*, e.g.
  `{"1": {"action": "switch", "policy": "stable"}, "2": {"action": "switch", "policy": "cautious"}, "p": {"action": "pause_toggle"}, "Escape": {"action": "estop"}}`.
  Actions reference policies **by name**; names come from `status()` at runtime, so the keymap survives
  different `--policy` sets (unknown names → key shown as unbound, not an error).
- Captured via browser `keydown` on the unified page; handlers call the exact same WebSocket send as the
  mouse buttons — one code path.
- v1 configurability: an in-page "Shortcuts" panel that displays bindings and lets you rebind
  (press-key-to-assign), persisted to `localStorage` overriding the JSON defaults. No server-side settings
  store yet. Reserve arrow keys for future velocity commands (`commands` already exists in `RobotState`;
  a `set_command(vx, vy, yaw)` service method is a natural, small Stage B+ follow-up — flag it to the user
  rather than sneaking it in).
- Caveat: browser key events only arrive when the unified page (not the viser iframe) has focus. Give the
  controls panel a visible focus state and click-to-focus so this is legible rather than mysterious.

**Concrete first steps:**
1. Spike Q1 (viser-in-iframe) before any layout work.
2. `web/index.html` + `web/app.js` + `web/keymap.json`; FastAPI static mount + `/ws` in swap_experiment.
3. Controls panel driven entirely by `status()` pushes (policy list, active, tripped, paused, backend).
4. Shortcuts: load keymap → merge localStorage → keydown dispatch → same send path as clicks.
5. Manual end-to-end check with the real sim running, then extend the headless smoke test to hit the HTTP
   endpoints.

**Scope: one full session** if the viser iframe cooperates; the tab-fallback costs little extra. Panel +
shortcuts is the bulk of the work.

### Stage C — Real robot view (GATED on hardware)

Same page, same controls panel, same transport; only the view area and the backend change.

- **Gate:** requires finishing/testing `deploy_real/real_adapter.py` (the `NotImplementedError` sections:
  button-gated startup state machine, CRC/publish), which requires a physical G1 + unitree_sdk2py. Do not
  attempt to "finish" RealAdapter speculatively without hardware — it's honestly documented as untested for
  a reason.
- **Buildable now without hardware:** the `backend` field in `status()`; per-control `available: bool`
  capability flags in status (adapter-declared, not UI-hardcoded); the graying-out logic in the panel; a
  telemetry widget layout fed by the same `RobotState` fields the sim already produces (dof_pos/vel, IMU-ish
  quantities, projected_gravity as a tilt gauge). A `--fake_real` stub adapter could exercise all of it.
- **Hardware-dependent:** camera feed (likely MJPEG or WebRTC from the G1 — research when hardware exists),
  real sensor rates, DDS wiring, and the on-robot deployment story for the transport server.
- **Scope:** multi-session, and not schedulable until hardware access exists.

### Stage D — LLM/chat layer (explicitly later; out of scope next session)

README §6 already lays this out: `ControlService`'s small, named, well-typed method surface behind a safety
layer *is* the LLM tool-calling interface. Stage D is "wire an LLM to call methods that already exist" —
a chat pane in the unified web whose backend translates natural language into the same Stage A messages
(`"be more careful"` → `request_switch("cautious")`), and/or an LLM-backed `Selector`. New architecture
required: none. Do not start this before A–C are stable; do keep Stage A's message schema clean so an MCP/tool
wrapper stays trivial.

---

## 4. Open questions — resolve these EARLY next session

1. **Can viser be iframed cleanly?** Its client is a websocket-driven React app; it may set
  `X-Frame-Options`/CSP, or its own keyboard/pointer handling may fight the host page. **Spike first**: serve
  a 5-line page iframing `http://localhost:9006/` and check rendering, camera controls, and whether the host
  page still receives keydown when the iframe has focus. Outcome decides iframe vs pop-out-tab fallback.
2. **Same process or separate?** Recommendation is FastAPI inside swap_experiment (§3-B). Confirm with the
  user before committing — it shapes ports, deployment, and Stage C.
3. **Keymap depth for v1:** checked-in JSON + localStorage rebinding (recommended) vs JSON-only vs full
  settings UI. Ask; don't gold-plate.
4. **How much Stage C shape now?** Recommendation: `backend` + capability flags + disabled tab placeholder
  only. Confirm the user doesn't want the `--fake_real` stub already in Stage B.
5. **`set_command` velocity control:** arrow-key "forward" implies a new service method + safety thinking
  (command limits). Real but separate scope — get explicit user sign-off before adding it.
6. **Ports:** run `las ports audit`; claim one new port for the unified web server (or reuse 9007 if the docs
  side-server is retired — still re-claim it under the new description).

---

## 5. What NOT to do (lessons already paid for)

- **Never bypass `SafetyGovernor`**, even "just in the UI layer". Only it confirms switches; only
  `safety.reset()` clears a trip. The estop-was-just-a-boolean bug was real, found in review, fixed and
  tested — don't reintroduce its cousins in the transport (e.g. a "force switch" debug message).
- **The web layer never touches `RobotAdapter` or `PolicySupervisor` directly.** Everything goes through
  `ControlService`. That symmetry is the entire reason Stage C and D are cheap.
- **Don't call ControlService from the socket thread.** Queue commands; execute in the sim loop (§3-A).
- **Respect known limitations:** `ObsSpec` mismatches only warn — keep obs dims consistent when adding
  policies; `TiltRecoverySelector` has no hysteresis and will fight manual switches — do NOT wire it live
  into the web demo without adding hysteresis/priority first.
- **Ports go through `las`** (`las ports audit` / `claim` / `free`). 9006 and 9007 are claimed; never start a
  server on an unclaimed port.
- **Don't "finish" `RealAdapter` without hardware.** Its NotImplementedError sections carry exact porting
  instructions for when a G1 + unitree_sdk2py exist; speculative completion would just be untested code
  wearing a tested costume.
- **No build-step frontend** (React/Vite/etc.). Plain HTML/JS/CSS, like `docs/index.html` — this is course
  material students must be able to read and run.
- **Don't break `--headless`** — it's the CI/sanity path. Extend it to cover the transport instead.
- **Emoji discipline:** the 🟢/🟡/🔴 status convention exists in the viser label; mirror it in the web panel
  rather than inventing a new one.

## 6. Definition of done for next session

- Stage A: `legged_gym/control/transport.py` + `--control_port` in swap_experiment + passing headless
  transport test. A `websocat`/Python one-liner can switch policies and estop from outside the process.
- Stage B: `web/` page served by the same process; Docs tab works; Simulator tab (iframe or documented
  pop-out fallback per Q1); controls panel fully mirrors viser buttons; keyboard switching works with the
  default keymap; Real-robot tab present but disabled with `backend`-driven reasoning.
- Open questions Q1–Q6 have recorded answers (a short note appended to this file is fine).
- README gets a short new subsection describing the web front-end once it exists (course-material tone).
- Committed and pushed; ports claimed/freed via `las`.
