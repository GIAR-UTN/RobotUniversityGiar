// Unified control web front-end — plain JS, no build step (same philosophy
// as docs/index.html; this is course material students read and run).
//
// Everything here goes through the ONE call surface, ControlService, via
// the JSON-over-WebSocket transport in legged_gym/control/transport.py.
// This file never talks to RobotAdapter/PolicySupervisor directly — see
// HANDOFF_control_web.md §5 for why that boundary matters.

let ws = null;
let latestStatus = null;
let lastStatusJSON = null; // dedupe key — status() pushes at ~10Hz whether or not anything changed
let msgId = 1;
let keymap = {};
let keyByPolicy = {}; // policy name -> shortcut key, recomputed on every render from LIST POSITION
let policyByKey = {}; // inverse of keyByPolicy, for keydown dispatch — see recomputePolicyShortcuts()
let renderedPolicyNames = null; // policies list actually painted into the DOM right now
let renderedTrainingJobsKey = null; // like renderedPolicyNames — dedupe key excluding elapsed_s
                                     // (elapsed_s would otherwise change on every ~10Hz status
                                     // push while a job is running and force a DOM rebuild every tick)
let trainingCatalog = null; // {tasks, base_policies} — fetched once per connection via training_catalog
let keysArmed = true; // true whenever no drawer is open over the simulator —
                       // keyboard shortcuts must not fire while reading Docs
                       // (e.g. arrow-key scrolling shouldn't drive the
                       // robot); see openDrawer()/closeDrawer() below

// ---- Simulator is a permanent base layer ----
// Two earlier approaches both failed here: (1) tab-switching the Simulator
// out of view left its viser <iframe> alive and rendering forever in the
// background — the CPU/GPU contention from that is what originally made the
// page unresponsive; (2) destroying/recreating that iframe on tab switch (to
// fix #1) backfired worse — tearing down a live WebGL context synchronously
// froze the renderer hard (~180% CPU, unresponsive 40+ seconds) under real
// host load. DO NOT hide, resize, or remove the Simulator's iframe on
// navigation. It is mounted once at boot and never touched again. Docs (and,
// later, Real-robot) are drawers that slide in OVER it instead — see
// index.html's .drawer CSS — so the simulator view itself never changes.
let viserPort = null; // filled in from /config at boot

function mountSimIframe() {
  if (!viserPort) return;
  const iframe = document.createElement('iframe');
  iframe.src = `http://localhost:${viserPort}/`;
  $('#view-sim').appendChild(iframe);
}

const $ = (sel) => document.querySelector(sel);
const panel = $('#panel');
const footer = $('#footer');
const connDot = $('#conn-dot');
const activeLabel = $('#active-label');
const policyList = $('#policy-list');

const chkPush = $('#chk-push');
const chkAutoCmd = $('#chk-auto-cmd');
const simPushDir = $('#sim-push-dir');
const hudVx = $('.hud-vx');
const hudVy = $('.hud-vy');
const hudYaw = $('.hud-yaw');
let draggingCommand = false; // true while the user has a command HUD grabbed —
                              // suppresses syncing FROM status() pushes so the
                              // drag doesn't fight the server echo

let commandRanges = null; // {vx:[lo,hi], vy:[lo,hi], yaw:[lo,hi]} from /config — the
                           // exact envelope this policy was trained across (see
                           // SimAdapter.set_command); ramping/mouse-look never drive
                           // past these bounds

// Sign→direction conventions (body frame, ROS REP-103: x fwd, y left, z up):
//   +vx = forward (w),  +vy = strafe LEFT (a),  +yaw = turn LEFT/CCW (ArrowLeft)
// HUD fills: vx up=+, vy left=+, yaw CCW=+. Verified against keymap.json signs
// and app.js mouse-look comment ("yaw+ means turn left").

// ---- WASD + mouse-look "cruise" movement ----
// GTA-style analog feel from digital keyboard input: holding a movement key
// accelerates that axis toward whichever end of its trained range the key
// points at; releasing it decelerates smoothly back to zero, like letting
// go of the stick/gas rather than holding a set speed. Both use frame-rate-
// independent exponential decay so the feel is identical regardless of
// refresh rate. Only active in manual mode — while "Random Movement" is on,
// the server drives these axes itself and the HUDs are a read-only display
// (see the `auto` check in frame() and applyStatus()).
const ACCEL_RATE = 6.0; // 1/s — approach rate toward a held target (higher = snappier build-up)
const DECEL_RATE = 4.0; // 1/s — coast-down rate toward zero after release
let lastFrameT = null;
let lastSendT = 0;
let cmdDirty = false;
const heldMoveKeys = new Set(); // keys currently held down, not tapped
let cruiseVx = 0, cruiseVy = 0, cruiseYaw = 0; // the single source of truth for
                                                // the manual command — HUDs,
                                                // WASD, and mouse-look all read
                                                // and write these same three
let mouseLookActive = false;

function send(method, params = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ method, params, id: msgId++ }));
}

// send() is fire-and-forget (its reply is ignored — the next status push is
// the source of truth, per the comment at the bottom of onmessage). The
// Create Policy form needs an actual reply (the catalog to render the form,
// or a job id / error from starting training), so it gets its own
// promise-based call() that tracks replies by id.
let pendingCalls = {}; // id -> {resolve, reject}

function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) { reject(new Error('not connected')); return; }
    const id = msgId++;
    pendingCalls[id] = { resolve, reject };
    ws.send(JSON.stringify({ method, params, id }));
  });
}

function connect() {
  const url = `ws://${location.host}/ws`;
  ws = new WebSocket(url);
  ws.onopen = () => {
    footer.textContent = 'connected'; connDot.className = 'ok';
    refreshTrainingCatalog();
    refreshSystemInfo();
  };
  ws.onclose = () => {
    footer.textContent = 'disconnected — retrying…';
    connDot.className = 'bad';
    setTimeout(connect, 1000);
  };
  ws.onerror = () => { connDot.className = 'bad'; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id != null && pendingCalls[msg.id]) {
      const { resolve, reject } = pendingCalls[msg.id];
      delete pendingCalls[msg.id];
      if ('error' in msg) reject(new Error(msg.error)); else resolve(msg.result);
      return;
    }
    if (msg.method === 'status') {
      // status() pushes at ~10Hz REGARDLESS of whether anything changed —
      // most ticks are identical to the last one. Skip all DOM work on a
      // dupe: this loop is what keeps the page usable even when the host
      // is under heavy CPU/memory pressure (see HANDOFF_control_web.md).
      const json = ev.data;
      if (json !== lastStatusJSON) {
        lastStatusJSON = json;
        applyStatus(msg.result);
      }
    } else if ('error' in msg) {
      console.error('ControlService error:', msg.error);
    }
    // Command replies (id-keyed) aren't otherwise handled — the next
    // status push (≤100ms later) is the source of truth for UI state.
  };
}

function policyButtonRow(name, active) {
  const row = document.createElement('div');
  row.className = 'policy-row';
  row.dataset.policy = name;

  const handle = document.createElement('span');
  handle.className = 'drag-handle';
  handle.setAttribute('aria-label', 'Reorder');
  handle.title = 'Drag to reorder — shortcut keys follow position (top = 0)';
  handle.textContent = '⋮⋮';
  row.appendChild(handle);

  const btn = document.createElement('button');
  btn.className = 'policy-btn' + (name === active ? ' active' : '');
  const label = document.createElement('span');
  label.textContent = name;
  btn.appendChild(label);
  const key = keyByPolicy[name];
  if (key) {
    const cap = document.createElement('span');
    cap.className = 'keycap';
    cap.textContent = key;
    btn.appendChild(cap);
  }
  btn.onclick = () => send('request_switch', { name });
  row.appendChild(btn);

  // Opens the rich info popup (§ policy info) — everything this session
  // knows about how `name` came to exist: the exact command, what it was
  // cloned from, entropy_coef/reward weight overrides, file paths, and (for
  // anything trained through this UI) the parsed noise-std/reward/episode-
  // length trend. This is the pivot the "Done" training-job cards used to
  // half-cover — see renderTrainingJobs()'s docstring for why those cards
  // no longer stick around once a policy exists.
  const info = document.createElement('button');
  info.className = 'policy-info-btn';
  info.type = 'button';
  info.textContent = '⚙';
  info.title = `Config — how "${name}" was trained, rename it`;
  info.onclick = (e) => { e.stopPropagation(); openPolicyInfo(name); };
  row.appendChild(info);

  // No keyboard shortcut, no auto-confirm — discarding a policy deletes its
  // checkpoint file, and this UI already deliberately dropped its one
  // irreversible keybinding once before (the E-STOP key). Disabled for the
  // active policy client-side too (not just server-side, which also
  // enforces it) so the reason it's unavailable is visible, not a
  // surprise error after clicking.
  const del = document.createElement('button');
  del.className = 'policy-delete-btn';
  del.type = 'button';
  del.textContent = '✕';
  del.title = name === active
    ? "Can't delete the active policy — switch to another one first"
    : `Delete "${name}" — removes it from this list and deletes its checkpoint file. Cannot be undone.`;
  del.disabled = name === active;
  del.onclick = (e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete policy "${name}"? This removes it from the list and deletes its checkpoint file on disk — cannot be undone.`)) return;
    del.disabled = true;
    call('delete_policy', { name }).then(() => {
      // The Policies list itself refreshes from the next status push
      // (status.policies already won't include `name`); the Clone-from
      // catalog is a separate fetch that isn't part of that push, so it
      // needs its own refresh or it'd keep offering a deleted checkpoint.
      refreshTrainingCatalog();
    }).catch((err) => {
      del.disabled = false;
      window.alert(`Couldn't delete "${name}": ${err.message}`);
    });
  };
  row.appendChild(del);

  return row;
}

// ---- policy order + shortcut assignment ----
// Shortcuts are NOT tied to a policy's name (keymap.json used to do that) —
// they're tied to POSITION in this drag-reorderable list, so reordering is
// the rebind UI: top slot is always '0', then '9' down to '1' for the rest.
// Capped at 10 policies for now (one digit key per slot); anything beyond
// that just has no shortcut.
const POLICY_ORDER_KEY = 'giar.policyOrder.v1';
const POLICY_SHORTCUT_KEYS = ['0', '9', '8', '7', '6', '5', '4', '3', '2', '1'];

function loadPolicyOrder() {
  const raw = localStorage.getItem(POLICY_ORDER_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function savePolicyOrder(names) {
  localStorage.setItem(POLICY_ORDER_KEY, JSON.stringify(names));
}

// Renaming a policy must not silently bump it to the bottom of the drag
// order (and therefore lose its shortcut key, which is position-based —
// see the comment above) just because its old name no longer matches
// anything in the saved list. Swap the name in place instead of leaving
// the stale entry for applyPolicyOrder() to filter out and re-append.
function renamePolicyOrder(oldName, newName) {
  const saved = loadPolicyOrder();
  if (!saved) return;
  const idx = saved.indexOf(oldName);
  if (idx === -1) return;
  saved[idx] = newName;
  savePolicyOrder(saved);
}

// Applies the saved drag order on top of the backend's raw name list —
// names the user has never seen before (a fresh --policy) land at the end
// in backend order; names the user deleted just drop out here for free.
function applyPolicyOrder(rawNames) {
  const saved = loadPolicyOrder();
  if (!saved) return rawNames;
  const rawSet = new Set(rawNames);
  const ordered = saved.filter((n) => rawSet.has(n));
  const orderedSet = new Set(ordered);
  for (const name of rawNames) {
    if (!orderedSet.has(name)) ordered.push(name);
  }
  return ordered;
}

function recomputePolicyShortcuts(names) {
  keyByPolicy = {};
  policyByKey = {};
  names.forEach((name, i) => {
    const key = POLICY_SHORTCUT_KEYS[i];
    if (!key) return;
    keyByPolicy[name] = key;
    policyByKey[key] = name;
  });
}

function renderPolicyList(names, active) {
  recomputePolicyShortcuts(names);
  policyList.innerHTML = '';
  for (const name of names) {
    policyList.appendChild(policyButtonRow(name, active));
  }
  initPolicyDrag();
}

function onPolicyReorder() {
  const names = [...policyList.querySelectorAll('.policy-row')].map((r) => r.dataset.policy);
  savePolicyOrder(names);
  // Re-render (rather than patch in place) so the keycaps painted on each
  // row immediately reflect the new position -> key assignment.
  renderPolicyList(names, latestStatus?.active);
}

function initPolicyDrag() {
  makeSortable(policyList, '.policy-row', ':scope > .drag-handle', onPolicyReorder);
}

function applyStatus(status) {
  // Captured before latestStatus is overwritten below — used just after to
  // detect an active-policy change from ANY source (row click, keyboard
  // shortcut, another connected client) and follow it in the info dock.
  const previousActive = latestStatus?.active;
  latestStatus = status;

  let color = status.ramping ? '🟡' : '🟢';
  if (status.safety_tripped) color = '🔴';
  let text = `${color} ${status.active}`;
  if (status.pending) text += ` → ${status.pending}`;
  if (status.paused) text += ' (paused)';
  activeLabel.innerHTML = `<span id="conn-dot" class="${connDot.className}"></span>${text}`;

  // The policy list itself rarely changes (only when --policy set or the
  // active/pending name changes) — rebuilding it from scratch every push
  // is wasted DOM churn, so only touch it when the visible set changed.
  // Display order (and therefore shortcut assignment) is the user's saved
  // drag order, not the raw backend order — see applyPolicyOrder().
  const rawNames = status.policies || [];
  const namesKey = rawNames.join(' ') + ' ' + status.active;
  if (namesKey !== renderedPolicyNames) {
    renderedPolicyNames = namesKey;
    renderPolicyList(applyPolicyOrder(rawNames), status.active);
  }

  // Keep the info dock following the active policy: if it's open and was
  // showing whatever used to be active, and the switch happened (row
  // click, keyboard shortcut, another client — anything, not just the
  // dock's own ↑/↓ nav, which already does this itself via stepPolicyInfo),
  // reload it onto the new active policy. Doesn't fire if the dock is
  // pinned to inspect a DIFFERENT, non-active policy opened via its ⓘ
  // button — that stays put, matching the existing "inspecting != running"
  // behavior documented on stepPolicyInfo().
  if (policyInfoOpen && status.active !== previousActive && policyInfoCurrentName === previousActive) {
    openPolicyInfo(status.active);
  }

  $('#btn-pause').firstChild.textContent = status.paused ? 'Resume ' : 'Pause ';

  const restartAvailable = status.capabilities?.restart !== false;
  const restartBtn = $('#btn-restart');
  restartBtn.disabled = !restartAvailable;
  restartBtn.title = restartAvailable ? '' : `not available on backend "${status.backend}"`;

  const realTab = document.querySelector('nav button[data-drawer="real"]');
  const realPlaceholder = $('#real-placeholder');
  if (status.backend === 'real') {
    realTab.disabled = false;
    realPlaceholder.textContent = '';
  } else {
    realTab.disabled = true;
    realTab.title = `backend is "${status.backend}" — connect a real robot to enable this view`;
    realPlaceholder.textContent = `Real-robot view unavailable: current backend is "${status.backend}".`;
  }

  let auto = false;
  if (status.random_events) {
    chkPush.checked = status.random_events.push_robots;
    if (document.activeElement !== simPushDir) simPushDir.value = status.random_events.push_dir || '';
    auto = status.random_events.auto_commands;
    chkAutoCmd.checked = auto;
    // While auto, the HUDs are a read-only display of what the sim is doing
    // on its own; while manual, they're live input controls. Either way,
    // don't stomp on a HUD the user currently has grabbed.
    [hudVx, hudVy, hudYaw].forEach((el) => { el.classList.toggle('disabled', auto); });
  }
  // Only sync FROM the server when nothing is actively driving the command
  // right now — otherwise this would fight a held key, an in-progress drag,
  // the mouse-look pad, or (in manual mode) the local decel-to-zero coast
  // with a ~100ms-stale echo of what we just sent. This sync is
  // intentionally immediate (no damping) — direct drag and server echo
  // must both feel 1:1, with no added lag; only accel/decel get smoothing.
  const activelyDriving = draggingCommand || heldMoveKeys.size > 0 || mouseLookActive || (!auto && !isSettled());
  if (status.command && !activelyDriving) {
    cruiseVx = status.command.vx;
    cruiseVy = status.command.vy;
    cruiseYaw = status.command.yaw;
    updateCommandUI();
  }

  renderTrainingJobs(status.training_jobs || []);
  renderTelemetry(status.telemetry);
}

// ---- live telemetry panel ----
// See ControlService._telemetry()'s own docstring: each entry already
// carries its own source/label/note/unit, so this is pure rendering, no
// per-variable knowledge duplicated here — a new telemetry field just
// shows up automatically.
const telemetryBody = $('#telemetry-body');

// Fixed decimal places (toFixed) already keeps digit COUNT constant; the
// remaining source of horizontal jitter is the sign — "-0.033" is one
// character wider than "0.033" — so a value crossing zero was shifting
// everything after it in the row. Reserving a sign column (a leading
// space for non-negative numbers) keeps every number the same width
// regardless of sign.
function padNum(v) {
  const s = v.toFixed(3);
  return v < 0 ? s : ` ${s}`;
}

function formatTeleValue(value, unit) {
  if (value == null) return '–';
  if (Array.isArray(value)) return `[${value.map(padNum).join(', ')}] ${unit}`;
  return `${padNum(value)} ${unit}`;
}

function renderTelemetry(telemetry) {
  if (!telemetry) {
    telemetryBody.innerHTML = '<p class="field-hint">Waiting for data&hellip;</p>';
    return;
  }
  // Update existing rows/cells in place rather than rebuilding the DOM —
  // the label/badge/note text never changes tick to tick, only the value,
  // so only the value node's textContent needs to touch the layout at all.
  // (One-time: clear the "Waiting for data…" placeholder on first arrival.)
  if (!telemetryBody.dataset.ready) {
    telemetryBody.innerHTML = '';
    telemetryBody.dataset.ready = '1';
  }
  for (const [key, t] of Object.entries(telemetry)) {
    let row = telemetryBody.querySelector(`.tele-row[data-key="${key}"]`);
    if (!row) {
      row = document.createElement('div');
      row.className = 'tele-row';
      row.dataset.key = key;
      const labelRow = document.createElement('div');
      labelRow.className = 'tele-label-row';
      const label = document.createElement('span');
      label.className = 'tele-label';
      label.textContent = t.label;
      const badge = document.createElement('span');
      badge.className = `tele-badge ${t.source}`;
      badge.textContent = t.source === 'sensor' ? 'sensor' : 'sim';
      labelRow.append(label, badge);
      const val = document.createElement('span');
      val.className = 'tele-val';
      row.appendChild(labelRow);
      row.appendChild(val);
      if (t.note) {
        const note = document.createElement('p');
        note.className = 'tele-note';
        note.textContent = t.note;
        row.appendChild(note);
      }
      telemetryBody.appendChild(row);
    }
    row.querySelector('.tele-val').textContent = formatTeleValue(t.value, t.unit);
  }
}

// ---- HUD rendering ----

function axisScale(range) {
  if (!commandRanges || !range) return 1;
  return Math.max(Math.abs(range[0]), Math.abs(range[1])) || 1;
}

function setHudVx(value) {
  const t = Math.max(-1, Math.min(1, value / axisScale(commandRanges?.vx)));
  const fill = hudVx.querySelector('.hud-fill-v');
  if (t >= 0) {
    fill.style.bottom = '50%';
    fill.style.top = (50 - t * 50) + '%';
    fill.style.background = 'var(--accent)';
  } else {
    fill.style.top = '50%';
    fill.style.bottom = (50 - Math.abs(t) * 50) + '%';
    fill.style.background = 'var(--accent2)';
  }
}

function setHudVy(value) {
  const t = Math.max(-1, Math.min(1, value / axisScale(commandRanges?.vy)));
  const fill = hudVy.querySelector('.hud-fill-h');
  // +vy = LEFT (see sign-convention comment above) — positive fills leftward.
  if (t >= 0) {
    fill.style.right = '50%';
    fill.style.left = (50 - t * 50) + '%';
    fill.style.background = 'var(--accent)';
  } else {
    fill.style.left = '50%';
    fill.style.right = (50 - Math.abs(t) * 50) + '%';
    fill.style.background = 'var(--accent2)';
  }
}

function polarToCartesian(cx, cy, r, angleDeg) {
  const a = (angleDeg - 90) * Math.PI / 180; // -90 so 0deg = straight up (12 o'clock)
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function describeArc(cx, cy, r, startDeg, endDeg) {
  const start = polarToCartesian(cx, cy, r, endDeg);
  const end = polarToCartesian(cx, cy, r, startDeg);
  const largeArc = Math.abs(endDeg - startDeg) <= 180 ? 0 : 1;
  const sweep = endDeg > startDeg ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} ${sweep} ${end.x} ${end.y}`;
}

function setHudYaw(value) {
  const t = Math.max(-1, Math.min(1, value / axisScale(commandRanges?.yaw)));
  // +yaw = turn LEFT/CCW; SVG rotate() is clockwise-positive, so negate.
  const deg = -t * 135;
  const needle = hudYaw.querySelector('.dial-needle');
  const arc = hudYaw.querySelector('.dial-arc');
  needle.setAttribute('transform', `rotate(${deg} 50 50)`);
  arc.setAttribute('d', describeArc(50, 50, 42, 0, deg));
  arc.style.stroke = t > 0 ? 'var(--accent)' : 'var(--accent2)';
}

function updateCommandUI() {
  setHudVx(cruiseVx);
  setHudVy(cruiseVy);
  setHudYaw(cruiseYaw);
  $('#val-vx').textContent = cruiseVx.toFixed(2);
  $('#val-vy').textContent = cruiseVy.toFixed(2);
  $('#val-yaw').textContent = cruiseYaw.toFixed(2);
}

// ---- tabs ----

// A drawer (Docs, later Real-robot) slides in OVER the permanently-mounted
// Simulator — see the CSS-level comment in index.html. Only one open at a
// time; clicking the already-open drawer's button retracts it instead of
// re-opening it (the toggle behavior asked for). Keyboard shortcuts are
// armed only while no drawer is open, so reading Docs can't accidentally
// drive the robot.
let openDrawerName = null;
let policyInfoOpen = false; // true while the policy info popup is up — see openPolicyInfo()

function setKeysArmed() {
  // Opening the policy detail dock must NOT disarm robot-control keys —
  // it's a side panel, not something covering the sim view the way a
  // drawer does (see openDrawerName's comment above). It used to also gate
  // on !policyInfoOpen because the dock's own keydown listener hijacked
  // ArrowUp/ArrowDown for policy nav; that hijack is gone (nav is
  // button/click-only now — see the prev/next wiring below), so there's no
  // longer any reason for the dock to touch robot-control arming at all.
  keysArmed = openDrawerName === null;
}

function closeDrawer() {
  if (!openDrawerName) return;
  document.getElementById(`drawer-${openDrawerName}`).classList.remove('open');
  document.querySelector(`#tabs button[data-drawer="${openDrawerName}"]`).classList.remove('active');
  openDrawerName = null;
  setKeysArmed();
}

function openDrawer(name) {
  if (openDrawerName === name) { closeDrawer(); return; }
  closeDrawer();
  document.getElementById(`drawer-${name}`).classList.add('open');
  document.querySelector(`#tabs button[data-drawer="${name}"]`).classList.add('active');
  openDrawerName = name;
  setKeysArmed();
}

document.querySelectorAll('#tabs button[data-drawer]').forEach((btn) => {
  btn.addEventListener('click', () => { if (!btn.disabled) openDrawer(btn.dataset.drawer); });
});

// ---- controls panel buttons ----

$('#btn-pause').addEventListener('click', () => {
  send(latestStatus?.paused ? 'resume' : 'pause');
});
$('#btn-restart').addEventListener('click', () => send('restart'));

// ---- stimuli: random events + manual velocity command ----

function clampToRange(v, range) {
  if (!range) return v;
  return Math.max(range[0], Math.min(range[1], v));
}

function sendCruiseCommand() {
  updateCommandUI();
  send('set_command', { vx: cruiseVx, vy: cruiseVy, yaw: cruiseYaw });
}

function engageManualIfNeeded() {
  // Taking the stick/keys/mouse takes manual control of the *velocity
  // command* specifically — it does NOT touch the "Random pushes" checkbox,
  // so a training-style shove can still knock you around while you drive.
  if (latestStatus?.random_events?.auto_commands !== false) {
    send('set_random_events', { push_robots: chkPush.checked, auto_commands: false, push_dir: simPushDir.value || null });
  }
}

function heldDirection(axis) {
  let unit = 0;
  for (const key of heldMoveKeys) {
    const b = keymap[key];
    if (b?.action === 'move' && b.axis === axis) unit += b.sign;
  }
  return Math.max(-1, Math.min(1, unit));
}

function smoothAxis(current, dir, range, dt, decelRate) {
  if (dir === 0 || !range) {
    // Released — coast smoothly down to a stop, GTA-style, instead of
    // freezing at whatever value was reached.
    if (Math.abs(current) < 1e-3) return 0;
    return current * Math.exp(-decelRate * dt);
  }
  const target = dir > 0 ? range[1] : range[0];
  const next = target + (current - target) * Math.exp(-ACCEL_RATE * dt);
  const snapped = Math.abs(next - target) < 1e-3 ? target : next;
  return dir > 0 ? Math.min(snapped, range[1]) : Math.max(snapped, range[0]);
}

function isSettled() {
  return Math.abs(cruiseVx) < 1e-3 && Math.abs(cruiseVy) < 1e-3 && Math.abs(cruiseYaw) < 1e-3;
}

function frame(now) {
  if (lastFrameT === null) lastFrameT = now;
  const dt = Math.min(0.05, (now - lastFrameT) / 1000);
  lastFrameT = now;

  // While "Random Movement" is on, the server drives these axes itself and
  // the HUDs are a read-only display (see applyStatus) — don't fight it
  // with local accel/decel.
  if (latestStatus?.random_events?.auto_commands !== true) {
    const dvx = heldDirection('vx');
    const dvy = heldDirection('vy');
    const dyaw = heldDirection('yaw');
    const nvx = smoothAxis(cruiseVx, dvx, commandRanges?.vx, dt, DECEL_RATE);
    const nvy = smoothAxis(cruiseVy, dvy, commandRanges?.vy, dt, DECEL_RATE);
    const nyaw = smoothAxis(cruiseYaw, dyaw, commandRanges?.yaw, dt, DECEL_RATE);
    if (nvx !== cruiseVx || nvy !== cruiseVy || nyaw !== cruiseYaw) {
      cruiseVx = nvx; cruiseVy = nvy; cruiseYaw = nyaw; cmdDirty = true;
    }
  }
  // Visuals stay fluid every frame; actual WS sends stay throttled to ~10Hz
  // (matching the previous fixed-interval cadence) so we don't flood the
  // socket at 60fps.
  if (cmdDirty && now - lastSendT >= 100) {
    sendCruiseCommand();
    cmdDirty = false;
    lastSendT = now;
  } else if (cmdDirty) {
    updateCommandUI();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

chkPush.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked, push_dir: simPushDir.value || null });
});
simPushDir.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked, push_dir: simPushDir.value || null });
});
chkAutoCmd.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked, push_dir: simPushDir.value || null });
  // Going manual right now should take the HUDs' CURRENT position as the
  // first command, rather than waiting for the user to nudge one first.
  if (!chkAutoCmd.checked) sendCruiseCommand();
});

// ---- directional HUD indicators: drag-to-set ----

function bindVerticalHud(hud, axis) {
  const track = hud.querySelector('.hud-track-v');
  let pointerId = null;
  const setFromEvent = (e) => {
    const rect = track.getBoundingClientRect();
    const centerY = rect.top + rect.height / 2;
    const t = Math.max(-1, Math.min(1, (centerY - e.clientY) / (rect.height / 2)));
    const scale = axisScale(commandRanges?.[axis]);
    const value = clampToRange(t * scale, commandRanges?.[axis]);
    if (axis === 'vx') cruiseVx = value; else if (axis === 'vy') cruiseVy = value; else cruiseYaw = value;
    engageManualIfNeeded();
    sendCruiseCommand();
  };
  track.addEventListener('pointerdown', (e) => {
    if (hud.classList.contains('disabled')) return;
    draggingCommand = true; pointerId = e.pointerId;
    track.setPointerCapture(e.pointerId);
    setFromEvent(e);
  });
  track.addEventListener('pointermove', (e) => { if (draggingCommand && e.pointerId === pointerId) setFromEvent(e); });
  track.addEventListener('pointerup', (e) => { if (e.pointerId === pointerId) { draggingCommand = false; pointerId = null; } });
}

function bindHorizontalHud(hud, axis) {
  const track = hud.querySelector('.hud-track-h');
  let pointerId = null;
  const setFromEvent = (e) => {
    const rect = track.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    // Pointer left of center => positive t => LEFT (matches +vy = LEFT).
    const t = Math.max(-1, Math.min(1, (centerX - e.clientX) / (rect.width / 2)));
    const scale = axisScale(commandRanges?.[axis]);
    const value = clampToRange(t * scale, commandRanges?.[axis]);
    if (axis === 'vx') cruiseVx = value; else if (axis === 'vy') cruiseVy = value; else cruiseYaw = value;
    engageManualIfNeeded();
    sendCruiseCommand();
  };
  track.addEventListener('pointerdown', (e) => {
    if (hud.classList.contains('disabled')) return;
    draggingCommand = true; pointerId = e.pointerId;
    track.setPointerCapture(e.pointerId);
    setFromEvent(e);
  });
  track.addEventListener('pointermove', (e) => { if (draggingCommand && e.pointerId === pointerId) setFromEvent(e); });
  track.addEventListener('pointerup', (e) => { if (e.pointerId === pointerId) { draggingCommand = false; pointerId = null; } });
}

function bindDialHud(hud, axis) {
  const svg = hud.querySelector('.hud-dial');
  let pointerId = null;
  const setFromEvent = (e) => {
    const rect = svg.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    // 0deg at top (12 o'clock), positive clockwise — same convention as setHudYaw's `deg`.
    let angle = Math.atan2(dx, -dy) * 180 / Math.PI;
    angle = Math.max(-135, Math.min(135, angle));
    const t = Math.max(-1, Math.min(1, -angle / 135));
    const scale = axisScale(commandRanges?.[axis]);
    const value = clampToRange(t * scale, commandRanges?.[axis]);
    cruiseYaw = value;
    engageManualIfNeeded();
    sendCruiseCommand();
  };
  svg.addEventListener('pointerdown', (e) => {
    if (hud.classList.contains('disabled')) return;
    draggingCommand = true; pointerId = e.pointerId;
    svg.setPointerCapture(e.pointerId);
    setFromEvent(e);
  });
  svg.addEventListener('pointermove', (e) => { if (draggingCommand && e.pointerId === pointerId) setFromEvent(e); });
  svg.addEventListener('pointerup', (e) => { if (e.pointerId === pointerId) { draggingCommand = false; pointerId = null; } });
}

bindVerticalHud(hudVx, 'vx');
bindHorizontalHud(hudVy, 'vy');
bindDialHud(hudYaw, 'yaw');

// ---- mouse look (yaw) — click-and-drag, NOT Pointer Lock ----
// Pointer Lock would hide the cursor and capture ALL mouse input, which
// breaks clicking every other button in this panel, doesn't work over the
// cross-origin Simulator iframe, and is force-released by the browser on
// Escape in a way pages can't override. Pointer Capture on
// a small dedicated pad gets 90% of the "mouse look" feel — drag past the
// pad's edges and movement keeps being tracked — with none of that risk,
// and it's just a mousemove listener: no cost to render speed, and nothing
// about it is simulator- or Genesis-specific, so it works identically
// against a real robot later (it's still just calling set_command).
const lookPad = $('#look-pad');
let lookPadLastX = 0;
const MOUSE_YAW_SENSITIVITY = 0.006; // rad/s of yaw per pixel dragged

lookPad.addEventListener('pointerdown', (e) => {
  mouseLookActive = true;
  lookPadLastX = e.clientX;
  lookPad.setPointerCapture(e.pointerId);
  lookPad.classList.add('active');
});
lookPad.addEventListener('pointermove', (e) => {
  if (!mouseLookActive) return;
  const dx = e.clientX - lookPadLastX;
  lookPadLastX = e.clientX;
  if (dx === 0) return;
  engageManualIfNeeded();
  // Dragging right should turn right; yaw+ means "turn left" (base_ang_vel[:,2],
  // right-hand rule around z), matching keymap.json's ArrowLeft: sign +1 —
  // so drag-right subtracts.
  cruiseYaw = clampToRange(cruiseYaw - dx * MOUSE_YAW_SENSITIVITY, commandRanges?.yaw);
  sendCruiseCommand();
});
function endMouseLook() {
  mouseLookActive = false;
  lookPad.classList.remove('active');
}
lookPad.addEventListener('pointerup', endMouseLook);
lookPad.addEventListener('pointercancel', endMouseLook);

// ---- keyboard shortcuts ----
// Gated on `keysArmed` (Simulator/Real-robot tab active), NOT on DOM focus —
// shortcuts must always work in that mode, per design, without the user
// having to click back into the sidebar first. The one wrinkle: a
// cross-origin iframe (the Simulator/Real-robot view) can steal actual
// browser focus when clicked, and then it — not this page — receives
// keydown/keyup. Rather than requiring the user to re-arm shortcuts
// manually, we detect that and immediately steal focus back (see the
// window blur handler below), since viser doesn't need keyboard focus for
// its mouse-driven orbit controls.

window.addEventListener('blur', () => {
  // Standard trick for detecting "focus moved into an iframe" from the
  // parent page — see HANDOFF_control_web.md §3-B "Keyboard shortcuts".
  setTimeout(() => {
    const active = document.activeElement;
    if (active && active.tagName === 'IFRAME') {
      if (keysArmed) panel.focus(); // steal focus back so shortcuts stay live
      return;
    }
    // Focus stayed in this document (e.g. the OS switched to another app
    // entirely) — we can't recapture that, so any physical keyup for a
    // currently-held key would be missed and the normal decel-on-release
    // (see frame()/smoothAxis) would never kick in. Zero out immediately
    // instead of leaving it stuck at speed with nothing to stop it.
    if (keysArmed && !document.hasFocus() && heldMoveKeys.size > 0) {
      heldMoveKeys.forEach((key) => setKeycapActive(key, false));
      heldMoveKeys.clear();
      cruiseVx = 0;
      cruiseVy = 0;
      cruiseYaw = 0;
      sendCruiseCommand();
    }
  }, 0);
});

// Shared by real keydown/keyup and keycap click/pointer handlers, so the
// action-dispatch logic exists in exactly one place.
function dispatchKeyAction(key) {
  // Policy-switch shortcuts are looked up first: they're assigned by list
  // position (recomputePolicyShortcuts()), not present in `keymap` at all.
  const policyName = policyByKey[key];
  if (policyName) { send('request_switch', { name: policyName }); return; }
  const binding = keymap[key];
  if (!binding) return;
  if (binding.action === 'pause_toggle') send(latestStatus?.paused ? 'resume' : 'pause');
  else if (binding.action === 'restart') $('#btn-restart').click();
}

function setKeycapActive(key, active) {
  // Multiple keycaps can share the same data-key (e.g. W and ArrowUp both
  // map to vx) — light up every matching keycap, not just the first one
  // in DOM order.
  document.querySelectorAll(`.keycap[data-key="${CSS.escape(key)}"]`).forEach((cap) => {
    cap.classList.toggle('active', active);
  });
}

// Keys the browser would otherwise use to scroll a focused, scrollable
// element (the sidebar) — block that default whenever shortcuts are armed,
// even for keys with no control bound to them, so pressing e.g. ArrowUp
// can't scroll the panel out from under the user.
const SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'PageUp', 'PageDown', 'Home', 'End']);

// The Create Policy form (and any other text/number/select field in the
// panel) needs to receive normal typing — 'p', 'r', '0'-'9', the arrow
// keys, and space are all bound shortcuts that would otherwise fire
// instead of being typed. `keysArmed` alone isn't enough to prevent that:
// it's about which VIEW has focus (sim vs. a drawer), not about what's
// focused within the panel.
function isTypingTarget(e) {
  const t = e.target;
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
}

document.addEventListener('keydown', (e) => {
  if (!keysArmed || isTypingTarget(e)) return;
  if (SCROLL_KEYS.has(e.key)) e.preventDefault();
  const binding = keymap[e.key];
  if (!binding && !policyByKey[e.key]) return;
  e.preventDefault();
  setKeycapActive(e.key, true);
  if (binding?.action === 'move') {
    if (heldMoveKeys.has(e.key)) return; // ignore OS key-repeat, already engaged
    heldMoveKeys.add(e.key);
    engageManualIfNeeded();
    // No immediate rampTick() call here — the rAF loop (frame()) picks up a
    // newly-held key within ~16ms on its own, so a manual kick isn't needed.
  } else {
    dispatchKeyAction(e.key);
  }
});

document.addEventListener('keyup', (e) => {
  if (isTypingTarget(e)) return;
  setKeycapActive(e.key, false);
  const binding = keymap[e.key];
  if (!binding || binding.action !== 'move') return;
  // No explicit decel kick needed here: releasing a key just drops it from
  // heldMoveKeys, so the next frame's heldDirection() for that axis is 0
  // and smoothAxis() takes over, coasting it back to zero on its own.
  heldMoveKeys.delete(e.key);
});

// ---- keycap click/pointer: clicking an on-screen key acts like pressing it ----
// Bound from boot() once `keymap` has actually been fetched — data-key
// lookups against an empty keymap would silently bind nothing.

function bindKeycapActions() {
  document.querySelectorAll('.keycap[data-key]').forEach((cap) => {
    const key = cap.dataset.key;
    const binding = keymap[key];
    // Policy-switch keycaps (the static Keyboard-legend digits) aren't in
    // `keymap` and `policyByKey` isn't populated yet at boot — bind by
    // static key-space membership instead; dispatchKeyAction() resolves the
    // actual policy at click time, once a status push has arrived.
    if (!binding && !POLICY_SHORTCUT_KEYS.includes(key)) return;
    if (binding?.action === 'move') {
      cap.addEventListener('pointerdown', () => {
        cap.classList.add('active');
        heldMoveKeys.add(key);
        engageManualIfNeeded();
      });
      const release = () => {
        cap.classList.remove('active');
        heldMoveKeys.delete(key);
      };
      cap.addEventListener('pointerup', release);
      cap.addEventListener('pointerleave', release);
    } else {
      cap.addEventListener('click', () => dispatchKeyAction(key));
    }
  });
}

// ---- generic drag-to-reorder ----
// Shared by the draggable panel sections and the Policies list: pick up an
// item by its .drag-handle, reflow live among siblings as the pointer
// crosses their vertical midpoints, then hand off to a caller-supplied
// onReorder() to persist the new order (and, for policies, to recompute
// shortcut keys from it).

let dragState = null;

function onDragHandleDown(e) {
  const handle = e.currentTarget;
  const { container, itemSelector, item, onReorder } = handle._sortableCtx;
  handle.setPointerCapture(e.pointerId);
  dragState = { container, itemSelector, item, onReorder, pointerId: e.pointerId };
  item.classList.add('dragging');
  document.addEventListener('pointermove', onDragHandleMove);
  document.addEventListener('pointerup', onDragHandleUp, { once: true });
}

function onDragHandleMove(e) {
  if (!dragState) return;
  const { container, itemSelector, item } = dragState;
  const siblings = [...container.querySelectorAll(itemSelector)].filter((el) => el !== item);
  for (const sib of siblings) {
    const r = sib.getBoundingClientRect();
    const mid = r.top + r.height / 2;
    if (e.clientY < mid && (sib.compareDocumentPosition(item) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      container.insertBefore(item, sib);
      break;
    }
    if (e.clientY > mid && (item.compareDocumentPosition(sib) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      container.insertBefore(item, sib.nextSibling);
      break;
    }
  }
}

function onDragHandleUp() {
  const { item, onReorder } = dragState;
  item.classList.remove('dragging');
  document.removeEventListener('pointermove', onDragHandleMove);
  onReorder();
  dragState = null;
}

function makeSortable(container, itemSelector, handleSelector, onReorder) {
  container.querySelectorAll(itemSelector).forEach((item) => {
    const handle = item.querySelector(handleSelector);
    if (!handle) return;
    handle._sortableCtx = { container, itemSelector, item, onReorder };
    handle.addEventListener('pointerdown', onDragHandleDown);
  });
}

// ---- draggable panel sections (reorder + persist) ----

const PANEL_ORDER_KEY = 'giar.panelOrder.v1';

function initSectionDrag() {
  makeSortable(panel, '.panel-section', ':scope > .section-head > .drag-handle', savePanelOrder);
}

function savePanelOrder() {
  const order = [...panel.querySelectorAll('.panel-section')].map((s) => s.dataset.section);
  localStorage.setItem(PANEL_ORDER_KEY, JSON.stringify(order));
}

function restorePanelOrder() {
  const raw = localStorage.getItem(PANEL_ORDER_KEY);
  if (!raw) return;
  let order;
  try { order = JSON.parse(raw); } catch { return; }
  order.forEach((name) => {
    const s = panel.querySelector(`.panel-section[data-section="${name}"]`);
    if (s) panel.appendChild(s);
  });
}

// ---- dismissible help popovers ----

let openPopover = null;

function togglePopover(btn) {
  const pop = btn.parentElement.querySelector('.help-popover');
  if (openPopover === pop) { closePopover(); return; }
  if (openPopover) closePopover();
  pop.hidden = false;
  btn.setAttribute('aria-expanded', 'true');
  openPopover = pop;
  openPopover._btn = btn;
}

function closePopover() {
  if (!openPopover) return;
  openPopover.hidden = true;
  openPopover._btn.setAttribute('aria-expanded', 'false');
  openPopover = null;
}

function initPopovers() {
  document.querySelectorAll('.help-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      togglePopover(btn);
    });
  });
  document.addEventListener('click', (e) => {
    if (openPopover && !e.target.closest('.help-popover') && !e.target.closest('.help-btn')) closePopover();
  });
  // capture:true so a popover closes immediately on Escape.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && openPopover) closePopover();
  }, true);
}

// ---- system info bar ----
// The user asked for this explicitly: don't make them guess what hardware
// is running the sim/training — show it, and use it as the basis for the
// Create Policy panel's num_envs suggestion + time estimate below.

const systemSummary = $('#system-summary');
const systemDetailsBtn = $('#system-details-btn');
const systemDetails = $('#system-details');
let systemInfo = null;

function refreshSystemInfo() {
  call('system_info').then((info) => {
    systemInfo = info;
    const gpuNote = info.cuda_available ? 'CUDA available (unused — training runs CPU)'
      : info.mps_available ? 'Metal available (unused — training runs CPU)' : 'no GPU';
    systemSummary.textContent =
      `${info.cpu_brand} · ${info.cpu_count} cores · ${info.ram_gb ?? '?'} GB RAM · ` +
      `${info.simulator}/${info.genesis_backend} · ${gpuNote}`;
    systemDetails.innerHTML = '';
    const dl = document.createElement('dl');
    const rows = [
      ['OS', info.os], ['Machine', info.machine], ['CPU', info.cpu_brand],
      ['Cores', info.cpu_count], ['RAM', `${info.ram_gb ?? '?'} GB`],
      ['Simulator', info.simulator], ['Sim backend', info.genesis_backend],
      ['Control backend', info.control_backend],
      ['CUDA', info.cuda_available ? 'yes' : 'no'], ['Metal (MPS)', info.mps_available ? 'yes' : 'no'],
      ['Suggested envs', `${info.suggested_num_envs.comfortable}–${info.suggested_num_envs.upper}`],
    ];
    for (const [k, v] of rows) {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      dl.appendChild(dt); dl.appendChild(dd);
    }
    systemDetails.appendChild(dl);

    if (!envsFieldTouched) {
      trainEnvs.value = info.suggested_num_envs.comfortable;
    }
    trainEnvsHint.textContent =
      `Suggested for this machine: ${info.suggested_num_envs.comfortable}–${info.suggested_num_envs.upper} ` +
      `(${info.cpu_count} cores detected) — more scales training speed less on CPU past that.`;
    updateEstimate();
  }).catch((e) => {
    systemSummary.textContent = 'system info unavailable';
    console.warn('system_info unavailable:', e.message);
  });
}

systemDetailsBtn.addEventListener('click', () => {
  const opening = systemDetails.hidden;
  systemDetails.hidden = !opening;
  systemDetailsBtn.setAttribute('aria-expanded', String(opening));
});
document.addEventListener('click', (e) => {
  if (!systemDetails.hidden && !e.target.closest('#system-bar')) {
    systemDetails.hidden = true;
    systemDetailsBtn.setAttribute('aria-expanded', 'false');
  }
});

// ---- create policy (training) ----
// The whole point of this panel, per the user ask: never hide the actual
// command behind the form — #train-cmd-preview always shows exactly what
// ControlService.start_training()/TrainingManager.start() will run (see
// legged_gym/control/training.py's display_command construction, which this
// mirrors on purpose so the two never drift apart silently).

const btnNewPolicy = $('#btn-new-policy');
const createPolicyForm = $('#create-policy-form');
const trainName = $('#train-name');
const trainBase = $('#train-base');
const trainTask = $('#train-task');
const trainIters = $('#train-iters');
const trainMinutes = $('#train-minutes');
const trainEnvs = $('#train-envs');
const trainEnvsHint = $('#train-envs-hint');
const trainVxLo = $('#train-vx-lo'), trainVxHi = $('#train-vx-hi');
const trainVyLo = $('#train-vy-lo'), trainVyHi = $('#train-vy-hi');
const trainYawLo = $('#train-yaw-lo'), trainYawHi = $('#train-yaw-hi');
const trainVarSelect = $('#train-var-select');
const trainVarNote = $('#train-var-note');
const trainHeight = $('#train-height');
const trainHeightAbsoluteHint = $('#train-height-absolute-hint');
const trainHeightModeTabs = $('#train-height-mode');
const trainHeightAbsolute = $('#train-height-absolute');
const trainHeightRelative = $('#train-height-relative');
const trainHeightReference = $('#train-height-reference');
const trainHeightDir = $('#train-height-dir');
const trainHeightDelta = $('#train-height-delta');
const trainHeightExtreme = $('#train-height-extreme');
const trainExtremeDirTabs = $('#train-extreme-dir');
const trainExtremeNote = $('#train-extreme-note');
const trainPush = $('#train-push');
const trainPushVel = $('#train-push-vel');
const trainPushInterval = $('#train-push-interval');
const trainPushDir = $('#train-push-dir');
const trainEntropyCoef = $('#train-entropy-coef');
const trainRewardScales = $('#train-reward-scales');
const trainCmdPreview = $('#train-cmd-preview');
const trainEstimate = $('#train-estimate');
const trainError = $('#train-error');
const btnStartTraining = $('#btn-start-training');
const trainingJobsEl = $('#training-jobs');

let envsFieldTouched = false; // stop overwriting the field with the suggested
                               // default once the user has typed their own value
trainEnvs.addEventListener('input', () => { envsFieldTouched = true; }, { once: true });

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

let estimateDebounce = null;
// Mirrors legged_gym/scripts/web_train.py's TIME_BUDGET_CHUNK_ITERS — the
// wall-clock deadline for --max_minutes is only re-checked this often, so a
// run can overshoot its budget by up to this many iterations' worth of time.
const TIME_BUDGET_CHUNK_ITERS = 10;

function updateEstimate() {
  clearTimeout(estimateDebounce);
  estimateDebounce = setTimeout(() => {
    const iterations = parseInt(trainIters.value, 10);
    const minutes = parseFloat(trainMinutes.value);
    const numEnvs = parseInt(trainEnvs.value, 10);
    const hasIters = Number.isFinite(iterations) && iterations > 0;
    const hasMinutes = Number.isFinite(minutes) && minutes > 0;
    if (!hasIters && !hasMinutes) {
      trainEstimate.textContent = 'Estimated time: – (set an iteration count or a minute budget)';
      return;
    }
    if (!Number.isFinite(numEnvs) || numEnvs <= 0) { trainEstimate.textContent = 'Estimated time: –'; return; }
    // Estimates are never exact — per-iteration cost drifts with machine
    // load, so this is always phrased as "about", and a minutes budget
    // always shows the iteration count it's expected to buy (not just
    // "time-boxed to N minutes" with no sense of how much training that
    // actually is) — see TrainingManager.estimate()'s own docstring.
    call('estimate_training_time', {
      num_envs: numEnvs,
      max_iterations: hasIters ? iterations : null,
      max_minutes: hasMinutes ? minutes : null,
    }).then((est) => {
      if (est.basis !== 'measured') {
        trainEstimate.textContent = hasMinutes && !hasIters
          ? `Estimated time: time-boxed to ${minutes} minute${minutes === 1 ? '' : 's'} — no runs on this machine yet, so an iteration count can't be estimated (the first run calibrates this).`
          : 'Estimated time: no runs on this machine yet — the first one calibrates this estimate.';
        return;
      }
      const boundBy = hasIters && hasMinutes
        ? (est.iterations < iterations ? 'minutes' : 'iterations')
        : hasMinutes ? 'minutes' : 'iterations';
      trainEstimate.textContent = boundBy === 'minutes'
        ? `Estimated time: ~${est.iterations} iterations in ${minutes} minute${minutes === 1 ? '' : 's'} (approximate — based on ${est.samples} previous run${est.samples === 1 ? '' : 's'} on this machine; the wall-clock deadline is checked every ${TIME_BUDGET_CHUNK_ITERS} iterations, so the run may overshoot by a few iterations' worth of time).`
        : `Estimated time: ~${formatDuration(est.seconds)} for ${est.iterations} iterations (based on ${est.samples} previous run${est.samples === 1 ? '' : 's'} on this machine)${hasMinutes ? ` — also capped at ${minutes} minute${minutes === 1 ? '' : 's'}, whichever hits first` : ''}.`;
    }).catch(() => { trainEstimate.textContent = 'Estimated time: –'; });
  }, 250);
}

function showTrainError(msg) {
  trainError.textContent = msg;
  trainError.classList.toggle('show', !!msg);
}

// ---- target variable: absolute value, relative to a reference, or an ----
// ---- extreme (lowest/highest) push against a physical bound         ----
// One variable at a time, chosen from #train-var-select — populated from
// TrainingManager.VARIABLE_REGISTRY (legged_gym/control/training.py) via
// training_catalog()'s 'variables' (task-independent: label/unit/source/
// flag/note) and task_defaults()'s 'variables' (task-dependent: reference/
// range, refetched on every task/clone-from change). Absolute/Relative
// resolve the same way they always did; Extreme resolves to the variable's
// configured physical bound (range[0]/range[1]) rather than an
// unconstrained optimum — see the panel's own copy for why (a truly
// unbounded "lowest" has a degenerate solution: lying on the ground).
let targetMode = 'absolute';
let extremeDir = 'lowest';
let targetReference = null; // {value: number|null, label: string} | null
let targetRange = null;     // [min, max] | null

function selectedVariable() {
  return trainVarSelect.value;
}

function variableMeta() {
  return trainingCatalog?.variables?.[selectedVariable()] || null;
}

function findBaseHeightTarget(baseName) {
  const p = trainingCatalog?.base_policies.find((b) => b.name === baseName);
  return p ? p.base_height_target : undefined;
}

function renderVariableChrome() {
  const meta = variableMeta();
  const unit = meta?.unit || '';
  const label = meta?.label || 'value';
  trainVarNote.textContent = meta
    ? `${meta.source === 'sensor' ? 'Real sensor' : 'Simulator ground truth'} — ${meta.note || ''}`
    : '';
  trainHeightAbsoluteHint.textContent =
    `${label} (${unit}) the reward tracks — e.g. set this to match a crouch base you're cloning from, so training doesn't pull it back up to the task's standing value.`;
}

function renderTargetReference() {
  const unit = variableMeta()?.unit || '';
  trainHeightReference.textContent = (!targetReference || targetReference.value == null)
    ? 'Reference: – (pick a task or clone-from base first)'
    : `Reference: ${targetReference.value.toFixed(3)} ${unit} (${targetReference.label})`;
  renderExtremeNote();
  updateCommandPreview();
}

function renderExtremeNote() {
  const unit = variableMeta()?.unit || '';
  const label = variableMeta()?.label || 'value';
  if (!targetRange) {
    trainExtremeNote.textContent = 'Resolves to – (pick a task or clone-from base first)';
    return;
  }
  const bound = extremeDir === 'lowest' ? targetRange[0] : targetRange[1];
  trainExtremeNote.textContent = `Resolves to ${bound.toFixed(3)} ${unit} — this task's configured ${extremeDir} bound for ${label.toLowerCase()}.`;
}

function refreshTargetReference() {
  const base = trainBase.value, task = trainTask.value;
  const varKey = selectedVariable();
  if (base && varKey === 'base_height') {
    // Fast path: base_height's reference for a clone-from base is already
    // in training_catalog's base_policies list, no extra round trip.
    targetReference = { value: findBaseHeightTarget(base) ?? null, label: `${base}'s task default` };
    targetRange = null; // extreme mode's clamp is still task-scoped — fall through below for it
  }
  if (!task && !base) { targetReference = null; targetRange = null; renderTargetReference(); return; }
  const effectiveTask = task || trainingCatalog?.base_policies.find((b) => b.name === base)?.task;
  if (!effectiveTask) { renderTargetReference(); return; }
  call('task_defaults', { task: effectiveTask }).then((d) => {
    const v = d.variables?.[varKey];
    if (!base) {
      targetReference = { value: v?.reference ?? null, label: `${effectiveTask}'s default` };
    }
    targetRange = v?.range ?? null;
    renderTargetReference();
  }).catch(() => { if (!base) targetReference = null; targetRange = null; renderTargetReference(); });
}

trainVarSelect.addEventListener('change', () => {
  renderVariableChrome();
  refreshTargetReference();
});

trainHeightModeTabs.querySelectorAll('button').forEach((btn) => {
  btn.addEventListener('click', () => {
    targetMode = btn.dataset.mode;
    trainHeightModeTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
    trainHeightAbsolute.hidden = targetMode !== 'absolute';
    trainHeightRelative.hidden = targetMode !== 'relative';
    trainHeightExtreme.hidden = targetMode !== 'extreme';
    if (targetMode === 'relative' || targetMode === 'extreme') refreshTargetReference();
    updateCommandPreview();
  });
});

trainExtremeDirTabs.querySelectorAll('button').forEach((btn) => {
  btn.addEventListener('click', () => {
    extremeDir = btn.dataset.dir;
    trainExtremeDirTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
    renderExtremeNote();
    updateCommandPreview();
  });
});

function resolveHeight() {
  if (targetMode === 'absolute') {
    const raw = trainHeight.value.trim();
    return raw === '' ? null : parseFloat(raw);
  }
  if (targetMode === 'extreme') {
    if (!targetRange) return undefined; // can't resolve yet — validation error
    return extremeDir === 'lowest' ? targetRange[0] : targetRange[1];
  }
  const deltaRaw = trainHeightDelta.value.trim();
  if (deltaRaw === '') return null; // no delta entered — leave the target unset (task default)
  if (!targetReference || targetReference.value == null) return undefined; // can't resolve yet — validation error
  const sign = trainHeightDir.value === 'lower' ? -1 : 1;
  return targetReference.value + sign * parseFloat(deltaRaw);
}

// ---- reward weights (advanced) — every <Cfg>.rewards.scales term the ----
// chosen task actually defines, one optional override field each. This is
// the direct answer to "what reward configurations aren't in the UI yet" —
// rather than hand-picking a few (alive, base_height, ...) and leaving the
// rest CLI-only, every term the backend's task_defaults().reward_scales
// returns gets a field, sourced from the task's own config so it never
// drifts out of sync with what a task actually rewards (see
// TrainingManager.task_defaults()'s docstring). Blank = task default,
// exactly like every other optional field in this form.
function refreshRewardScaleFields() {
  const base = trainBase.value, task = trainTask.value;
  const effectiveTask = task || trainingCatalog?.base_policies.find((b) => b.name === base)?.task;
  if (!effectiveTask) { renderRewardScaleFields({}); return Promise.resolve(); }
  // Returns the promise (rather than firing-and-forgetting like every other
  // "refresh a task-dependent field" helper in this file) so
  // restoreTrainFormConfig() can wait for these inputs to exist before
  // writing a saved value into them — they don't exist until this async
  // fetch resolves and renderRewardScaleFields() builds the DOM.
  return call('task_defaults', { task: effectiveTask }).then((d) => {
    renderRewardScaleFields(d.reward_scales || {}, d.reward_scale_notes || {});
  }).catch(() => renderRewardScaleFields({}, {}));
}

function renderRewardScaleFields(scales, notes = {}) {
  // Preserve whatever the user already typed for a term that's still
  // present after a task change (fine-tuning bases share the same reward
  // vocabulary almost always) — only terms that no longer exist get
  // dropped, silently, same as any other field that stops applying.
  const prevValues = {};
  trainRewardScales.querySelectorAll('input').forEach((el) => {
    if (el.value.trim() !== '') prevValues[el.dataset.term] = el.value;
  });

  trainRewardScales.innerHTML = '';
  for (const [term, defaultValue] of Object.entries(scales).sort(([a], [b]) => a.localeCompare(b))) {
    const wrap = document.createElement('div');
    wrap.className = 'reward-scale-field';
    const label = document.createElement('label');
    label.textContent = notes[term] ? `${term} ⓘ` : term;
    label.htmlFor = `train-reward-${term}`;
    if (notes[term]) label.title = notes[term];
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.id = `train-reward-${term}`;
    input.dataset.term = term;
    input.placeholder = String(defaultValue);
    if (prevValues[term] !== undefined) input.value = prevValues[term];
    input.addEventListener('input', updateCommandPreview);
    input.addEventListener('change', updateCommandPreview);
    wrap.appendChild(label);
    wrap.appendChild(input);
    trainRewardScales.appendChild(wrap);
  }
}

function rewardScaleOverrides() {
  const overrides = {};
  trainRewardScales.querySelectorAll('input').forEach((el) => {
    const raw = el.value.trim();
    if (raw !== '') overrides[el.dataset.term] = parseFloat(raw);
  });
  return Object.keys(overrides).length ? overrides : null;
}

// ---- auto-naming a new policy after its clone-from base ----
// Default is "<base>_<n>", n incrementing past whatever's already taken —
// e.g. cloning from "stable_step_two" suggests "stable_step_two_2", cloning
// from "stable_step_3" suggests "stable_step_4" (a trailing number gets
// incremented in place instead of appended again). Shown as the Name
// field's placeholder, never written into the field itself — typing an
// explicit name always wins (composeTrainingParams() only falls back to
// this when the field is left blank), same as every other "leave blank for
// a computed default" field in this form.
const DEFAULT_NAME_PLACEHOLDER = 'e.g. cautious_v2';

function suggestedPolicyName() {
  const base = trainBase.value;
  if (!base) return null;
  const known = new Set([
    ...(trainingCatalog?.base_policies || []).map((p) => p.name),
    ...(latestStatus?.policies || []),
  ]);
  const m = base.match(/^(.*?)(\d+)$/);
  const prefix = m ? m[1] : `${base}_`;
  let n = m ? parseInt(m[2], 10) + 1 : 2;
  let candidate = `${prefix}${n}`;
  while (known.has(candidate)) { n += 1; candidate = `${prefix}${n}`; }
  return candidate;
}

function refreshNamePlaceholder() {
  trainName.placeholder = suggestedPolicyName() || DEFAULT_NAME_PLACEHOLDER;
  updateCommandPreview(); // the preview's --name should track the same fallback
}

// ---- remembering the last-used Create Policy config across sessions ----
// "recordar la config anterior" — every field except the (now auto-named)
// policy name itself gets snapshotted to localStorage on a successful
// submit and restored the next time the panel opens, so repeating a
// curriculum step (same task/base/entropy_coef/reward weights, just a new
// name and maybe a longer budget) doesn't mean retyping the whole form.
const TRAIN_FORM_STORAGE_KEY = 'giar.trainFormConfig.v1';

function snapshotTrainFormConfig() {
  const rewardScales = {};
  trainRewardScales.querySelectorAll('input').forEach((el) => {
    if (el.value.trim() !== '') rewardScales[el.dataset.term] = el.value;
  });
  return {
    task: trainTask.value, base: trainBase.value,
    numEnvs: trainEnvs.value, iters: trainIters.value, minutes: trainMinutes.value,
    vxLo: trainVxLo.value, vxHi: trainVxHi.value,
    vyLo: trainVyLo.value, vyHi: trainVyHi.value,
    yawLo: trainYawLo.value, yawHi: trainYawHi.value,
    varKey: trainVarSelect.value, targetMode,
    height: trainHeight.value, heightDir: trainHeightDir.value, heightDelta: trainHeightDelta.value,
    extremeDir,
    push: trainPush.value, pushVel: trainPushVel.value, pushInterval: trainPushInterval.value, pushDir: trainPushDir.value,
    entropyCoef: trainEntropyCoef.value,
    rewardScales,
  };
}

function saveTrainFormConfig() {
  try { localStorage.setItem(TRAIN_FORM_STORAGE_KEY, JSON.stringify(snapshotTrainFormConfig())); } catch { /* best-effort */ }
}

function loadTrainFormConfig() {
  try {
    const raw = localStorage.getItem(TRAIN_FORM_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

// Applies a saved config on top of whatever refreshTrainingCatalog() just
// populated the selects with — task/base are only restored if that name
// still exists (a deleted policy silently falls back to whatever's
// already selected, same "don't offer something no longer real" rule
// forget_source()/refreshTrainingCatalog() already follow elsewhere).
function restoreTrainFormConfig() {
  const cfg = loadTrainFormConfig();
  if (!cfg) return;

  if (cfg.task && trainingCatalog?.tasks?.includes(cfg.task)) trainTask.value = cfg.task;
  if (cfg.base && trainingCatalog?.base_policies?.some((p) => p.name === cfg.base)) trainBase.value = cfg.base;
  trainEnvs.value = cfg.numEnvs || '';
  trainIters.value = cfg.iters || '';
  trainMinutes.value = cfg.minutes || '';
  trainVxLo.value = cfg.vxLo || ''; trainVxHi.value = cfg.vxHi || '';
  trainVyLo.value = cfg.vyLo || ''; trainVyHi.value = cfg.vyHi || '';
  trainYawLo.value = cfg.yawLo || ''; trainYawHi.value = cfg.yawHi || '';
  if (cfg.varKey && trainingCatalog?.variables?.[cfg.varKey]) trainVarSelect.value = cfg.varKey;
  targetMode = cfg.targetMode || 'absolute';
  trainHeightModeTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.mode === targetMode));
  trainHeightAbsolute.hidden = targetMode !== 'absolute';
  trainHeightRelative.hidden = targetMode !== 'relative';
  trainHeightExtreme.hidden = targetMode !== 'extreme';
  trainHeight.value = cfg.height || '';
  trainHeightDir.value = cfg.heightDir || 'raise';
  trainHeightDelta.value = cfg.heightDelta || '';
  extremeDir = cfg.extremeDir || 'lowest';
  trainExtremeDirTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.dir === extremeDir));
  trainPush.value = cfg.push || '';
  trainPushVel.value = cfg.pushVel || '';
  trainPushInterval.value = cfg.pushInterval || '';
  trainPushDir.value = cfg.pushDir || '';
  trainEntropyCoef.value = cfg.entropyCoef || '';

  renderVariableChrome();
  refreshTargetReference();
  refreshNamePlaceholder();
  // Reward-scale inputs don't exist until this resolves (they're rebuilt
  // from the restored task/base) — write the saved overrides in once they
  // do, rather than racing renderRewardScaleFields()'s own DOM rebuild.
  refreshRewardScaleFields().then(() => {
    if (cfg.rewardScales) {
      for (const [term, val] of Object.entries(cfg.rewardScales)) {
        const el = trainRewardScales.querySelector(`input[data-term="${term}"]`);
        if (el) el.value = val;
      }
    }
    updateCommandPreview();
  });
}

function refreshTrainingCatalog() {
  call('training_catalog').then((catalog) => {
    trainingCatalog = catalog;
    const prevTask = trainTask.value, prevBase = trainBase.value;
    trainTask.innerHTML = '';
    for (const t of catalog.tasks) {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      const note = catalog.task_notes?.[t];
      if (note) opt.title = note;
      trainTask.appendChild(opt);
    }
    if (catalog.tasks.includes(prevTask)) trainTask.value = prevTask;

    trainBase.innerHTML = '<option value="">— from scratch —</option>';
    for (const p of catalog.base_policies) {
      const opt = document.createElement('option');
      opt.value = p.name;
      // Fine-tuning needs train_checkpoint (rsl_rl's raw format), not
      // checkpoint (the exported/deployable one) — see
      // TrainingManager._train_checkpoint_from_export()'s docstring for
      // why those are different files. A policy can have one without the
      // other, so these are two distinct reasons "Clone from" is disabled.
      opt.textContent = !p.checkpoint ? `${p.name} (no checkpoint on this machine)`
        : !p.train_checkpoint ? `${p.name} (no training checkpoint to fine-tune from)`
        : p.name;
      opt.disabled = !p.train_checkpoint;
      trainBase.appendChild(opt);
    }
    if (catalog.base_policies.some((p) => p.name === prevBase)) trainBase.value = prevBase;

    const prevVar = trainVarSelect.value;
    trainVarSelect.innerHTML = '';
    for (const [key, meta] of Object.entries(catalog.variables || {})) {
      const opt = document.createElement('option');
      opt.value = key; opt.textContent = meta.label;
      trainVarSelect.appendChild(opt);
    }
    if (catalog.variables?.[prevVar]) trainVarSelect.value = prevVar;
    renderVariableChrome();

    if (targetMode === 'relative' || targetMode === 'extreme') refreshTargetReference();
    refreshRewardScaleFields();
    refreshNamePlaceholder();
    updateCommandPreview();
  }).catch((e) => {
    // Not fatal — the panel just can't populate its selects yet (e.g. the
    // server doesn't have a TrainingManager configured at all). Silent:
    // this is checked again on every reconnect.
    console.warn('training_catalog unavailable:', e.message);
  });
}

function rangePair(loEl, hiEl) {
  const lo = loEl.value.trim(), hi = hiEl.value.trim();
  if (lo === '' && hi === '') return null;
  if (lo === '' || hi === '') return undefined; // incomplete — caller treats as a validation error
  return [parseFloat(lo), parseFloat(hi)];
}

function composeTrainingParams() {
  // Blank Name field = auto-name from the clone-from base (see
  // suggestedPolicyName()); typing an explicit name always overrides it.
  // Only from-scratch runs with nothing typed fall through to '' — nothing
  // to base a suggestion on there, so that's still a validation error.
  const name = trainName.value.trim() || suggestedPolicyName() || '';
  const task = trainTask.value;
  const iterations = parseInt(trainIters.value, 10);
  const minutes = parseFloat(trainMinutes.value);
  const numEnvs = parseInt(trainEnvs.value, 10);
  const base = trainBase.value || null;
  const cmdVx = rangePair(trainVxLo, trainVxHi);
  const cmdVy = rangePair(trainVyLo, trainVyHi);
  const cmdYaw = rangePair(trainYawLo, trainYawHi);
  const height = resolveHeight();
  const push = trainPush.value || null; // '' -> null -> "leave task default"
  const pushVelRaw = trainPushVel.value.trim();
  const pushVel = pushVelRaw === '' ? null : parseFloat(pushVelRaw);
  const pushIntervalRaw = trainPushInterval.value.trim();
  const pushInterval = pushIntervalRaw === '' ? null : parseFloat(pushIntervalRaw);
  const pushDir = trainPushDir.value || null;
  const entropyCoefRaw = trainEntropyCoef.value.trim();
  const entropyCoef = entropyCoefRaw === '' ? null : parseFloat(entropyCoefRaw);
  const rewardScales = rewardScaleOverrides();
  return {
    name, task, iterations: Number.isFinite(iterations) ? iterations : null,
    minutes: Number.isFinite(minutes) ? minutes : null, numEnvs, base, cmdVx, cmdVy, cmdYaw,
    height, push, pushVel, pushInterval, pushDir, entropyCoef, rewardScales,
  };
}

function updateCommandPreview() {
  const p = composeTrainingParams();
  const parts = [
    'python legged_gym/scripts/web_train.py',
    `--task ${p.task || '<task>'}`,
    `--name ${p.name || '<policy name>'}`,
    `--num_envs ${Number.isFinite(p.numEnvs) ? p.numEnvs : '<num_envs>'}`,
    '--headless --cpu',
    '--result_path <assigned by the server>',
  ];
  if (p.iterations !== null) parts.push(`--max_iterations ${p.iterations}`);
  if (p.minutes !== null) parts.push(`--max_minutes ${p.minutes}`);
  if (p.iterations === null && p.minutes === null) parts.push('--max_iterations <or> --max_minutes <required>');
  if (p.base) {
    const source = trainingCatalog?.base_policies.find((b) => b.name === p.base);
    parts.push(`--from_checkpoint ${source?.train_checkpoint || '<' + p.base + "'s training checkpoint>"}`);
  }
  if (p.cmdVx) parts.push(`--cmd_vx_range ${p.cmdVx[0]} ${p.cmdVx[1]}`);
  if (p.cmdVy) parts.push(`--cmd_vy_range ${p.cmdVy[0]} ${p.cmdVy[1]}`);
  if (p.cmdYaw) parts.push(`--cmd_yaw_range ${p.cmdYaw[0]} ${p.cmdYaw[1]}`);
  const targetFlag = variableMeta()?.flag || 'base_height_target';
  if (p.height === undefined) parts.push(`--${targetFlag} <no reference yet — pick a task or clone-from base>`);
  else if (p.height !== null) parts.push(`--${targetFlag} ${p.height.toFixed(3)}`);
  if (p.push) parts.push(`--push_robots ${p.push}`);
  if (p.pushVel !== null) parts.push(`--max_push_vel_xy ${p.pushVel}`);
  if (p.pushInterval !== null) parts.push(`--push_interval_s ${p.pushInterval}`);
  if (p.pushDir) parts.push(`--push_dir ${p.pushDir}`);
  if (p.entropyCoef !== null) parts.push(`--entropy_coef ${p.entropyCoef}`);
  if (p.rewardScales) {
    for (const [term, value] of Object.entries(p.rewardScales).sort(([a], [b]) => a.localeCompare(b))) {
      parts.push(`--reward_scale ${term} ${value}`);
    }
  }
  trainCmdPreview.textContent = parts.join(' ');
}

btnNewPolicy.addEventListener('click', () => {
  const opening = createPolicyForm.hidden;
  createPolicyForm.hidden = !opening;
  btnNewPolicy.textContent = opening ? 'Cancel' : '+ New policy…';
  if (opening) {
    showTrainError('');
    restoreTrainFormConfig(); // last-used config, if any — see its own docstring
    refreshNamePlaceholder();
    updateCommandPreview();
    updateEstimate();
    trainName.focus();
  }
});

[trainName, trainBase, trainTask, trainIters, trainMinutes, trainEnvs,
 trainVxLo, trainVxHi, trainVyLo, trainVyHi, trainYawLo, trainYawHi,
 trainHeight, trainHeightDir, trainHeightDelta, trainPush, trainPushVel, trainPushInterval, trainPushDir,
 trainEntropyCoef].forEach((el) => {
  el.addEventListener('input', updateCommandPreview);
  el.addEventListener('change', updateCommandPreview);
});
[trainBase, trainTask].forEach((el) => {
  el.addEventListener('change', () => {
    if (targetMode === 'relative' || targetMode === 'extreme') refreshTargetReference();
    refreshRewardScaleFields();
    refreshNamePlaceholder(); // the auto-name suggestion tracks "Clone from"
  });
});
[trainIters, trainMinutes, trainEnvs].forEach((el) => {
  el.addEventListener('input', updateEstimate);
  el.addEventListener('change', updateEstimate);
});

createPolicyForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const p = composeTrainingParams();
  if (!p.name) return showTrainError('Policy name is required.');
  if (!p.task) return showTrainError('Pick a task.');
  if (p.iterations === null && p.minutes === null) return showTrainError('Set a time budget — iterations, minutes, or both.');
  if (p.iterations !== null && p.iterations <= 0) return showTrainError('Iterations must be positive.');
  if (p.minutes !== null && p.minutes <= 0) return showTrainError('Minutes must be positive.');
  if (!Number.isFinite(p.numEnvs) || p.numEnvs <= 0) return showTrainError('Parallel environments must be positive.');
  if (p.cmdVx === undefined || p.cmdVy === undefined || p.cmdYaw === undefined) {
    return showTrainError('Fill in both ends of a command range, or leave the whole pair blank.');
  }
  if (p.height === undefined) {
    return showTrainError('No reference height to raise/lower from yet — pick a task or a clone-from base.');
  }

  showTrainError('');
  btnStartTraining.disabled = true;
  call('start_training', {
    policy_name: p.name, task: p.task, num_envs: p.numEnvs,
    max_iterations: p.iterations, max_minutes: p.minutes,
    base_policy: p.base, cmd_vx: p.cmdVx, cmd_vy: p.cmdVy, cmd_yaw: p.cmdYaw,
    base_height_target: p.height,
    push_robots: p.push === null ? null : p.push === 'on',
    max_push_vel_xy: p.pushVel, push_interval_s: p.pushInterval, push_dir: p.pushDir,
    entropy_coef: p.entropyCoef, reward_scale_overrides: p.rewardScales,
  }).then(() => {
    saveTrainFormConfig(); // snapshot BEFORE reset() clears every field below
    createPolicyForm.reset();
    createPolicyForm.hidden = true;
    btnNewPolicy.textContent = '+ New policy…';
    targetMode = 'absolute';
    trainHeightModeTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.mode === 'absolute'));
    trainHeightAbsolute.hidden = false;
    trainHeightRelative.hidden = true;
    trainHeightExtreme.hidden = true;
  }).catch((e) => {
    showTrainError(e.message);
  }).finally(() => {
    btnStartTraining.disabled = false;
  });
});

function renderTrainingJobs(jobs) {
  // Dedupe on everything except elapsed_s's raw per-tick ticking — elapsed_s
  // pushes every status update while a job runs (~10Hz), and rebuilding
  // this DOM that often would reintroduce exactly the perf issue the
  // policy-list dedupe above already exists to avoid (see
  // HANDOFF_control_web.md's post-merge incident note). iterations_done IS
  // included, deliberately — it only changes once per
  // TIME_BUDGET_CHUNK_ITERS chunk (web_train.py's write_progress(), a few
  // times a minute at most, not 10Hz), so re-rendering when it changes is
  // what actually shows live progress instead of a static "training…".
  const key = jobs.map((j) => `${j.id}:${j.status}:${j.error || ''}:${j.iterations_done ?? ''}`).join('|');
  if (key === renderedTrainingJobsKey) return;
  const justFinished = renderedTrainingJobsKey !== null &&
    jobs.some((j) => j.status === 'done' && !renderedTrainingJobsKey.includes(`${j.id}:done`));
  renderedTrainingJobsKey = key;

  // A finished job's whole story (command, paths, metrics) now lives in
  // the resulting policy's own info popup — see openPolicyInfo() /
  // TrainingManager.finalize_policy(). Nothing here ever prunes
  // TrainingManager.jobs, so without this filter every "done" card would
  // just accumulate in this panel forever, duplicating (with less detail)
  // what the Policies list's ⓘ button already shows. 'running' (live
  // progress) and 'failed' (needs attention, not recorded anywhere else)
  // still show here — those aren't policies yet.
  const visibleJobs = jobs.filter((j) => j.status !== 'done');

  trainingJobsEl.innerHTML = '';
  for (const job of visibleJobs) {
    const row = document.createElement('div');
    row.className = `job-row ${job.status}`;
    const head = document.createElement('div');
    head.className = 'job-head';
    const name = document.createElement('span');
    name.className = 'job-name';
    name.textContent = job.policy_name;
    const statusEl = document.createElement('span');
    statusEl.className = 'job-status';
    if (job.status === 'running') {
      // iterations_done is a live snapshot (updated every
      // TIME_BUDGET_CHUNK_ITERS iterations — see web_train.py's
      // write_progress()/TrainingManager._refresh_progress()), null until
      // the first chunk completes.
      const cap = job.max_iterations ? `/${job.max_iterations}` : '';
      statusEl.textContent = job.iterations_done != null
        ? `training… (${job.iterations_done}${cap} iterations, ${job.elapsed_s}s)`
        : `training… (${job.elapsed_s}s)`;
    } else {
      statusEl.textContent = `${job.status} (${job.elapsed_s}s)`;
    }
    head.appendChild(name);
    head.appendChild(statusEl);
    row.appendChild(head);
    const cmd = document.createElement('div');
    cmd.className = 'job-cmd';
    cmd.textContent = job.command;
    row.appendChild(cmd);
    if (job.error) {
      const err = document.createElement('div');
      err.className = 'job-error';
      err.textContent = job.error;
      row.appendChild(err);
    }
    trainingJobsEl.appendChild(row);
  }

  // A job finishing means a new policy just got hot-loaded (see
  // swap_experiment.py's drain_finished_training()) — refresh the catalog
  // so it's immediately choosable as a fine-tuning base too.
  if (justFinished) refreshTrainingCatalog();
}

// ---- policy info popup ----
// Backed by ControlService.policy_info() (legged_gym/control/service.py),
// which reads policies/<name>/meta.json — written once, at the end of
// TrainingManager.finalize_policy(), from the job that produced the policy
// PLUS a parse of that job's own .log file (see
// legged_gym/control/training.py's parse_training_log()). Nothing here
// talks to the log file directly; everything shown is already baked into
// meta.json by the time this popup can open. A policy with no meta.json at
// all (an externally-sourced checkpoint, e.g. `stable`) gets the "no
// training info" empty state instead of an error — see policy_info()'s own
// docstring for why that's a None, not a raised error.

const policyInfoDock = $('#policy-info-dock');
const policyInfoTitle = $('#policy-info-title');
const policyInfoBody = $('#policy-info-body');
const policyInfoClose = $('#policy-info-close');
const policyInfoPrev = $('#policy-info-prev');
const policyInfoNext = $('#policy-info-next');

let policyInfoCurrentName = null; // whichever policy the popup is showing right now —
                                   // drives the up/down (prev/next) nav buttons below

// The up/down buttons walk the SAME order as the Policies list (drag order,
// top to bottom) — reading it straight off the live DOM (like
// onPolicyReorder() already does) rather than keeping a second copy of the
// order in sync, so it can never drift from what the user actually sees.
function policyOrderNames() {
  return [...policyList.querySelectorAll('.policy-row')].map((r) => r.dataset.policy);
}

function openPolicyInfo(name) {
  policyInfoCurrentName = name;
  policyInfoTitle.value = name;
  policyInfoBody.innerHTML = 'Loading…';
  policyInfoDock.hidden = false;
  policyInfoOpen = true;
  setKeysArmed();

  const order = policyOrderNames();
  const idx = order.indexOf(name);
  policyInfoPrev.disabled = idx <= 0;
  policyInfoNext.disabled = idx === -1 || idx >= order.length - 1;

  call('policy_info', { name }).then((info) => {
    policyInfoBody.innerHTML = '';
    policyInfoBody.appendChild(info ? renderPolicyInfo(info) : renderPolicyInfoEmpty());
  }).catch((e) => {
    policyInfoBody.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'info-empty';
    p.textContent = `Couldn't load info: ${e.message}`;
    policyInfoBody.appendChild(p);
  });
}

function stepPolicyInfo(delta) {
  const order = policyOrderNames();
  const idx = order.indexOf(policyInfoCurrentName);
  if (idx === -1) return;
  const nextIdx = idx + delta;
  if (nextIdx < 0 || nextIdx >= order.length) return;
  const name = order[nextIdx];
  // Stepping through policies here is meant for side-by-side comparison —
  // what you're reading in the dock and what's actually running in the
  // simulator should always be the same one, so ↑/↓ also switches the
  // active policy (same request the row's own button sends), not just the
  // info shown. Opening a SPECIFIC policy's info via its ⓘ button (see
  // policyButtonRow()) deliberately does NOT do this — inspecting a policy
  // you're not currently running shouldn't switch you onto it.
  send('request_switch', { name });
  openPolicyInfo(name);
}

function closePolicyInfo() {
  policyInfoDock.hidden = true;
  policyInfoOpen = false;
  policyInfoCurrentName = null;
  setKeysArmed();
}

policyInfoClose.addEventListener('click', closePolicyInfo);
policyInfoPrev.addEventListener('click', () => stepPolicyInfo(-1));
policyInfoNext.addEventListener('click', () => stepPolicyInfo(1));

// ---- inline rename — the title is now an editable field. Renaming a
// policy touches more than one file on disk (see TrainingManager.
// rename_policy()'s docstring), so this always goes through the backend
// call rather than just relabeling something client-side. ----
function commitPolicyRename() {
  const oldName = policyInfoCurrentName;
  const newName = policyInfoTitle.value.trim();
  if (!oldName || !newName || newName === oldName) {
    policyInfoTitle.value = oldName || '';
    return;
  }
  policyInfoTitle.disabled = true;
  call('rename_policy', { old_name: oldName, new_name: newName }).then(() => {
    policyInfoCurrentName = newName;
    policyInfoTitle.disabled = false;
    policyInfoTitle.value = newName;
    renamePolicyOrder(oldName, newName);
    // The Policies list picks up the new name from the next status push;
    // the Clone-from catalog is a separate fetch, same reason
    // delete_policy's onclick above refreshes it explicitly.
    refreshTrainingCatalog();
  }).catch((err) => {
    policyInfoTitle.disabled = false;
    policyInfoTitle.value = oldName;
    window.alert(`Couldn't rename "${oldName}": ${err.message}`);
  });
}
policyInfoTitle.addEventListener('blur', commitPolicyRename);
policyInfoTitle.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); policyInfoTitle.blur(); }
  else if (e.key === 'Escape') {
    e.preventDefault();
    policyInfoTitle.value = policyInfoCurrentName || '';
    policyInfoTitle.blur();
  }
});

// No backdrop anymore — it's a permanent docked column now, not an overlay,
// so there's no "outside" click to close it; Escape/✕/prev-next buttons are
// the only ways out. Moving between policies is button/click-only (see
// policyInfoPrev/Next above) — arrow keys are NOT bound to this dock, so
// they always mean robot-control, dock open or not (see setKeysArmed()).
document.addEventListener('keydown', (e) => {
  if (!policyInfoOpen) return;
  if (e.target === policyInfoTitle) return; // renaming — the input's own handler owns Enter/Escape
  if (e.key === 'Escape') { closePolicyInfo(); return; }
}, true);

function renderPolicyInfoEmpty() {
  const p = document.createElement('p');
  p.className = 'info-empty';
  p.textContent = 'No training info recorded for this policy — it was registered from a checkpoint '
    + "trained outside this UI (e.g. an externally-sourced .pt with no local training history), "
    + 'so there is no command/log to show.';
  return p;
}

function fmtDate(epochSeconds) {
  if (epochSeconds == null) return '–';
  return new Date(epochSeconds * 1000).toLocaleString();
}

function fmtDuration(seconds) {
  if (seconds == null) return '–';
  const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function infoSection(titleText) {
  const section = document.createElement('div');
  section.className = 'info-section';
  const h = document.createElement('h3');
  h.textContent = titleText;
  section.appendChild(h);
  return section;
}

// Each pair is [label, value] (skipped entirely when value is null/empty —
// most rows here are genuinely "not applicable" and shouldn't take up
// space) or [label, value, placeholder] to force the row to ALWAYS render,
// showing `placeholder` when empty — for a field like "Cloned from" that
// should always occupy the same line, so the popup's layout doesn't jump
// around every time the up/down nav (see stepPolicyInfo()) lands on a
// policy trained from scratch vs. one that was fine-tuned.
function kvList(pairs) {
  const dl = document.createElement('dl');
  dl.className = 'info-kv';
  for (const [k, v, placeholder] of pairs) {
    const empty = v == null || v === '';
    if (empty && placeholder === undefined) continue;
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = empty ? placeholder : v;
    dl.appendChild(dt); dl.appendChild(dd);
  }
  return dl;
}

// Shared between the stat cards (renderPolicyInfo) and the chart lines
// below, so a card's color always matches its curve. episode_length is
// listed first — it's the metric currently open (see
// HANDOFF_stability_curriculum.md §6, the plateau that isn't diagnosed
// yet), everything else is secondary context for now.
const METRIC_SPECS = [
  { key: 'episode_length', label: 'Episode length', color: 'var(--accent, #c77b2f)' },
  { key: 'reward', label: 'Mean reward', color: 'var(--accent2)' },
  { key: 'noise_std', label: 'Action noise std', color: 'var(--danger, #d9534f)' },
];

// Plain inline SVG polylines — no charting dependency, three series on
// independent 0–max scales (their units aren't comparable) sharing one
// x-axis. Didactic over precise: this is "did it trend the right way", not
// a dashboard.
//
// x-axis is normalized to 0–100% of THIS policy's own run (iteration count
// varies a lot between a from-scratch run and a resumed curriculum step —
// see HANDOFF_stability_curriculum.md §5) with gridlines drawn at fixed
// 0/25/50/75/100% pixel positions. The gridlines, not the data, are what
// give the eye a constant reference — without them the same pixel column
// means a different fraction-of-training on every policy, which is what
// made stepping through policies with ↑/↓ feel like the chart "jumped".
function renderSeriesChart(series) {
  const width = 560, height = 100, pad = 4, tickH = 8;
  const plotH = height - tickH;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.classList.add('info-chart');

  const iterations = series.map((p) => p.iteration);
  const xMin = Math.min(...iterations), xMax = Math.max(...iterations) || 1;
  const progress = (it) => (xMax === xMin ? 0 : (it - xMin) / (xMax - xMin)); // 0..1, this run's own span
  const xScale = (it) => pad + (width - 2 * pad) * progress(it);

  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const x = xScale(xMin + frac * (xMax - xMin));
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x.toFixed(1)); line.setAttribute('x2', x.toFixed(1));
    line.setAttribute('y1', '0'); line.setAttribute('y2', plotH.toFixed(1));
    line.classList.add('info-chart-grid');
    svg.appendChild(line);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', x.toFixed(1)); label.setAttribute('y', height - 1);
    label.setAttribute('text-anchor', frac === 0 ? 'start' : frac === 1 ? 'end' : 'middle');
    label.classList.add('info-chart-tick');
    label.textContent = `${Math.round(frac * 100)}%`;
    svg.appendChild(label);
  }

  for (const spec of METRIC_SPECS) {
    const values = series.map((p) => p[spec.key]);
    const yMax = Math.max(...values, 1e-6), yMin = Math.min(...values, 0);
    const yScale = (v) => plotH - pad - (plotH - 2 * pad) * ((v - yMin) / ((yMax - yMin) || 1));
    const points = series.map((p) => `${xScale(p.iteration).toFixed(1)},${yScale(p[spec.key]).toFixed(1)}`).join(' ');
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', points);
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', spec.color);
    line.setAttribute('stroke-width', '1.5');
    svg.appendChild(line);
  }
  return svg;
}

// Plain-language explanations for `legged_robot.py`'s `_reward_<name>` /
// `g1.py`'s `_reward_<name>` functions — this site is didactic first, so a
// reward term should never appear as an unexplained identifier. Sign
// convention noted where it isn't obvious ("penalizes X" = negative scale
// in the task config, "rewards X" = positive). Falls back to the term name
// itself in renderRewardTerms() if a term isn't listed here yet.
const REWARD_TERM_GLOSSARY = {
  base_height: 'Penalizes the pelvis being above/below a target height off the ground — pulls the robot toward a specific standing/crouching posture.',
  tracking_lin_vel: 'Rewards matching the commanded forward/strafe velocity (the WASD/arrow-key command) — how well it goes where it’s told.',
  tracking_ang_vel: 'Rewards matching the commanded turn rate (yaw) — how well it turns at the commanded speed.',
  orientation: 'Penalizes the body tilting away from upright (gravity not pointing straight down in the body frame) — keeps it from leaning or toppling.',
  lin_vel_z: 'Penalizes vertical (bouncing) velocity of the base — discourages bobbing up and down.',
  ang_vel_xy: 'Penalizes roll/pitch angular velocity — discourages wobbling side to side or front to back.',
  contact: 'Rewards feet touching down at the right point in the walking gait cycle — encourages a proper stance/swing rhythm instead of shuffling or dragging.',
  contact_no_vel: 'Penalizes a foot moving while it’s supposed to be planted on the ground — discourages slipping/skidding on stance feet.',
  alive: 'A flat per-step bonus just for still being upright — the direct incentive to survive longer, not fall.',
  hip_pos: 'Penalizes hip joints straying from their default/neutral angle — keeps the stance from splaying too wide or crossing.',
  feet_swing_height: 'Rewards lifting a swinging foot to a target clearance height — keeps it from scuffing/tripping on the ground mid-step.',
  feet_air_time: 'Rewards a foot staying airborne for a healthy swing duration — discourages tiny shuffling steps.',
  dof_acc: 'Penalizes joint acceleration — discourages jerky, high-effort motion.',
  dof_vel: 'Penalizes joint velocity — discourages fast, thrashy joint motion.',
  dof_power: 'Penalizes mechanical power drawn by the joints (torque × velocity) — a rough energy-efficiency term.',
  torques: 'Penalizes the raw torque commanded at each joint — discourages straining the motors.',
  action_rate: 'Penalizes the action changing a lot step-to-step — discourages twitchy, high-frequency control output.',
  action_smoothness: 'Penalizes changes in the RATE of action change (second derivative) — a stronger smoothness term than action_rate.',
  dof_pos_limits: 'Penalizes joints approaching their physical range-of-motion limits — keeps it away from the mechanical stops.',
  torque_limits: 'Penalizes torque commands approaching the motor’s rated limit.',
  collision: 'Penalizes non-foot body parts (shins, torso, etc.) touching something — discourages the robot bumping into itself or the environment.',
  termination: 'A one-time penalty applied the instant an episode ends in failure (e.g. a fall) — separate from `alive`, which pays out every step it doesn’t.',
  no_fly: 'Penalizes having zero feet in contact with the ground at once — discourages an ungrounded/floating gait.',
  feet_stumble: 'Penalizes a foot catching on a vertical surface (like a step edge) while moving horizontally — a stumble detector.',
  feet_slip: 'Penalizes horizontal foot velocity while that foot is in contact — another anti-skidding term, foot-velocity-based rather than binary.',
  foot_clearance: 'Rewards a swinging foot reaching a target height at the right point in its swing — similar intent to feet_swing_height, different formulation.',
  foot_landing_vel: 'Penalizes a foot moving fast (vertically) at the moment it touches down — discourages hard, stomping landings.',
  foot_acc: 'Penalizes foot acceleration — discourages sudden, jerky foot motion.',
  keep_balance: 'A composite balance-related bonus specific to this task/robot — see `_reward_keep_balance()` in the env source for the exact formula.',
  dof_vel_stand_still: 'Penalizes joint velocity while the commanded velocity is ~zero — keeps a "stand still" command from turning into idle fidgeting.',
  dof_pos_stand_still: 'Penalizes joints drifting from default pose while the commanded velocity is ~zero — same idea, for posture instead of motion.',
  feet_contact_stand_still: 'Rewards keeping both feet planted while the commanded velocity is ~zero — discourages stepping in place when told to stand still.',
  dof_close_to_default: 'Penalizes joints straying from their default pose in general — a mild prior toward a "neutral" posture.',
  dof_close_to_default_stand_still: 'Same as dof_close_to_default, but only applied while the commanded velocity is ~zero.',
};

function renderRewardTerms(terms) {
  const wrap = document.createElement('div');
  wrap.className = 'info-terms';
  const maxAbs = Math.max(...Object.values(terms).map(Math.abs), 1e-9);
  for (const [term, value] of Object.entries(terms).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))) {
    const row = document.createElement('div');
    row.className = 'info-term-row';
    const name = document.createElement('span');
    name.className = 'info-term-name';
    name.tabIndex = 0;
    name.setAttribute('data-tip', REWARD_TERM_GLOSSARY[term] || `No description written for "${term}" yet.`);
    const nameLabel = document.createElement('span');
    nameLabel.className = 'info-term-name-label';
    nameLabel.textContent = term;
    name.appendChild(nameLabel);
    const track = document.createElement('div');
    track.className = 'info-term-bar-track';
    const bar = document.createElement('div');
    bar.className = 'info-term-bar';
    // Track spans [-maxAbs, +maxAbs] centered at 50% — a term's bar grows
    // right from center if it rewards (positive), left if it penalizes.
    const halfPct = (Math.abs(value) / maxAbs) * 50;
    bar.style.width = `${halfPct}%`;
    bar.style.left = value >= 0 ? '50%' : `${50 - halfPct}%`;
    bar.style.background = value >= 0 ? 'var(--accent2)' : 'var(--danger, #d9534f)';
    track.appendChild(bar);
    const val = document.createElement('span');
    val.className = 'info-term-val';
    val.textContent = value.toFixed(4);
    row.appendChild(name); row.appendChild(track); row.appendChild(val);
    wrap.appendChild(row);
  }
  return wrap;
}

function renderPolicyInfo(info) {
  const frag = document.createDocumentFragment();

  // Result (stat cards + chart) always renders FIRST — see #policy-info-top
  // in index.html — so comparing policies with ↑/↓ never requires
  // scrolling past Provenance/Training-run to reach the numbers that
  // actually matter for comparison.
  if (info.metrics?.final) {
    const top = document.createElement('div');
    top.id = 'policy-info-top';

    const results = infoSection('Result');
    const statRow = document.createElement('div');
    statRow.className = 'info-stat-row';
    for (const spec of METRIC_SPECS) {
      const val = info.metrics.final[spec.key];
      const box = document.createElement('div');
      box.className = spec.key === 'episode_length' ? 'info-stat featured' : 'info-stat';
      box.style.setProperty('--stat-color', spec.color);
      const v = document.createElement('div'); v.className = 'val';
      v.textContent = val != null ? val.toFixed(spec.key === 'reward' || spec.key === 'noise_std' ? 3 : 1) : '–';
      const l = document.createElement('div'); l.className = 'lbl'; l.textContent = spec.label;
      box.appendChild(v); box.appendChild(l);
      statRow.appendChild(box);
    }
    results.appendChild(statRow);

    if (info.metrics.series?.length > 1) {
      results.appendChild(renderSeriesChart(info.metrics.series));
      const caption = document.createElement('div');
      caption.className = 'info-chart-caption';
      caption.textContent = `${info.metrics.series[0].iteration}→${info.metrics.series[info.metrics.series.length - 1].iteration} iterations, normalized to % of this run — episode length / reward / noise std`;
      results.appendChild(caption);
    }
    top.appendChild(results);
    frag.appendChild(top);
  }

  const provenance = infoSection('Provenance');
  provenance.appendChild(kvList([
    ['Task', info.task],
    ['Cloned from', info.base_policy, '– (trained from scratch)'],
    ['Trained via', info.trained_via],
    ['Created', fmtDate(info.created_at)],
  ]));
  if (info.command) {
    const cmd = document.createElement('code');
    cmd.className = 'info-cmd';
    cmd.textContent = info.command;
    provenance.appendChild(cmd);
  }
  frag.appendChild(provenance);

  if (info.entropy_coef != null || (info.reward_scale_overrides && Object.keys(info.reward_scale_overrides).length)) {
    const tuning = infoSection('Tuning overrides (vs. task defaults)');
    const pairs = [['entropy_coef', info.entropy_coef]];
    for (const [k, v] of Object.entries(info.reward_scale_overrides || {})) pairs.push([`reward: ${k}`, v]);
    tuning.appendChild(kvList(pairs));
    frag.appendChild(tuning);
  }

  const run = infoSection('Training run');
  run.appendChild(kvList([
    ['Iterations', info.iterations_done],
    ['Requested budget', info.max_iterations ? `${info.max_iterations} iterations` : (info.max_minutes ? `${info.max_minutes} min` : null)],
    ['Wall time', fmtDuration(info.elapsed_s)],
    ['Parallel envs', info.num_envs],
  ]));
  frag.appendChild(run);

  if (info.metrics?.final_reward_terms) {
    const terms = infoSection('What the reward actually optimized (final iteration)');
    terms.appendChild(renderRewardTerms(info.metrics.final_reward_terms));
    frag.appendChild(terms);
  }

  const paths = infoSection('Files');
  paths.appendChild(kvList([
    ['Checkpoint', info.checkpoint_path],
    ['Fine-tunable from', info.has_train_checkpoint ? 'yes (train_checkpoint.pt)' : 'no'],
    ['Full log', info.log_path],
  ]));
  frag.appendChild(paths);

  return frag;
}

// ---- boot ----

async function boot() {
  const [config, km] = await Promise.all([
    fetch('/config').then((r) => r.json()),
    fetch('/keymap.json').then((r) => r.json()),
  ]);
  keymap = Object.fromEntries(
    Object.entries(km).filter(([k, v]) => v && typeof v === 'object' && !k.startsWith('_'))
  );
  // Policy-switch shortcuts (keyByPolicy/policyByKey) are NOT loaded from
  // keymap.json — they're derived from list position on every render; see
  // recomputePolicyShortcuts().

  // The Simulator is the permanent base layer (see the block comment near
  // mountSimIframe()) — mount it once, right here at boot, and never again.
  viserPort = config.viser_port;
  mountSimIframe();

  // Clamp the HUDs (and, via commandRanges, the arrow keys) to the exact
  // velocity envelope this policy was trained across (env_cfg.commands.ranges
  // — see SimAdapter.set_command), not an arbitrary UI guess.
  if (config.command_ranges) {
    commandRanges = config.command_ranges;
  }

  bindKeycapActions();
  initSectionDrag();
  restorePanelOrder();
  initPopovers();

  connect();
  // keysArmed starts true — no drawer is open at boot, so the (always-
  // mounted) Simulator is what's showing; see openDrawer()/closeDrawer().
  setKeysArmed();
}

boot();
