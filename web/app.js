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
let policyListExpanded = false; // false = only the first POLICY_VISIBLE_COUNT rows are shown;
                                 // toggled by the "Show all" button, survives re-renders (drag
                                 // reorder, status pushes) until the user collapses it again
let metricListExpanded = false; // same idea as policyListExpanded, but for the Result panel's
                                 // metric list — survives stepping between policies with ↑/↓
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
  // ?darkMode forces viser's own color scheme client-side. Its Python-side
  // equivalent (ViserServer.gui.configure_theme(dark_mode=True)) is also set
  // server-side (see viser_viewer.py) but is a persisted *message* the client
  // applies asynchronously after connecting — this URL flag is read at first
  // paint, so the panel never flashes light before flipping dark.
  iframe.src = `http://localhost:${viserPort}/?darkMode`;
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
const chkEpisodeTimeout = $('#chk-episode-timeout');
const episodeTimeoutRow = $('#episode-timeout-row');
const episodeTimeoutS = $('#episode-timeout-s');
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
  // When --token is set on the server (see swap_experiment.py --real / -h),
  // it's never served up automatically — the operator shares a link like
  // http://<host>:<port>/?token=... out of band (same mechanism a
  // home-made controller uses, see docs/index.html §13), and this page
  // just forwards its own query string onto the socket.
  const token = new URLSearchParams(location.search).get('token');
  const url = `ws://${location.host}/ws${token ? `?token=${encodeURIComponent(token)}` : ''}`;
  ws = new WebSocket(url);
  ws.onopen = () => {
    footer.textContent = 'connected'; connDot.className = 'ok';
    familySwitchInFlight = false;
    refreshTrainingCatalog();
    refreshSystemInfo();
    refreshFamilyList();
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

// Collapsed by default to match the shortcut cap above — the first 10 slots
// are the only ones with a keycap anyway, so anything past that is already
// "extra" and belongs behind "Show all" rather than pushing the Policies
// panel taller than its neighbors (Live Telemetry etc).
const POLICY_VISIBLE_COUNT = 10;

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
  const visibleCount = policyListExpanded ? names.length : POLICY_VISIBLE_COUNT;
  names.slice(0, visibleCount).forEach((name) => {
    policyList.appendChild(policyButtonRow(name, active));
  });

  const hiddenCount = names.length - POLICY_VISIBLE_COUNT;
  if (hiddenCount > 0) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'policy-show-all-btn';
    toggle.textContent = policyListExpanded ? 'Show fewer' : `Show all (${names.length})`;
    toggle.onclick = () => {
      policyListExpanded = !policyListExpanded;
      renderPolicyList(names, latestStatus?.active);
    };
    policyList.appendChild(toggle);
  }

  initPolicyDrag();
}

const familyList = $('#family-list');
let familySwitchInFlight = false;
let familyListExpanded = false; // "Show all" toggle -- see renderFamilyList()

// A family switch kills THIS process and relaunches a new one for the
// chosen task (see ControlService.switch_family()'s docstring — Genesis
// can't rebuild its scene in-process). Row click uses send() (fire-and-
// forget), never call(): the switch itself drops the socket, so a
// call() awaiting a reply would hang forever (pendingCalls aren't
// cleaned up on close).
function familyButtonRow(name, current, hasPolicies) {
  const row = document.createElement('div');
  row.className = 'policy-row';
  const btn = document.createElement('button');
  btn.className = 'policy-btn' + (name === current ? ' active' : '');
  btn.textContent = name;
  btn.disabled = name === current || familySwitchInFlight || !hasPolicies;
  if (!hasPolicies) btn.title = `No trained policies for '${name}' yet — train one first`;
  btn.onclick = () => {
    familySwitchInFlight = true;
    footer.textContent = `switching to '${name}' — this page will reconnect on its own (10-20s)…`;
    send('switch_family', { task: name });
  };
  row.appendChild(btn);
  return row;
}

function renderFamilyList(families) {
  if (!families) return;
  familyList.innerHTML = '';
  // Collapsed (default): only tasks with at least one trained local policy
  // -- switch_family() itself rejects the rest server-side, and this repo
  // registers ~20 tasks across several unrelated robot bodies (G1, Go2, K1,
  // Tron1), so showing the full list by default is mostly noise, not
  // options. "Show all" reveals every registered task anyway (disabled/
  // greyed for the ones with no policy yet) -- useful for seeing what's
  // available to train next, per the Create Policy panel's own task list.
  const withPolicies = families.tasks.filter((name) =>
    name === families.current || (families.policies_per_task[name] || []).length > 0);
  const names = familyListExpanded ? families.tasks : withPolicies;
  names.forEach((name) => {
    const hasPolicies = name === families.current || (families.policies_per_task[name] || []).length > 0;
    familyList.appendChild(familyButtonRow(name, families.current, hasPolicies));
  });

  const hiddenCount = families.tasks.length - withPolicies.length;
  if (hiddenCount > 0) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'policy-show-all-btn';
    toggle.textContent = familyListExpanded ? 'Show fewer' : `Show all (${families.tasks.length})`;
    toggle.onclick = () => {
      familyListExpanded = !familyListExpanded;
      renderFamilyList(families);
    };
    familyList.appendChild(toggle);
  }
}

function refreshFamilyList() {
  call('list_families').then(renderFamilyList).catch((e) => {
    console.warn('list_families unavailable:', e.message);
  });
}

function onPolicyReorder() {
  // While collapsed, only the visible rows exist in the DOM to drag — any
  // names hidden behind "Show all" aren't part of this drag at all, so
  // splice them back in (in their prior relative order) rather than let
  // them silently fall out of the saved order.
  const visibleNames = [...policyList.querySelectorAll('.policy-row')].map((r) => r.dataset.policy);
  const rawNames = latestStatus?.policies || [];
  const visibleSet = new Set(visibleNames);
  const hiddenNames = applyPolicyOrder(rawNames).filter((n) => !visibleSet.has(n));
  const names = [...visibleNames, ...hiddenNames];
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
  // Shows the current FAMILY (task), not the active policy -- which
  // family/experiment is running is the more useful at-a-glance fact here;
  // the active policy within it is already highlighted in the Policies
  // list below (see policyButtonRow's `.active` class).
  let text = `${color} ${status.current_task ?? '—'}`;
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

  // episode_timeout_s is absent on adapters that don't support it (e.g. a
  // not-yet-built RealAdapter path) — hide the whole control rather than
  // showing a toggle that would just NotImplementedError on click.
  const timeoutSupported = 'episode_timeout_s' in status;
  chkEpisodeTimeout.closest('.toggle-row').style.display = timeoutSupported ? '' : 'none';
  if (timeoutSupported) {
    const enabled = status.episode_timeout_s != null;
    chkEpisodeTimeout.checked = enabled;
    episodeTimeoutRow.hidden = !enabled;
    if (enabled && document.activeElement !== episodeTimeoutS) {
      episodeTimeoutS.value = status.episode_timeout_s;
    }
  }

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

function sendEpisodeTimeout() {
  const seconds = Math.max(1, Number(episodeTimeoutS.value) || 1);
  send('set_episode_timeout', { seconds: chkEpisodeTimeout.checked ? seconds : null });
}
chkEpisodeTimeout.addEventListener('change', () => {
  episodeTimeoutRow.hidden = !chkEpisodeTimeout.checked;
  sendEpisodeTimeout();
});
episodeTimeoutS.addEventListener('change', () => {
  if (chkEpisodeTimeout.checked) sendEpisodeTimeout();
});

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

// ---- Hardware drawer ----
// The user asked for this explicitly: don't make them guess what hardware
// is running the sim/training — show it, and use it as the basis for the
// Create Policy panel's num_envs suggestion + time estimate below. Used to
// be a top-of-page bar showing only THIS machine; once training could also
// run on Kaggle, "the machine" stopped being a single answer, so this is
// now a full drawer (like Docs) showing both side by side, plus why
// num_envs/the time estimate differ between them (see
// TrainingManager.estimate()'s backend param — local and Kaggle are pooled
// from separate history, never mixed).

const hardwarePanel = $('#hardware-panel');
let systemInfo = null;

// info.kaggle_profile entries are undefined until this session's server has
// actually restarted with them (Python doesn't hot-reload — see
// legged_gym/control/*.py edits) — fall back to an explicit "–" instead of
// silently rendering "undefined"/"NaN", which is confusing to tell apart
// from a real value.
function fmtOrDash(v, suffix = '') {
  return (v === undefined || v === null) ? '–' : `${v}${suffix}`;
}
function renderHardwarePanel(info) {
  if (!hardwarePanel) return;
  const kp = info.kaggle_profile || {};
  hardwarePanel.innerHTML = '';

  const local = document.createElement('div');
  local.className = 'hw-section';
  local.innerHTML = `<h2>This machine <span class="hw-badge">live, measured</span></h2>`;
  const localDl = document.createElement('dl');
  const localRows = [
    ['OS', info.os], ['Machine', info.machine], ['CPU', info.cpu_brand],
    ['Cores', info.cpu_count], ['RAM', `${info.ram_gb ?? '?'} GB`],
    ['GPU', info.cuda_available ? 'CUDA available (unused — local training runs CPU)'
      : info.mps_available ? 'Metal available (unused — local training runs CPU)' : 'none'],
    ['Simulator', `${info.simulator} (${info.genesis_backend})`],
    ['Control backend', info.control_backend],
    ['Suggested parallel envs', `${info.suggested_num_envs.comfortable}–${info.suggested_num_envs.upper}`],
  ];
  for (const [k, v] of localRows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    localDl.appendChild(dt); localDl.appendChild(dd);
  }
  local.appendChild(localDl);
  const localP = document.createElement('p');
  localP.textContent = 'Every environment runs vectorized but not free — more parallel envs than a few per '
    + 'CPU core stops scaling training speed. The suggested range above comes from core count; the time '
    + 'estimate in Create Policy switches to real measured numbers once at least one local job has finished.';
  local.appendChild(localP);
  hardwarePanel.appendChild(local);

  const kaggle = document.createElement('div');
  kaggle.className = 'hw-section';
  kaggle.innerHTML = info.kaggle_available
    ? `<h2>Kaggle (free GPU tier) <span class="hw-badge">measured once, not live</span></h2>`
    : `<h2>Kaggle (free GPU tier) <span class="hw-badge">not configured</span></h2>`;
  if (info.kaggle_available && Object.keys(kp).length === 0) {
    // kaggle_profile is missing entirely, not just partially — the server
    // process predates it (Python doesn't hot-reload code changes) rather
    // than Kaggle itself withholding data. A dash-filled table here would
    // look like a real "unknown" answer instead of a stale-server symptom.
    const p = document.createElement('p');
    p.textContent = 'This server needs a restart to show Kaggle\'s profile (its running process predates '
      + 'this panel\'s data).';
    kaggle.appendChild(p);
  } else if (info.kaggle_available) {
    const kaggleDl = document.createElement('dl');
    const kaggleRows = [
      ['GPU', fmtOrDash(kp.gpu)], ['Compute capability', fmtOrDash(kp.compute_capability)],
      ['VRAM', fmtOrDash(kp.vram_gb, ' GB')],
      ['CPU cores', fmtOrDash(kp.cpu_cores)], ['RAM', fmtOrDash(kp.ram_gb, ' GB')],
      ['Simulator', kp.simulator ? `${kp.simulator} (not Genesis — see Docs)` : '–'],
      ['Per-job bootstrap', kp.bootstrap_overhead_s != null
        ? `~${Math.round(kp.bootstrap_overhead_s / 60)} min (fresh venv + Isaac Gym install, every job)` : '–'],
      ['Session cap', kp.session_cap_hours != null
        ? `${kp.session_cap_hours}h (Kaggle's own GPU session limit)` : '–'],
      ['Suggested parallel envs', kp.suggested_num_envs
        ? `${kp.suggested_num_envs.comfortable}–${kp.suggested_num_envs.upper}` : '–'],
    ];
    for (const [k, v] of kaggleRows) {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      kaggleDl.appendChild(dt); kaggleDl.appendChild(dd);
    }
    kaggle.appendChild(kaggleDl);
    const kaggleP = document.createElement('p');
    kaggleP.textContent = 'Not a live probe (there\'s no Kaggle session to query without spending a kernel), '
      + 'but not a guess either — measured directly with a dedicated diagnostic kernel (nvidia-smi, '
      + 'torch.cuda.get_device_properties, /proc/meminfo — see HANDOFF_kaggle_cloud_gpu.md). Kaggle assigns '
      + 'this per session on its free tier, so a future session could get something different. Genesis\'s GPU '
      + 'backend needs Volta+ hardware this tier doesn\'t have, so Kaggle jobs run Isaac Gym instead — a '
      + 'different simulator than every local job (see Docs → Simulators). Its own time estimate in Create '
      + 'Policy is measured separately from local runs, and includes the per-job bootstrap above, which '
      + 'dominates a short job\'s wall-clock time in a way local runs never account for. The suggested '
      + 'parallel-envs range is real too, not a guess — a diagnostic kernel confirmed g1 creates cleanly at '
      + '4096 envs on this exact P100 (no OOM), matching typical community guidance for a 16GB GPU on a '
      + 'non-vision humanoid task.';
    kaggle.appendChild(kaggleP);
  } else {
    const p = document.createElement('p');
    p.textContent = 'No ~/.kaggle/kaggle.json found on the server — the Kaggle backend isn\'t set up on this machine.';
    kaggle.appendChild(p);
  }
  hardwarePanel.appendChild(kaggle);

  const choose = document.createElement('div');
  choose.innerHTML = '<h2>Choosing where to train</h2>';
  const chooseP = document.createElement('p');
  chooseP.textContent = 'Create Policy\'s "Run on" toggle picks the backend per job — the num_envs suggestion '
    + 'and time estimate shown there always reflect whichever one is selected, using that backend\'s own '
    + 'history, never the other one\'s.';
  choose.appendChild(chooseP);
  hardwarePanel.appendChild(choose);
}

// The suggestion (both the number and the hint text) reflects whichever
// backend is currently selected — local's is CPU-core-based, Kaggle's comes
// from a real diagnostic kernel (see kaggle_profile's own comment in
// training.py) — never a blend of the two.
function updateEnvsHint() {
  if (!systemInfo) return;
  if (trainBackend === 'kaggle') {
    const kp = systemInfo.kaggle_profile;
    if (!kp || !kp.suggested_num_envs) {
      trainEnvsHint.textContent = 'Suggested range unavailable — this server needs a restart to pick up Kaggle\'s profile.';
      return;
    }
    if (!envsFieldTouched) trainEnvs.value = kp.suggested_num_envs.comfortable;
    trainEnvsHint.textContent =
      `Suggested for Kaggle's GPU: ${kp.suggested_num_envs.comfortable}–${kp.suggested_num_envs.upper} ` +
      `— confirmed g1 creates cleanly up to 4096 envs on that hardware (see Hardware tab).`;
  } else {
    const s = systemInfo.suggested_num_envs;
    if (!envsFieldTouched) trainEnvs.value = s.comfortable;
    trainEnvsHint.textContent =
      `Suggested for this machine: ${s.comfortable}–${s.upper} ` +
      `(${systemInfo.cpu_count} cores detected) — more scales training speed less on CPU past that.`;
  }
}

function refreshSystemInfo() {
  call('system_info').then((info) => {
    systemInfo = info;
    renderHardwarePanel(info);
    updateEnvsHint();
    updateEstimate();

    // Only offer the toggle when ~/.kaggle/kaggle.json exists server-side
    // (see TrainingManager.system_info()'s kaggle_available) — otherwise
    // every Kaggle job would just fail immediately with a credentials error.
    trainBackendRow.hidden = !info.kaggle_available;
    if (!info.kaggle_available && trainBackend === 'kaggle') {
      setTrainBackend('local');
      updateCommandPreview();
    }
  }).catch((e) => {
    if (hardwarePanel) hardwarePanel.textContent = 'System info unavailable.';
    console.warn('system_info unavailable:', e.message);
  });
}

// ---- create policy (training) ----
// The whole point of this panel, per the user ask: never hide the actual
// command behind the form — #train-cmd-preview always shows the rugiar CLI
// invocation (legged_gym/cli/rugiar.py) that reproduces what this Start
// button is about to do via ControlService.start_training()/
// TrainingManager.start(), so it's copy-pasteable into a real terminal.

const btnNewPolicy = $('#btn-new-policy');
const createPolicyForm = $('#create-policy-form');
const trainBackendRow = $('#train-backend-row');
const trainBackendTabs = $('#train-backend');
const trainBackendHint = $('#train-backend-hint');
const trainName = $('#train-name');
const trainBase = $('#train-base');
const trainBaseSimWarning = $('#train-base-sim-warning');
const trainTask = $('#train-task');
const trainIters = $('#train-iters');
const trainMinutes = $('#train-minutes');
const trainEnvs = $('#train-envs');
const trainEnvsHint = $('#train-envs-hint');
const trainVxLo = $('#train-vx-lo'), trainVxHi = $('#train-vx-hi');
const trainVyLo = $('#train-vy-lo'), trainVyHi = $('#train-vy-hi');
const trainYawLo = $('#train-yaw-lo'), trainYawHi = $('#train-yaw-hi');
// The 4 always-visible target-variable fields (see TARGET_VAR_KEYS below) —
// one {abs, pct, ref, note, warn} element set per registered variable, keyed
// the same way trainingCatalog.variables/task_defaults().variables are.
const TARGET_VAR_KEYS = ['base_height', 'lin_vel_z', 'ang_vel_xy', 'orientation_tilt'];
const targetVarEls = Object.fromEntries(TARGET_VAR_KEYS.map((key) => [key, {
  abs: $(`#target-abs-${key}`),
  pct: $(`#target-pct-${key}`),
  ref: $(`#target-ref-${key}`),
  note: $(`#target-note-${key}`),
  warn: $(`#target-warn-${key}`),
}]));
const trainPush = $('#train-push');
const trainPushVel = $('#train-push-vel');
const trainPushInterval = $('#train-push-interval');
const trainPushDir = $('#train-push-dir');
const trainEntropyCoef = $('#train-entropy-coef');
const trainRewardScales = $('#train-reward-scales');
const trainCmdPreview = $('#train-cmd-preview');
const trainCmdCopy = $('#train-cmd-copy');
trainCmdCopy.addEventListener('click', () => copyToClipboard(trainCmdPreview.textContent, trainCmdCopy));
const trainEstimate = $('#train-estimate');
const trainError = $('#train-error');
const btnStartTraining = $('#btn-start-training');
const trainingJobsEl = $('#policy-training-jobs');

// ---- fuse policies (see legged_gym/control/fusion.py + rugiar fuse) ----
const btnFusePolicy = $('#btn-fuse-policy');
const fusePolicyForm = $('#fuse-policy-form');
const fuseSourcesEl = $('#fuse-sources');
const fuseMethod = $('#fuse-method');
const fuseMethodDescription = $('#fuse-method-description');
const fuseName = $('#fuse-name');
const fuseTaskWarning = $('#fuse-task-warning');
const fuseTaskRow = $('#fuse-task-row');
const fuseExportTask = $('#fuse-export-task');
const fuseCmdPreview = $('#fuse-cmd-preview');
const fuseCmdCopy = $('#fuse-cmd-copy');
fuseCmdCopy.addEventListener('click', () => copyToClipboard(fuseCmdPreview.textContent, fuseCmdCopy));
const fuseWarningsEl = $('#fuse-warnings');
const fuseErrorEl = $('#fuse-error');
const btnStartFusion = $('#btn-start-fusion');

function showFuseError(msg) {
  fuseErrorEl.textContent = msg;
  fuseErrorEl.classList.toggle('show', !!msg);
}

function showFuseWarnings(warnings) {
  fuseWarningsEl.textContent = (warnings || []).map((w) => `⚠ ${w}`).join('\n');
  fuseWarningsEl.classList.toggle('show', !!(warnings || []).length);
}

// One checkbox + weight input per fusable local policy (train_checkpoint.pt
// present — same requirement Clone-from has). Checked state/weights survive
// a catalog refresh (e.g. right after this very panel fuses something) by
// name, same spirit as refreshTrainingCatalog()'s prevTask/prevBase.
function refreshFuseSources() {
  const prev = new Map(fuseCheckedSources().map((s) => [s.name, s.weight]));
  fuseSourcesEl.innerHTML = '';
  for (const p of trainingCatalog?.fusable_policies || []) {
    const row = document.createElement('div');
    row.className = 'fuse-source-row';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = p.name;
    checkbox.className = 'fuse-source-checkbox';
    checkbox.checked = prev.has(p.name);

    const label = document.createElement('label');
    label.textContent = p.name;

    const task = document.createElement('span');
    task.className = 'fuse-source-task';
    task.textContent = p.task;

    const weight = document.createElement('input');
    weight.type = 'number';
    weight.min = '0'; weight.step = 'any';
    weight.className = 'fuse-source-weight';
    weight.value = prev.get(p.name) ?? '1';
    weight.disabled = !checkbox.checked;

    checkbox.addEventListener('change', () => {
      weight.disabled = !checkbox.checked;
      refreshFuseTaskDisambiguation();
      updateFuseCmdPreview();
    });
    weight.addEventListener('input', updateFuseCmdPreview);

    row.append(checkbox, label, task, weight);
    fuseSourcesEl.appendChild(row);
  }
}

function fuseCheckedSources() {
  return Array.from(fuseSourcesEl.querySelectorAll('.fuse-source-row')).flatMap((row) => {
    const checkbox = row.querySelector('.fuse-source-checkbox');
    if (!checkbox.checked) return [];
    const name = checkbox.value;
    const task = trainingCatalog?.fusable_policies?.find((p) => p.name === name)?.task;
    const weight = parseFloat(row.querySelector('.fuse-source-weight').value);
    return [{ name, task, weight }];
  });
}

function refreshFuseMethods() {
  const prev = fuseMethod.value;
  fuseMethod.innerHTML = '';
  for (const m of trainingCatalog?.fusion_methods || []) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.available ? m.label : `${m.label} (coming soon)`;
    opt.disabled = !m.available;
    opt.title = m.description;
    fuseMethod.appendChild(opt);
  }
  if ([...fuseMethod.options].some((o) => o.value === prev && !o.disabled)) {
    fuseMethod.value = prev;
  } else {
    const firstAvailable = [...fuseMethod.options].find((o) => !o.disabled);
    if (firstAvailable) fuseMethod.value = firstAvailable.value;
  }
  updateFuseMethodDescription();
}

function updateFuseMethodDescription() {
  const info = trainingCatalog?.fusion_methods?.find((m) => m.id === fuseMethod.value);
  fuseMethodDescription.textContent = info?.description || '';
}

// export_task only needs deciding when the checked sources disagree on
// task — see TrainingManager.fuse_policies()'s docstring for why a
// mismatched task is a warning, not a hard stop, and why THIS field still
// has to be a real, obs/action-compatible task regardless.
function refreshFuseTaskDisambiguation() {
  const sources = fuseCheckedSources();
  const tasks = [...new Set(sources.map((s) => s.task).filter(Boolean))];
  if (tasks.length <= 1) {
    fuseTaskWarning.hidden = true;
    fuseTaskRow.hidden = true;
    updateFuseCmdPreview();
    return;
  }
  fuseTaskWarning.hidden = false;
  fuseTaskWarning.textContent =
    `Checked sources span different tasks (${tasks.join(', ')}) — pick which one the fused policy is registered under.`;
  fuseTaskRow.hidden = false;
  const prev = fuseExportTask.value;
  fuseExportTask.innerHTML = '';
  for (const t of tasks) {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    fuseExportTask.appendChild(opt);
  }
  if (tasks.includes(prev)) fuseExportTask.value = prev;
  updateFuseCmdPreview();
}

function updateFuseCmdPreview() {
  const sources = fuseCheckedSources();
  const parts = ['rugiar fuse'];
  if (sources.length) parts.push(`--policies ${sources.map((s) => s.name).join(' ')}`);
  if (fuseName.value.trim()) parts.push(`--name ${fuseName.value.trim()}`);
  if (sources.some((s) => s.weight !== 1 && Number.isFinite(s.weight))) {
    parts.push(`--weights ${sources.map((s) => (Number.isFinite(s.weight) ? s.weight : '?')).join(' ')}`);
  }
  if (fuseMethod.value && fuseMethod.value !== 'weighted_average') parts.push(`--method ${fuseMethod.value}`);
  if (!fuseTaskRow.hidden && fuseExportTask.value) parts.push(`--export_task ${fuseExportTask.value}`);
  fuseCmdPreview.textContent = parts.join(' ');
}

fuseName.addEventListener('input', updateFuseCmdPreview);
fuseMethod.addEventListener('change', () => { updateFuseMethodDescription(); updateFuseCmdPreview(); });
fuseExportTask.addEventListener('change', updateFuseCmdPreview);

btnFusePolicy.addEventListener('click', () => {
  const opening = fusePolicyForm.hidden;
  fusePolicyForm.hidden = !opening;
  btnFusePolicy.textContent = opening ? 'Cancel' : '⚛ Fuse policies…';
  if (opening) {
    showFuseError('');
    showFuseWarnings([]);
    refreshFuseSources();
    refreshFuseMethods();
    refreshFuseTaskDisambiguation();
    updateFuseCmdPreview();
  }
});

fusePolicyForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const sources = fuseCheckedSources();
  const name = fuseName.value.trim();

  if (sources.length < 2) return showFuseError('Check at least 2 policies to fuse.');
  if (sources.some((s) => !Number.isFinite(s.weight) || s.weight < 0)) {
    return showFuseError('Every checked policy needs a weight ≥ 0.');
  }
  if (sources.every((s) => s.weight === 0)) return showFuseError('At least one weight must be greater than 0.');
  if (!name) return showFuseError('Output policy name is required.');
  // Same "no silent overwrite" guard createPolicyForm enforces (see its own
  // submit handler above) — finalize-style writers refuse to touch an
  // existing policies/<name>/ rather than overwrite it.
  if ((latestStatus?.policies || []).includes(name)) {
    return showFuseError(
      `'${name}' already exists — fusing into this name would overwrite its checkpoint. Pick a different name (e.g. '${name}_2').`);
  }
  if (!fuseTaskRow.hidden && !fuseExportTask.value) return showFuseError('Pick which task to register the fused policy under.');

  showFuseError('');
  showFuseWarnings([]);
  btnStartFusion.disabled = true;
  call('fuse_policies', {
    names: sources.map((s) => s.name),
    out_name: name,
    weights: sources.map((s) => s.weight),
    method: fuseMethod.value,
    export_task: !fuseTaskRow.hidden ? fuseExportTask.value : null,
  }).then((result) => {
    fusePolicyForm.reset();
    // The new policy needs to show up in the Clone-from/Fuse-from lists right
    // away, not just the Policies list (which status() already refreshes on
    // its own) — same refresh a just-finished training job triggers (see
    // poll()'s justFinished branch below).
    refreshTrainingCatalog();
    // Fusion succeeded either way — a warning (e.g. "sources trained on
    // different simulators") is informational, not a reason to keep the form
    // open. fuse_policies() already baked it into the new policy's
    // meta.json (see TrainingManager.fuse_policies()), so it's not lost —
    // it shows up under that policy's own info popup (renderPolicyInfo()'s
    // "Fusion" section) instead of flashing here and disappearing.
    fusePolicyForm.hidden = true;
    btnFusePolicy.textContent = '⚛ Fuse policies…';
  }).catch((e) => {
    showFuseError(e.message);
  }).finally(() => {
    btnStartFusion.disabled = false;
  });
});

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
    const where = trainBackend === 'kaggle' ? 'on Kaggle' : 'on this machine';
    call('estimate_training_time', {
      num_envs: numEnvs,
      max_iterations: hasIters ? iterations : null,
      max_minutes: hasMinutes ? minutes : null,
      backend: trainBackend,
    }).then((est) => {
      if (est.basis !== 'measured') {
        trainEstimate.textContent = hasMinutes && !hasIters
          ? `Estimated time: time-boxed to ${minutes} minute${minutes === 1 ? '' : 's'} — no runs ${where} yet, so an iteration count can't be estimated (the first run calibrates this).`
          : `Estimated time: no runs ${where} yet — the first one calibrates this estimate.`;
        return;
      }
      const boundBy = hasIters && hasMinutes
        ? (est.iterations < iterations ? 'minutes' : 'iterations')
        : hasMinutes ? 'minutes' : 'iterations';
      // Kaggle's ~3-4min per-job bootstrap (fresh venv + Isaac Gym install,
      // every job — see kaggle_backend.py) is baked into whatever history
      // exists, so it's already reflected in est.seconds; called out
      // separately here because it dominates a SHORT job's wall-clock time
      // in a way local runs never have to account for.
      const bootstrapNote = trainBackend === 'kaggle'
        ? ' (includes Kaggle’s ~3–4 min per-job environment bootstrap, baked into the history this is based on)' : '';
      trainEstimate.textContent = boundBy === 'minutes'
        ? `Estimated time: ~${est.iterations} iterations in ${minutes} minute${minutes === 1 ? '' : 's'} (approximate — based on ${est.samples} previous run${est.samples === 1 ? '' : 's'} ${where}; the wall-clock deadline is checked every ${TIME_BUDGET_CHUNK_ITERS} iterations, so the run may overshoot by a few iterations' worth of time)${bootstrapNote}.`
        : `Estimated time: ~${formatDuration(est.seconds)} for ${est.iterations} iterations (based on ${est.samples} previous run${est.samples === 1 ? '' : 's'} ${where})${bootstrapNote}${hasMinutes ? ` — also capped at ${minutes} minute${minutes === 1 ? '' : 's'}, whichever hits first` : ''}.`;
    }).catch(() => { trainEstimate.textContent = 'Estimated time: –'; });
  }, 250);
}

function showTrainError(msg) {
  trainError.textContent = msg;
  trainError.classList.toggle('show', !!msg);
}

// ---- target variables: 4 always-visible fields, each independently ----
// ---- optional (Absolute value, or a signed % change off a reference) ----
// Populated from TrainingManager.VARIABLE_REGISTRY (legged_gym/control/
// training.py) via training_catalog()'s 'variables' (task-independent:
// label/unit/source/flag/note) and task_defaults()'s 'variables'
// (task-dependent: reference, refetched on every task/clone-from change).
// See resolveTarget()/refreshTargetReferences() below.
let trainBackend = 'local'; // 'local' | 'kaggle' — see #train-backend's click handler below
let targetReference = null; // {value: number|null, label: string} | null
let targetRange = null;     // [min, max] | null

// Shared by the tab click handler, restoreTrainFormConfig() (remembering the
// last-used backend across panel opens — see TRAIN_FORM_STORAGE_KEY), and
// refreshSystemInfo()'s kaggle_available correction, so all three apply the
// exact same backend-switch side effects instead of three copies drifting
// apart.
function setTrainBackend(value) {
  trainBackend = value;
  trainBackendTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.backend === trainBackend));
  trainBackendHint.textContent = trainBackend === 'kaggle'
    ? 'Runs on a free Kaggle GPU kernel instead of this machine’s CPU — much faster, and there’s no live iteration count while it runs. Clone-from uploads the base checkpoint to Kaggle automatically (as a private Dataset) before the kernel starts.'
    : '';
  refreshCloneFromMismatchWarning();
}

// Kaggle jobs train under Isaac Gym, local jobs under Genesis (see
// TrainingJob.simulator's docstring in training.py) — different contact/PD
// dynamics, so fine-tuning a base trained under ONE engine on the OTHER is a
// real sim2sim transfer, not a guaranteed-compatible continuation. Not
// blocked server-side (see TrainingManager.start()'s own comment) — flagged
// here instead, so an informed choice still goes through.
function targetSimulator() {
  return trainBackend === 'kaggle' ? 'isaacgym' : 'genesis';
}

function refreshCloneFromMismatchWarning() {
  const base = trainingCatalog?.base_policies?.find((p) => p.name === trainBase.value);
  const baseSimulator = base?.simulator || 'genesis';
  const target = targetSimulator();
  if (base && baseSimulator !== target) {
    trainBaseSimWarning.textContent =
      `'${base.name}' was trained under ${baseSimulator === 'isaacgym' ? 'Isaac Gym' : 'Genesis'}, but this run ` +
      `will train under ${target === 'isaacgym' ? 'Isaac Gym (Kaggle)' : 'Genesis (this machine)'} — different ` +
      `contact/PD dynamics, so this fine-tune is a real sim2sim transfer, not a guaranteed-compatible continuation.`;
    trainBaseSimWarning.hidden = false;
  } else {
    trainBaseSimWarning.hidden = true;
  }
}

// targetReferences[key] = {value: number|null, label: string} | null — one
// per variable, all 4 resolved together off a single task_defaults() call
// (it already returns every registered variable's reference at once).
let targetReferences = {};

function renderVariableChrome() {
  for (const key of TARGET_VAR_KEYS) {
    const meta = trainingCatalog?.variables?.[key];
    const els = targetVarEls[key];
    if (meta) {
      els.note.textContent = `${meta.source === 'sensor' ? 'Real sensor' : 'Simulator ground truth'} — ${meta.note || ''}`;
    }
  }
}

function renderTargetReferences() {
  for (const key of TARGET_VAR_KEYS) {
    const els = targetVarEls[key];
    const unit = trainingCatalog?.variables?.[key]?.unit || '';
    const ref = targetReferences[key];
    if (!ref || ref.value == null) {
      els.ref.textContent = 'Reference: – (pick a task or clone-from base first)';
      els.warn.hidden = true;
      continue;
    }
    els.ref.textContent = `Reference: ${ref.value.toFixed(3)} ${unit} (${ref.label})`;
    // A 0 reference makes '% change' a no-op (any percent of 0 is still 0)
    // — the 3 non-height variables default to exactly 0 unless a clone-from
    // base already set a nonzero target, so this is a real footgun worth
    // flagging inline rather than letting someone type "+20%" and get
    // silently nothing.
    els.warn.hidden = ref.value !== 0;
    els.warn.textContent = 'Reference is 0 — % change has no effect here; use Absolute instead.';
  }
  updateCommandPreview();
}

function refreshTargetReferences() {
  const base = trainBase.value, task = trainTask.value;
  if (!task && !base) { targetReferences = {}; renderTargetReferences(); return; }
  // A clone-from base's reference is always its OWN task's default, not the
  // (possibly different) currently-selected target task — a '% change'
  // delta should be relative to what the base was actually trained at.
  // Same rule for every registered variable, base_height included.
  const referenceTask = base
    ? trainingCatalog?.base_policies.find((b) => b.name === base)?.task
    : task;
  if (!referenceTask) { renderTargetReferences(); return; }
  call('task_defaults', { task: referenceTask }).then((d) => {
    targetReferences = {};
    for (const key of TARGET_VAR_KEYS) {
      const v = d.variables?.[key];
      targetReferences[key] = {
        value: v?.reference ?? null,
        label: base ? `${base}'s task default` : `${referenceTask}'s default`,
      };
    }
    renderTargetReferences();
  }).catch(() => { targetReferences = {}; renderTargetReferences(); });
}

// Resolves one variable's target from its Absolute/% change pair:
//   both blank -> null (unset, task default)
//   abs filled, pct blank -> the absolute number
//   pct filled, abs blank -> reference * (1 + pct/100), or undefined if no
//     reference has resolved yet (submit-time validation error)
//   both filled -> NaN (sentinel: ambiguous, submit-time validation error)
function resolveTarget(key) {
  const els = targetVarEls[key];
  const absRaw = els.abs.value.trim();
  const pctRaw = els.pct.value.trim();
  if (absRaw !== '' && pctRaw !== '') return NaN;
  if (absRaw !== '') return parseFloat(absRaw);
  if (pctRaw === '') return null;
  const ref = targetReferences[key];
  if (!ref || ref.value == null) return undefined;
  return ref.value * (1 + parseFloat(pctRaw) / 100);
}

function resolveAllTargets() {
  return Object.fromEntries(TARGET_VAR_KEYS.map((key) => [key, resolveTarget(key)]));
}

trainBackendTabs.querySelectorAll('button').forEach((btn) => {
  btn.addEventListener('click', () => {
    setTrainBackend(btn.dataset.backend);
    updateCommandPreview();
    updateEnvsHint(); // different backend = different suggested range, see updateEnvsHint()
    updateEstimate(); // different backend = different history bucket, see estimate()
  });
});

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
    label.textContent = term;
    label.htmlFor = `train-reward-${term}`;
    // Same didactic hover/focus tooltip used for the merged Result list
    // (REWARD_TERM_GLOSSARY, renderMetricList()) — falls back to
    // the backend's REWARD_SCALE_NOTES for weight-only terms the glossary
    // doesn't cover (e.g. crouch_depth), so every field gets an
    // explanation rather than just the ones with a backend note.
    const tip = REWARD_TERM_GLOSSARY[term] || notes[term];
    if (tip) {
      label.classList.add('info-term-name');
      label.tabIndex = 0;
      label.setAttribute('data-tip', tip);
    }
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
// incremented in place instead of appended again). Actually WRITTEN into
// the field (not just shown as a placeholder) as long as the user hasn't
// touched it yet — see nameFieldTouched below — specifically so reusing an
// existing policy's name by accident (silently overwriting its checkpoint —
// see finalize_policy()'s plain shutil.copyfile) takes a deliberate edit,
// not just forgetting to type a name. Typing anything yourself always wins
// from then on (composeTrainingParams() only falls back to the suggestion
// if the field is left blank).
const DEFAULT_NAME_PLACEHOLDER = 'e.g. cautious_v2';

// Mirrors envsFieldTouched's pattern: once the user edits the Name field by
// hand, auto-naming stops overwriting it — reset whenever the panel opens
// fresh (see btnNewPolicy's click handler).
let nameFieldTouched = false;
trainName.addEventListener('input', () => { nameFieldTouched = true; });

function suggestedPolicyName() {
  const base = trainBase.value;
  const known = new Set([
    ...(trainingCatalog?.base_policies || []).map((p) => p.name),
    ...(latestStatus?.policies || []),
  ]);
  if (!base) {
    // Nothing to fine-tune from, but still avoid handing back a name that's
    // already taken — increment DEFAULT_NAME_PLACEHOLDER's own stem rather
    // than leaving a from-scratch run to collide with an existing policy.
    if (!known.has('new_policy')) return null; // keep the friendlier placeholder text when there's no collision risk
    let n = 2;
    while (known.has(`new_policy_${n}`)) n += 1;
    return `new_policy_${n}`;
  }
  const m = base.match(/^(.*?)(\d+)$/);
  const prefix = m ? m[1] : `${base}_`;
  let n = m ? parseInt(m[2], 10) + 1 : 2;
  let candidate = `${prefix}${n}`;
  while (known.has(candidate)) { n += 1; candidate = `${prefix}${n}`; }
  return candidate;
}

function refreshNamePlaceholder() {
  const suggestion = suggestedPolicyName();
  trainName.placeholder = suggestion || DEFAULT_NAME_PLACEHOLDER;
  if (!nameFieldTouched && suggestion) trainName.value = suggestion;
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
    task: trainTask.value, base: trainBase.value, backend: trainBackend,
    numEnvs: trainEnvs.value, iters: trainIters.value, minutes: trainMinutes.value,
    vxLo: trainVxLo.value, vxHi: trainVxHi.value,
    vyLo: trainVyLo.value, vyHi: trainVyHi.value,
    yawLo: trainYawLo.value, yawHi: trainYawHi.value,
    targets: Object.fromEntries(TARGET_VAR_KEYS.map((key) => [key, {
      abs: targetVarEls[key].abs.value, pct: targetVarEls[key].pct.value,
    }])),
    push: trainPush.value, pushVel: trainPushVel.value, pushInterval: trainPushInterval.value, pushDir: trainPushDir.value,
    entropyCoef: trainEntropyCoef.value,
    rewardScales,
  };
}

function saveTrainFormConfig(overrides = {}) {
  try { localStorage.setItem(TRAIN_FORM_STORAGE_KEY, JSON.stringify({ ...snapshotTrainFormConfig(), ...overrides })); } catch { /* best-effort */ }
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
  // Only restore 'kaggle' if the toggle is actually offered right now (see
  // refreshSystemInfo()'s kaggle_available check) — otherwise the row is
  // hidden and there'd be no visible control to switch back off it.
  if (cfg.backend === 'kaggle' && !trainBackendRow.hidden) setTrainBackend('kaggle');
  else setTrainBackend('local');
  if (cfg.base && trainingCatalog?.base_policies?.some((p) => p.name === cfg.base)) {
    trainBase.value = cfg.base;
  }
  trainEnvs.value = cfg.numEnvs || '';
  trainIters.value = cfg.iters || '';
  trainMinutes.value = cfg.minutes || '';
  trainVxLo.value = cfg.vxLo || ''; trainVxHi.value = cfg.vxHi || '';
  trainVyLo.value = cfg.vyLo || ''; trainVyHi.value = cfg.vyHi || '';
  trainYawLo.value = cfg.yawLo || ''; trainYawHi.value = cfg.yawHi || '';
  for (const key of TARGET_VAR_KEYS) {
    targetVarEls[key].abs.value = cfg.targets?.[key]?.abs || '';
    targetVarEls[key].pct.value = cfg.targets?.[key]?.pct || '';
  }
  trainPush.value = cfg.push || '';
  trainPushVel.value = cfg.pushVel || '';
  trainPushInterval.value = cfg.pushInterval || '';
  trainPushDir.value = cfg.pushDir || '';
  trainEntropyCoef.value = cfg.entropyCoef || '';

  renderVariableChrome();
  refreshTargetReferences();
  refreshNamePlaceholder();
  refreshCloneFromMismatchWarning();
  updateEnvsHint(); // backend-dependent hint text — see setTrainBackend() above
  updateEstimate();
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

// Rescans ./policies/ for anything trained by a separate process (the
// rugiar CLI, most commonly) since this server started — see
// ControlService.refresh_local_policies()'s docstring for why this can't
// just happen automatically every tick. The Policies list itself repaints
// from the next status push (supervisor.policies already has the new
// name(s) once the call resolves); the Clone-from dropdown is a separate
// fetch, same as after delete_policy, so it needs its own refresh too.
const refreshPoliciesBtn = $('#refresh-policies-btn');
refreshPoliciesBtn.addEventListener('click', () => {
  refreshPoliciesBtn.disabled = true;
  call('refresh_local_policies').then((added) => {
    refreshPoliciesBtn.disabled = false;
    refreshTrainingCatalog();
    if (added.length) {
      footer.textContent = `found ${added.length} new polic${added.length === 1 ? 'y' : 'ies'}: ${added.join(', ')}`;
    } else {
      footer.textContent = 'no new policies found';
    }
    setTimeout(() => { footer.textContent = 'connected'; }, 3000);
  }).catch((err) => {
    refreshPoliciesBtn.disabled = false;
    window.alert(`Refresh failed: ${err.message}`);
  });
});

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

    renderVariableChrome();
    refreshTargetReferences();
    refreshRewardScaleFields();
    refreshNamePlaceholder();
    refreshCloneFromMismatchWarning();
    updateCommandPreview();

    refreshFuseSources();
    refreshFuseMethods();
    refreshFuseTaskDisambiguation();
    updateFuseCmdPreview();
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
  const targets = resolveAllTargets();
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
    targets, push, pushVel, pushInterval, pushDir, entropyCoef, rewardScales,
    backend: trainBackend,
  };
}

function updateCommandPreview() {
  // rugiar (legged_gym/cli/rugiar.py) is a thin argv wrapper around this
  // same TrainingManager.start() — showing ITS invocation instead of the
  // raw `python legged_gym/scripts/web_train.py ...` subprocess line means
  // what's previewed here is something a user can actually paste into a
  // terminal and run themselves, not an internal implementation detail.
  const p = composeTrainingParams();
  const parts = [
    'rugiar train',
    `--task ${p.task || '<task>'}`,
    `--name ${p.name || '<policy name>'}`,
    `--num_envs ${Number.isFinite(p.numEnvs) ? p.numEnvs : '<num_envs>'}`,
  ];
  if (p.iterations !== null) parts.push(`--max_iterations ${p.iterations}`);
  if (p.minutes !== null) parts.push(`--max_minutes ${p.minutes}`);
  if (p.iterations === null && p.minutes === null) parts.push('--max_iterations <or> --max_minutes <required>');
  if (p.base) parts.push(`--from_policy ${p.base}`);
  if (p.backend === 'kaggle') parts.push('--backend kaggle');
  if (p.cmdVx) parts.push(`--cmd_vx_range ${p.cmdVx[0]} ${p.cmdVx[1]}`);
  if (p.cmdVy) parts.push(`--cmd_vy_range ${p.cmdVy[0]} ${p.cmdVy[1]}`);
  if (p.cmdYaw) parts.push(`--cmd_yaw_range ${p.cmdYaw[0]} ${p.cmdYaw[1]}`);
  for (const key of TARGET_VAR_KEYS) {
    const flag = trainingCatalog?.variables?.[key]?.flag || `${key}_target`;
    const value = p.targets[key];
    if (Number.isNaN(value)) parts.push(`--${flag} <ambiguous — both Absolute and % change are set>`);
    else if (value === undefined) parts.push(`--${flag} <no reference yet — pick a task or clone-from base>`);
    else if (value !== null) parts.push(`--${flag} ${value.toFixed(3)}`);
  }
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
    nameFieldTouched = false; // fresh open = re-suggest a name, see suggestedPolicyName()
    restoreTrainFormConfig(); // last-used config, if any — see its own docstring
    refreshNamePlaceholder();
    updateCommandPreview();
    updateEstimate();
    trainName.focus();
    trainName.select(); // pre-filled with a suggested name — select it so typing overwrites cleanly
  }
});

const targetVarInputs = TARGET_VAR_KEYS.flatMap((key) => [targetVarEls[key].abs, targetVarEls[key].pct]);
[trainName, trainBase, trainTask, trainIters, trainMinutes, trainEnvs,
 trainVxLo, trainVxHi, trainVyLo, trainVyHi, trainYawLo, trainYawHi,
 ...targetVarInputs, trainPush, trainPushVel, trainPushInterval, trainPushDir,
 trainEntropyCoef].forEach((el) => {
  el.addEventListener('input', updateCommandPreview);
  el.addEventListener('change', updateCommandPreview);
});
[trainBase, trainTask].forEach((el) => {
  el.addEventListener('change', () => {
    refreshTargetReferences();
    refreshRewardScaleFields();
    refreshNamePlaceholder(); // the auto-name suggestion tracks "Clone from"
    refreshCloneFromMismatchWarning();
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
  // finalize_policy() writes into policies/<name>/ with a plain
  // shutil.copyfile — reusing an existing policy's name silently overwrites
  // its checkpoint.pt/train_checkpoint.pt/meta.json the moment this job
  // finishes, with no undo. Block it outright rather than warn-and-allow —
  // this exact mistake already cost a real run once.
  if ((latestStatus?.policies || []).includes(p.name)) {
    return showTrainError(
      `'${p.name}' already exists — training with this name would overwrite its checkpoint when this run finishes. Pick a different name (e.g. '${p.name}_2').`);
  }
  if (!p.task) return showTrainError('Pick a task.');
  if (p.iterations === null && p.minutes === null) return showTrainError('Set a time budget — iterations, minutes, or both.');
  if (p.iterations !== null && p.iterations <= 0) return showTrainError('Iterations must be positive.');
  if (p.minutes !== null && p.minutes <= 0) return showTrainError('Minutes must be positive.');
  if (!Number.isFinite(p.numEnvs) || p.numEnvs <= 0) return showTrainError('Parallel environments must be positive.');
  if (p.cmdVx === undefined || p.cmdVy === undefined || p.cmdYaw === undefined) {
    return showTrainError('Fill in both ends of a command range, or leave the whole pair blank.');
  }
  for (const key of TARGET_VAR_KEYS) {
    const label = trainingCatalog?.variables?.[key]?.label || key;
    if (Number.isNaN(p.targets[key])) {
      return showTrainError(`Set either Absolute or % change for ${label}, not both.`);
    }
    if (p.targets[key] === undefined) {
      return showTrainError(`No reference for ${label} yet — pick a task or a clone-from base.`);
    }
  }

  showTrainError('');
  btnStartTraining.disabled = true;
  call('start_training', {
    policy_name: p.name, task: p.task, num_envs: p.numEnvs,
    max_iterations: p.iterations, max_minutes: p.minutes,
    base_policy: p.base, cmd_vx: p.cmdVx, cmd_vy: p.cmdVy, cmd_yaw: p.cmdYaw,
    base_height_target: p.targets.base_height,
    lin_vel_z_target: p.targets.lin_vel_z,
    ang_vel_xy_target: p.targets.ang_vel_xy,
    orientation_tilt_target: p.targets.orientation_tilt,
    push_robots: p.push === null ? null : p.push === 'on',
    max_push_vel_xy: p.pushVel, push_interval_s: p.pushInterval, push_dir: p.pushDir,
    entropy_coef: p.entropyCoef, reward_scale_overrides: p.rewardScales,
    backend: p.backend,
  }).then(() => {
    // snapshot BEFORE reset() clears every field below. base: p.name (not
    // trainBase.value, the run's OWN clone-from source) so the next "New
    // policy" panel defaults to cloning from what THIS run just produced —
    // curriculum-style training chains forward step-to-step without having
    // to reselect the base you just finished each time.
    saveTrainFormConfig({ base: p.name });
    createPolicyForm.reset();
    createPolicyForm.hidden = true;
    btnNewPolicy.textContent = '+ New policy…';
    setTrainBackend('local');
  }).catch((e) => {
    showTrainError(e.message);
  }).finally(() => {
    btnStartTraining.disabled = false;
  });
});

// job.max_iterations/max_minutes are the SAME budget the "Estimated time: …"
// line under the form already quoted before this run started (see
// updateEstimate()) — this just re-derives "how much is left" from the
// live iterations_done/elapsed_s the job reports, so the card can show
// progress without asking the server for a fresh estimate every tick.
function jobProgress(job) {
  const budgetS = job.max_minutes ? job.max_minutes * 60 : null;
  const hasIterBudget = job.max_iterations && job.iterations_done != null;
  let pct = null;
  if (hasIterBudget) {
    pct = job.iterations_done / job.max_iterations;
  } else if (budgetS) {
    pct = job.elapsed_s / budgetS;
  }
  if (pct != null) pct = Math.max(0, Math.min(1, pct));

  let etaText = null;
  if (hasIterBudget && job.iterations_done > 0) {
    // per-iteration rate observed so far this run — steadier than a
    // client-side average across past runs (what updateEstimate() uses
    // pre-start) since it reflects THIS machine's current load.
    const ratePerIter = job.elapsed_s / job.iterations_done;
    let remainingS = ratePerIter * (job.max_iterations - job.iterations_done);
    if (budgetS) remainingS = Math.min(remainingS, Math.max(0, budgetS - job.elapsed_s));
    etaText = remainingS <= 0 ? 'finishing up…' : `~${formatDuration(remainingS)} left`;
  } else if (budgetS) {
    const remainingS = Math.max(0, budgetS - job.elapsed_s);
    etaText = remainingS <= 0 ? 'finishing up…' : `~${formatDuration(remainingS)} left (time-boxed)`;
  } else if (hasIterBudget) {
    etaText = 'estimating…'; // iterations_done is 0 — no rate yet
  }
  // Kaggle jobs have no live iterations_done at all while running (see
  // kaggle_backend.py's module docstring), so with no minutes budget
  // either there's nothing to base a % or ETA on — indeterminate bar.
  return { pct, etaText };
}

function requestedBudgetText(job) {
  const parts = [];
  if (job.max_iterations) parts.push(`${job.max_iterations} iterations`);
  if (job.max_minutes) parts.push(`${job.max_minutes} min cap`);
  return parts.length ? parts.join(' / ') : '–';
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied ✓';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = original; btn.classList.remove('copied'); }, 1400);
  }).catch(() => { btn.textContent = 'Copy failed'; });
}

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
  //
  // A Kaggle job has no iterations_done signal at all while running (see
  // kaggle_backend.py's module docstring — no live progress, just queued/
  // running/complete) — without something else in the key, its card would
  // freeze at whatever elapsed_s happened to be on the FIRST render and
  // never move again, which is exactly what happened here. A coarse 5s
  // elapsed bucket gives it a heartbeat without going back to 10Hz.
  const key = jobs.map((j) => `${j.id}:${j.status}:${j.error || ''}:${j.iterations_done ?? ''}:` +
    `${j.kaggle_kernel_slug || ''}:${j.status === 'running' ? Math.floor(j.elapsed_s / 5) : ''}`).join('|');
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
    const nameGroup = document.createElement('span');
    nameGroup.className = 'job-name-group';
    const name = document.createElement('span');
    name.className = 'job-name';
    name.textContent = job.policy_name;
    nameGroup.appendChild(name);
    if (job.backend === 'kaggle') {
      const badge = document.createElement('span');
      badge.className = 'job-backend-badge';
      if (job.kaggle_kernel_slug) {
        const link = document.createElement('a');
        link.href = `https://www.kaggle.com/code/${job.kaggle_kernel_slug}`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = '☁️ Kaggle';
        badge.appendChild(link);
      } else {
        badge.textContent = '☁️ Kaggle';
      }
      nameGroup.appendChild(badge);
    }
    head.appendChild(nameGroup);
    const statusEl = document.createElement('span');
    statusEl.className = 'job-status';
    statusEl.textContent = job.status === 'running' ? 'training…' : job.status;
    head.appendChild(statusEl);
    row.appendChild(head);

    if (job.status === 'running') {
      const { pct, etaText } = jobProgress(job);
      const track = document.createElement('div');
      track.className = 'job-progress-track' + (pct == null ? ' indeterminate' : '');
      const fill = document.createElement('div');
      fill.className = 'job-progress-fill';
      if (pct != null) fill.style.width = `${(pct * 100).toFixed(1)}%`;
      track.appendChild(fill);
      row.appendChild(track);

      const meta = document.createElement('div');
      meta.className = 'job-progress-meta';
      // iterations_done is a live snapshot (updated every
      // TIME_BUDGET_CHUNK_ITERS iterations — see web_train.py's
      // write_progress()/TrainingManager._refresh_progress()), null until
      // the first chunk completes.
      const doneSpan = document.createElement('span');
      doneSpan.innerHTML = job.iterations_done != null
        ? `<strong>${job.iterations_done}</strong>${job.max_iterations ? `/${job.max_iterations}` : ''} iterations`
        : (job.backend === 'kaggle' ? 'no live iteration count (Kaggle)' : 'starting…');
      meta.appendChild(doneSpan);
      const elapsedSpan = document.createElement('span');
      elapsedSpan.textContent = `${formatDuration(job.elapsed_s)} elapsed`;
      meta.appendChild(elapsedSpan);
      const budgetSpan = document.createElement('span');
      budgetSpan.textContent = `requested: ${requestedBudgetText(job)}`;
      meta.appendChild(budgetSpan);
      if (etaText) {
        const etaSpan = document.createElement('span');
        etaSpan.innerHTML = `<strong>${etaText}</strong>`;
        meta.appendChild(etaSpan);
      }
      row.appendChild(meta);
    } else {
      const statusMeta = document.createElement('div');
      statusMeta.className = 'job-progress-meta';
      statusMeta.textContent = `after ${formatDuration(job.elapsed_s)}`;
      row.appendChild(statusMeta);
    }

    // Exactly the command this job is running (or ran) — see TrainingJob.
    // command's own docstring: "exactly what the UI previewed, minus the
    // interpreter path" — so copy/paste here reproduces this run locally.
    const cmdRow = document.createElement('div');
    cmdRow.className = 'job-cmd-row';
    const cmd = document.createElement('code');
    cmd.className = 'job-cmd';
    cmd.textContent = job.command;
    cmdRow.appendChild(cmd);
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'job-copy-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.title = 'Copy the exact command';
    copyBtn.onclick = () => copyToClipboard(job.command, copyBtn);
    cmdRow.appendChild(copyBtn);
    row.appendChild(cmdRow);

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

// Which chart metrics are checked — lives OUTSIDE renderPolicyInfo so
// stepping between policies with ↑/↓ (each a fresh renderPolicyInfo call)
// keeps whatever the user last chose instead of resetting to the 3
// defaults every time. Seeded with the defaults on first load; a key not
// present in a given policy's own menu just has nothing to render — it
// stays remembered in case a later policy has that same reward term.
const selectedChartKeys = new Set(METRIC_SPECS.map((s) => s.key));

// Cycled for any per-run reward-term series (series points' `rew_*` keys)
// beyond the three METRIC_SPECS defaults — those aren't known ahead of time,
// they're whatever training.py found common to every sampled block of THIS
// run, so they can't have fixed colors the way the three defaults do.
const EXTRA_SERIES_COLORS = ['#7b8fd9', '#d97bc7', '#7bd9a8', '#d9b87b', '#9b7bd9', '#d97b7b', '#7bc7d9', '#b8d97b'];

// A term's color must stay the same no matter which policy's menu it's
// drawn from — different policies have different sets of common reward
// terms (see training.py's common_terms intersection), so picking colors
// by POSITION in each policy's own sorted list reassigns colors every time
// you step to a different policy. Hashing the key name into the palette
// instead makes the mapping depend only on the key itself.
function hashColor(key) {
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) | 0;
  return EXTRA_SERIES_COLORS[Math.abs(hash) % EXTRA_SERIES_COLORS.length];
}

// Turns one series point's keys into the full menu of chartable metrics for
// this run: the three defaults (only if this run actually has them) plus
// every `rew_<term>` key training.py included, alphabetized and given a
// swatch color so the same term gets the same color everywhere — see
// hashColor() above.
function buildMetricMenu(series) {
  if (!series.length) return [];
  const keys = Object.keys(series[0]).filter((k) => k !== 'iteration');
  const core = METRIC_SPECS.filter((s) => keys.includes(s.key));
  const extraKeys = keys.filter((k) => !METRIC_SPECS.some((s) => s.key === k)).sort();
  const extras = extraKeys.map((key) => ({
    key,
    label: key.startsWith('rew_') ? key.slice(4).replace(/_/g, ' ') : key,
    color: hashColor(key),
  }));
  return [...core, ...extras];
}

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
function renderSeriesChart(series, specs) {
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

  for (const spec of specs) {
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
// itself in metricTooltip() if a term isn't listed here yet.
const REWARD_TERM_GLOSSARY = {
  base_height: 'Penalizes the base being above/below a target height off the ground — pulls the robot toward a specific standing/crouching posture. Target set in "Target variable" above.',
  tracking_lin_vel: 'Rewards matching the commanded forward/strafe velocity (the WASD/arrow-key command) — how well it goes where it’s told.',
  tracking_ang_vel: 'Rewards matching the commanded turn rate (yaw) — how well it turns at the commanded speed.',
  orientation: 'Penalizes the body tilting away from a target amount off upright (gravity not pointing straight down in the body frame) — keeps it from leaning or toppling. Target set in "Target variable" above; defaults to upright (0).',
  lin_vel_z: 'Penalizes vertical (bouncing) velocity of the base away from a target — discourages bobbing up and down. Target set in "Target variable" above; defaults to 0.',
  ang_vel_xy: 'Penalizes roll/pitch angular velocity away from a target — discourages wobbling side to side or front to back. Target set in "Target variable" above; defaults to 0.',
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
  actual_lin_vel_x: 'Not a reward term — a plain diagnostic of the base\'s actual forward/strafe velocity, logged for every run so it can be compared against tracking_lin_vel\'s target.',
};

// Tooltip text for the 3 fixed METRIC_SPECS keys — these are training-run
// stats, not `_reward_<name>` terms, so they don't belong in
// REWARD_TERM_GLOSSARY above but still need the same didactic hover/focus
// treatment everywhere they're shown (Result list and the chart checklist).
const CORE_METRIC_GLOSSARY = {
  episode_length: 'How many steps the average episode lasted before ending (falling, timing out, etc.) in this logged block — the main signal for "is it still falling over".',
  reward: 'The average total reward per episode in this logged block — the sum of every weighted reward term, the one number PPO is actually optimizing.',
  noise_std: 'The standard deviation of the exploration noise PPO is currently adding to actions — starts high for exploration and should shrink as training converges.',
};

// Shared by the Result list and the chart checklist so both use the exact
// same wording for the same key.
function metricTooltip(key) {
  if (key in CORE_METRIC_GLOSSARY) return CORE_METRIC_GLOSSARY[key];
  const term = key.startsWith('rew_') ? key.slice(4) : key;
  return REWARD_TERM_GLOSSARY[term] || `No description written for "${term}" yet.`;
}

// Shared data-tip + .info-term-name/.info-term-name-label wrapping — one
// place so every row in the merged Result list uses the same tooltip
// mechanism.
function metricLabelSpan(spec) {
  const name = document.createElement('span');
  name.className = 'info-term-name';
  name.tabIndex = 0;
  name.setAttribute('data-tip', metricTooltip(spec.key));
  const label = document.createElement('span');
  label.className = 'info-term-name-label';
  label.textContent = spec.label;
  name.appendChild(label);
  return name;
}

// A metric's "current value" — info.metrics.final only ever has the 3 core
// keys (see training.py's parse_training_log), so any extra rew_<term> key
// has to come from the last (most recent) point of the series instead.
function metricFinalValue(key, info) {
  if (info.metrics.final && info.metrics.final[key] != null) return info.metrics.final[key];
  const series = info.metrics.series;
  if (series && series.length) {
    const v = series[series.length - 1][key];
    if (v != null) return v;
  }
  return null;
}

function formatMetricValue(key, val) {
  const decimals = key === 'episode_length' ? 1 : key === 'reward' || key === 'noise_std' ? 3 : 4;
  return val.toFixed(decimals);
}

// isCoreMetric()'s three keys are training-run stats (a count, a sum, a
// std-dev) — not reward terms — so they don't get the pos/neg centered bar
// below; only actual reward-term rows are commensurable that way.
function isCoreMetric(key) {
  return METRIC_SPECS.some((s) => s.key === key);
}

// One merged row list backing the "Result" section: the chartable menu
// (core stats + whatever rew_<term> keys are common across every sampled
// block, see buildMetricMenu()) PLUS any reward term that only showed up in
// the final logged block (present in final_reward_terms but not common
// enough to make the series) — those still get a bar+value, just no
// checkbox, since there's no series to plot them against.
function buildDisplayRows(info, menu) {
  const rows = menu.map((spec) => ({ ...spec, chartable: true }));
  const menuKeys = new Set(rows.map((r) => r.key));
  const finalTerms = info.metrics.final_reward_terms || {};
  const extraTerms = Object.keys(finalTerms).filter((t) => !menuKeys.has(`rew_${t}`)).sort();
  for (const term of extraTerms) {
    const key = `rew_${term}`;
    rows.push({ key, label: term.replace(/_/g, ' '), color: hashColor(key), chartable: false });
  }
  return rows;
}

function rowValue(row, info) {
  return row.chartable ? metricFinalValue(row.key, info) : info.metrics.final_reward_terms[row.key.slice(4)];
}

// Replaces the old three-way split (a plain Result list mirroring the
// checkboxes, the checkbox grid itself, and a separate "what the reward
// actually optimized" bar list) with one list: checkbox (chart on/off,
// absent when there's no chart to toggle), swatch (matches the chart's
// line color), name+tooltip, a centered pos/neg bar for actual reward
// terms, and the value — all in one row so nothing needs cross-referencing
// between sections. `interactive` is false when there's no series to chart
// (a policy with only a couple of logged blocks) — checkboxes are dropped
// entirely rather than shown disabled, since "select for a chart that
// doesn't exist" isn't a real choice.
const METRIC_LIST_VISIBLE_COUNT = 10;

function renderMetricList(wrap, rows, info, interactive, checkboxes, refreshChart, syncCheckboxes) {
  wrap.replaceChildren();
  wrap.classList.toggle('non-interactive', !interactive);
  checkboxes.length = 0;
  const termRows = rows.filter((r) => !isCoreMetric(r.key));
  const maxAbs = Math.max(...termRows.map((r) => Math.abs(rowValue(r, info) ?? 0)), 1e-9);
  const presentRows = rows.filter((r) => rowValue(r, info) != null);
  const visibleRows = metricListExpanded ? presentRows : presentRows.slice(0, METRIC_LIST_VISIBLE_COUNT);
  for (const row of visibleRows) {
    const val = rowValue(row, info);
    const line = document.createElement('div');
    line.className = 'info-result-row';

    if (interactive) {
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      if (row.chartable) {
        chk.checked = selectedChartKeys.has(row.key);
        chk.addEventListener('change', () => {
          if (chk.checked) selectedChartKeys.add(row.key);
          else selectedChartKeys.delete(row.key);
          refreshChart();
        });
        checkboxes.push({ chk, spec: row });
      } else {
        chk.disabled = true;
        chk.title = 'No series data to chart for this run';
      }
      line.appendChild(chk);
    }

    const swatch = document.createElement('span');
    swatch.className = 'info-chart-check-swatch';
    swatch.style.background = row.color;
    line.appendChild(swatch);
    line.appendChild(metricLabelSpan(row));

    const track = document.createElement('div');
    track.className = 'info-result-bar-track';
    if (!isCoreMetric(row.key)) {
      track.classList.add('info-term-bar-track');
      const bar = document.createElement('div');
      bar.className = 'info-term-bar';
      // Track spans [-maxAbs, +maxAbs] centered at 50% — a term's bar grows
      // right from center if it rewards (positive), left if it penalizes.
      const halfPct = (Math.abs(val) / maxAbs) * 50;
      bar.style.width = `${halfPct}%`;
      bar.style.left = val >= 0 ? '50%' : `${50 - halfPct}%`;
      bar.style.background = val >= 0 ? 'var(--accent2)' : 'var(--danger, #d9534f)';
      track.appendChild(bar);
    }
    line.appendChild(track);

    const v = document.createElement('span');
    v.className = 'info-result-val';
    v.textContent = formatMetricValue(row.key, val);
    line.appendChild(v);

    if (interactive && row.chartable) {
      line.title = 'Double-click to show only this variable on the chart';
      line.addEventListener('dblclick', (e) => {
        e.preventDefault();
        selectedChartKeys.clear();
        selectedChartKeys.add(row.key);
        syncCheckboxes();
        refreshChart();
      });
    }
    wrap.appendChild(line);
  }

  if (presentRows.length > METRIC_LIST_VISIBLE_COUNT) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'info-result-show-all-btn';
    toggle.textContent = metricListExpanded ? 'Show fewer' : `Show all (${presentRows.length})`;
    toggle.addEventListener('click', () => {
      metricListExpanded = !metricListExpanded;
      renderMetricList(wrap, rows, info, interactive, checkboxes, refreshChart, syncCheckboxes);
    });
    wrap.appendChild(toggle);
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
    const series = info.metrics.series || [];
    // The 3 core stats (only if this run has series data to chart) plus
    // every rew_<term> the run tracked, or just the core specs as a plain
    // list when there's no chartable series at all (e.g. a policy with
    // only a few logged blocks).
    const menu = series.length > 1 ? buildMetricMenu(series) : METRIC_SPECS.filter((s) => info.metrics.final[s.key] != null);
    const rows = buildDisplayRows(info, menu);

    // Chart renders first — see #policy-info-top — so the merged list
    // below it (checkbox + bar + value per metric, see renderMetricList())
    // is read as "detail behind the lines above" rather than the other way
    // around.
    const listWrap = document.createElement('div');
    listWrap.className = 'info-result-list';
    const interactive = series.length > 1 && menu.length > 1;

    if (series.length > 1) {
      const chartWrap = document.createElement('div');
      chartWrap.appendChild(renderSeriesChart(series, menu.filter((s) => selectedChartKeys.has(s.key))));
      results.appendChild(chartWrap);

      const caption = document.createElement('div');
      caption.className = 'info-chart-caption';
      caption.textContent = `${series[0].iteration}→${series[series.length - 1].iteration} iterations, normalized to % of this run`;
      results.appendChild(caption);

      // Checked state comes from the module-level selectedChartKeys, not a
      // per-render default, so the selection survives stepping to a
      // different policy with ↑/↓.
      if (interactive) {
        const checkboxes = [];
        function refreshChart() {
          const active = menu.filter((s) => selectedChartKeys.has(s.key));
          chartWrap.replaceChild(renderSeriesChart(series, active), chartWrap.firstChild);
          renderMetricList(listWrap, rows, info, true, checkboxes, refreshChart, syncCheckboxes);
        }
        function syncCheckboxes() {
          for (const { chk, spec } of checkboxes) chk.checked = selectedChartKeys.has(spec.key);
        }

        const checklistHead = document.createElement('div');
        checklistHead.className = 'info-chart-checks-head';
        const selectAllBtn = document.createElement('button');
        selectAllBtn.type = 'button';
        selectAllBtn.className = 'info-chart-select-all-btn';
        selectAllBtn.textContent = 'Select all';
        selectAllBtn.addEventListener('click', () => {
          for (const spec of menu) selectedChartKeys.add(spec.key);
          syncCheckboxes();
          refreshChart();
        });
        checklistHead.appendChild(selectAllBtn);
        results.appendChild(checklistHead);

        renderMetricList(listWrap, rows, info, true, checkboxes, refreshChart, syncCheckboxes);
      } else {
        renderMetricList(listWrap, rows, info, false, [], null, null);
      }
    } else {
      renderMetricList(listWrap, rows, info, false, [], null, null);
    }
    results.appendChild(listWrap);
    top.appendChild(results);
    frag.appendChild(top);
  }

  const provenance = infoSection('Provenance');
  provenance.appendChild(kvList([
    ['Task', info.task],
    ['Simulator', info.simulator],
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

  // Only present on a fused policy (TrainingManager.fuse_policies() bakes this into
  // meta.json) — same object shape the "Command that will run" preview builds, plus
  // whatever `warnings` fuse_policies() returned (e.g. "sources trained on different
  // simulators"). This is the ONLY place those warnings live once the Fuse policies
  // panel closes — see its submit handler above for why it doesn't show them itself.
  if (info.fusion) {
    const fusion = infoSection('Fusion');
    const methodLabel = trainingCatalog?.fusion_methods?.find((m) => m.id === info.fusion.method)?.label
      || info.fusion.method;
    fusion.appendChild(kvList([
      ['Method', methodLabel],
      ['Sources', (info.fusion.sources || [])
        .map((s) => `${s.name} (${s.task}, weight ${s.weight})`).join(', ')],
    ]));
    if (info.fusion.warnings?.length) {
      const warn = document.createElement('p');
      warn.className = 'info-warn';
      warn.textContent = info.fusion.warnings.map((w) => `⚠ ${w}`).join('\n');
      fusion.appendChild(warn);
    }
    frag.appendChild(fusion);
  }

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

  // --camera wasn't passed: leave #camera-section hidden (its default in
  // index.html) rather than pointing <img> at a stream that will never
  // publish a frame.
  if (config.camera_enabled) {
    $('#camera-feed').src = '/camera.mjpg';
    $('#camera-section').hidden = false;
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
