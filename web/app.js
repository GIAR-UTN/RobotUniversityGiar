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
let keyByPolicy = {}; // policy name -> bound key, precomputed once at boot (not per render)
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
  return btn;
}

function applyStatus(status) {
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
  const names = status.policies || [];
  const namesKey = names.join(' ') + ' ' + status.active;
  if (namesKey !== renderedPolicyNames) {
    renderedPolicyNames = namesKey;
    policyList.innerHTML = '';
    for (const name of names) {
      policyList.appendChild(policyButtonRow(name, status.active));
    }
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

function setKeysArmed() {
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
    send('set_random_events', { push_robots: chkPush.checked, auto_commands: false });
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
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked });
});
chkAutoCmd.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked });
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
  const binding = keymap[key];
  if (!binding) return;
  if (binding.action === 'switch') send('request_switch', { name: binding.policy });
  else if (binding.action === 'pause_toggle') send(latestStatus?.paused ? 'resume' : 'pause');
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
  if (!binding) return;
  e.preventDefault();
  setKeycapActive(e.key, true);
  if (binding.action === 'move') {
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
    if (!binding) return;
    if (binding.action === 'move') {
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

// ---- draggable panel sections (reorder + persist) ----

const PANEL_ORDER_KEY = 'giar.panelOrder.v1';
let dragState = null;

function onHandleDown(e) {
  const handle = e.currentTarget;
  const section = handle.closest('.panel-section');
  handle.setPointerCapture(e.pointerId);
  dragState = { section, pointerId: e.pointerId };
  section.classList.add('dragging');
  document.addEventListener('pointermove', onHandleMove);
  document.addEventListener('pointerup', onHandleUp, { once: true });
}

function onHandleMove(e) {
  if (!dragState) return;
  const { section } = dragState;
  const siblings = [...panel.querySelectorAll('.panel-section:not(.dragging)')];
  for (const sib of siblings) {
    const r = sib.getBoundingClientRect();
    const mid = r.top + r.height / 2;
    if (e.clientY < mid && (sib.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      panel.insertBefore(section, sib);
      break;
    }
    if (e.clientY > mid && (section.compareDocumentPosition(sib) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      panel.insertBefore(section, sib.nextSibling);
      break;
    }
  }
}

function onHandleUp() {
  dragState.section.classList.remove('dragging');
  document.removeEventListener('pointermove', onHandleMove);
  savePanelOrder();
  dragState = null;
}

function initSectionDrag() {
  document.querySelectorAll('.drag-handle').forEach((handle) => {
    handle.addEventListener('pointerdown', onHandleDown);
  });
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
const trainHeight = $('#train-height');
const trainPush = $('#train-push');
const trainPushVel = $('#train-push-vel');
const trainPushInterval = $('#train-push-interval');
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
    if (!hasIters) {
      // Minutes-only: nothing to estimate — the time budget IS the answer.
      trainEstimate.textContent = `Estimated time: time-boxed to ${minutes} minute${minutes === 1 ? '' : 's'} — stops then regardless of iterations reached.`;
      return;
    }
    if (!Number.isFinite(numEnvs) || numEnvs <= 0) { trainEstimate.textContent = 'Estimated time: –'; return; }
    call('estimate_training_time', { max_iterations: iterations, num_envs: numEnvs }).then((est) => {
      const cap = hasMinutes ? ` (also capped at ${minutes} minute${minutes === 1 ? '' : 's'}, whichever hits first)` : '';
      trainEstimate.textContent = est.basis === 'measured'
        ? `Estimated time: ~${formatDuration(est.seconds)} (based on ${est.samples} previous run${est.samples === 1 ? '' : 's'} on this machine)${cap}`
        : `Estimated time: no runs on this machine yet — the first one calibrates this estimate.${cap}`;
    }).catch(() => { trainEstimate.textContent = 'Estimated time: –'; });
  }, 250);
}

function showTrainError(msg) {
  trainError.textContent = msg;
  trainError.classList.toggle('show', !!msg);
}

function refreshTrainingCatalog() {
  call('training_catalog').then((catalog) => {
    trainingCatalog = catalog;
    const prevTask = trainTask.value, prevBase = trainBase.value;
    trainTask.innerHTML = '';
    for (const t of catalog.tasks) {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      trainTask.appendChild(opt);
    }
    if (catalog.tasks.includes(prevTask)) trainTask.value = prevTask;

    trainBase.innerHTML = '<option value="">— from scratch —</option>';
    for (const p of catalog.base_policies) {
      const opt = document.createElement('option');
      opt.value = p.name;
      opt.textContent = p.checkpoint ? p.name : `${p.name} (no checkpoint on this machine)`;
      opt.disabled = !p.checkpoint;
      trainBase.appendChild(opt);
    }
    if (catalog.base_policies.some((p) => p.name === prevBase)) trainBase.value = prevBase;

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
  const name = trainName.value.trim();
  const task = trainTask.value;
  const iterations = parseInt(trainIters.value, 10);
  const minutes = parseFloat(trainMinutes.value);
  const numEnvs = parseInt(trainEnvs.value, 10);
  const base = trainBase.value || null;
  const cmdVx = rangePair(trainVxLo, trainVxHi);
  const cmdVy = rangePair(trainVyLo, trainVyHi);
  const cmdYaw = rangePair(trainYawLo, trainYawHi);
  const heightRaw = trainHeight.value.trim();
  const height = heightRaw === '' ? null : parseFloat(heightRaw);
  const push = trainPush.value || null; // '' -> null -> "leave task default"
  const pushVelRaw = trainPushVel.value.trim();
  const pushVel = pushVelRaw === '' ? null : parseFloat(pushVelRaw);
  const pushIntervalRaw = trainPushInterval.value.trim();
  const pushInterval = pushIntervalRaw === '' ? null : parseFloat(pushIntervalRaw);
  return {
    name, task, iterations: Number.isFinite(iterations) ? iterations : null,
    minutes: Number.isFinite(minutes) ? minutes : null, numEnvs, base, cmdVx, cmdVy, cmdYaw,
    height, push, pushVel, pushInterval,
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
    parts.push(`--from_checkpoint ${source?.checkpoint || '<' + p.base + "'s checkpoint>"}`);
  }
  if (p.cmdVx) parts.push(`--cmd_vx_range ${p.cmdVx[0]} ${p.cmdVx[1]}`);
  if (p.cmdVy) parts.push(`--cmd_vy_range ${p.cmdVy[0]} ${p.cmdVy[1]}`);
  if (p.cmdYaw) parts.push(`--cmd_yaw_range ${p.cmdYaw[0]} ${p.cmdYaw[1]}`);
  if (p.height !== null) parts.push(`--base_height_target ${p.height}`);
  if (p.push) parts.push(`--push_robots ${p.push}`);
  if (p.pushVel !== null) parts.push(`--max_push_vel_xy ${p.pushVel}`);
  if (p.pushInterval !== null) parts.push(`--push_interval_s ${p.pushInterval}`);
  trainCmdPreview.textContent = parts.join(' ');
}

btnNewPolicy.addEventListener('click', () => {
  const opening = createPolicyForm.hidden;
  createPolicyForm.hidden = !opening;
  btnNewPolicy.textContent = opening ? 'Cancel' : '+ New policy…';
  if (opening) { showTrainError(''); updateCommandPreview(); updateEstimate(); trainName.focus(); }
});

[trainName, trainBase, trainTask, trainIters, trainMinutes, trainEnvs,
 trainVxLo, trainVxHi, trainVyLo, trainVyHi, trainYawLo, trainYawHi,
 trainHeight, trainPush, trainPushVel, trainPushInterval].forEach((el) => {
  el.addEventListener('input', updateCommandPreview);
  el.addEventListener('change', updateCommandPreview);
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

  showTrainError('');
  btnStartTraining.disabled = true;
  call('start_training', {
    policy_name: p.name, task: p.task, num_envs: p.numEnvs,
    max_iterations: p.iterations, max_minutes: p.minutes,
    base_policy: p.base, cmd_vx: p.cmdVx, cmd_vy: p.cmdVy, cmd_yaw: p.cmdYaw,
    base_height_target: p.height,
    push_robots: p.push === null ? null : p.push === 'on',
    max_push_vel_xy: p.pushVel, push_interval_s: p.pushInterval,
  }).then(() => {
    createPolicyForm.reset();
    createPolicyForm.hidden = true;
    btnNewPolicy.textContent = '+ New policy…';
  }).catch((e) => {
    showTrainError(e.message);
  }).finally(() => {
    btnStartTraining.disabled = false;
  });
});

function renderTrainingJobs(jobs) {
  // Dedupe on everything EXCEPT elapsed_s — elapsed_s ticks up every status
  // push while a job runs, and rebuilding this DOM at ~10Hz for that alone
  // would reintroduce exactly the perf issue the policy-list dedupe above
  // already exists to avoid (see HANDOFF_control_web.md's post-merge
  // incident note). A running job's card just doesn't show a live timer.
  const key = jobs.map((j) => `${j.id}:${j.status}:${j.error || ''}`).join('|');
  if (key === renderedTrainingJobsKey) return;
  const justFinished = renderedTrainingJobsKey !== null &&
    jobs.some((j) => j.status === 'done' && !renderedTrainingJobsKey.includes(`${j.id}:done`));
  renderedTrainingJobsKey = key;

  trainingJobsEl.innerHTML = '';
  for (const job of jobs) {
    const row = document.createElement('div');
    row.className = `job-row ${job.status}`;
    const head = document.createElement('div');
    head.className = 'job-head';
    const name = document.createElement('span');
    name.className = 'job-name';
    name.textContent = job.policy_name;
    const statusEl = document.createElement('span');
    statusEl.className = 'job-status';
    statusEl.textContent = job.status === 'running' ? 'training…' : `${job.status} (${job.elapsed_s}s)`;
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

// ---- boot ----

async function boot() {
  const [config, km] = await Promise.all([
    fetch('/config').then((r) => r.json()),
    fetch('/keymap.json').then((r) => r.json()),
  ]);
  keymap = Object.fromEntries(
    Object.entries(km).filter(([k, v]) => v && typeof v === 'object' && !k.startsWith('_'))
  );
  keyByPolicy = {};
  for (const [key, binding] of Object.entries(keymap)) {
    if (binding.action === 'switch' && binding.policy) keyByPolicy[binding.policy] = key;
  }

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
