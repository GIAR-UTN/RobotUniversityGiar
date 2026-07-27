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
let keysArmed = false; // true while the Simulator/Real-robot tab is active —
                        // keyboard shortcuts always work in that mode, and
                        // never while reading Docs; see selectView() below

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

function connect() {
  const url = `ws://${location.host}/ws`;
  ws = new WebSocket(url);
  ws.onopen = () => { footer.textContent = 'connected'; connDot.className = 'ok'; };
  ws.onclose = () => {
    footer.textContent = 'disconnected — retrying…';
    connDot.className = 'bad';
    setTimeout(connect, 1000);
  };
  ws.onerror = () => { connDot.className = 'bad'; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
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

  const realTab = document.querySelector('nav button[data-view="real"]');
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

function selectView(name) {
  document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.remove('active'));
  $(`#view-${name}`).classList.add('active');
  document.querySelector(`#tabs button[data-view="${name}"]`).classList.add('active');
  // The control sidebar (and its keyboard shortcuts) only make sense while
  // actively driving the robot — Simulator or Real-robot — not while
  // reading Docs, which gets the full viewport width instead.
  keysArmed = name === 'sim' || name === 'real';
  document.body.classList.toggle('controls-active', keysArmed);
}

document.querySelectorAll('#tabs button').forEach((btn) => {
  btn.addEventListener('click', () => { if (!btn.disabled) selectView(btn.dataset.view); });
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

document.addEventListener('keydown', (e) => {
  if (!keysArmed) return;
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

  const simView = $('#view-sim');
  const iframe = document.createElement('iframe');
  iframe.src = `http://localhost:${config.viser_port}/`;
  simView.appendChild(iframe);

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
  // keysArmed starts false (Docs is the default tab) and flips on whenever
  // the Simulator/Real-robot tab is selected — see selectView().
}

boot();
