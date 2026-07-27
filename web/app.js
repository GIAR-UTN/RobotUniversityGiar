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
let panelFocused = true; // page loads with the panel implicitly "focused"

const $ = (sel) => document.querySelector(sel);
const panel = $('#panel');
const footer = $('#footer');
const connDot = $('#conn-dot');
const activeLabel = $('#active-label');
const policyList = $('#policy-list');

const chkPush = $('#chk-push');
const chkAutoCmd = $('#chk-auto-cmd');
const sliderVx = $('#slider-vx');
const sliderVy = $('#slider-vy');
const sliderYaw = $('#slider-yaw');
let draggingCommand = false; // true while the user has a command slider grabbed —
                              // suppresses syncing slider position FROM status()
                              // pushes so the drag doesn't fight the server echo

let commandRanges = null; // {vx:[lo,hi], vy:[lo,hi], yaw:[lo,hi]} from /config — the
                           // exact envelope this policy was trained across (see
                           // SimAdapter.set_command); ramping/mouse-look never drive
                           // past these bounds

// ---- WASD + mouse-look "cruise" movement ----
// Holding a movement key ramps that axis by RAMP_STEP_FRACTION of its trained
// range every RAMP_INTERVAL_MS, toward whichever end of the range the key
// points at. Releasing the key does NOT return it to zero — it freezes at
// whatever value was reached (cruise control, not a brake pedal), by simply
// no longer being included in the ramp; the backend already holds the last
// set_command until told otherwise (see SimAdapter._apply_manual_command),
// so no extra "hold" logic is needed server-side.
const RAMP_STEP_FRACTION = 0.10;
const RAMP_INTERVAL_MS = 120;
const heldMoveKeys = new Set(); // keys currently held down, not tapped
let cruiseVx = 0, cruiseVy = 0, cruiseYaw = 0; // the single source of truth for
                                                // the manual command — sliders,
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
  btn.textContent = name;
  const key = keyByPolicy[name];
  if (key) {
    const kbd = document.createElement('kbd');
    kbd.textContent = key;
    btn.appendChild(kbd);
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

  $('#btn-pause').textContent = status.paused ? 'Resume' : 'Pause';

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

  if (status.random_events) {
    chkPush.checked = status.random_events.push_robots;
    const auto = status.random_events.auto_commands;
    chkAutoCmd.checked = auto;
    // While auto, the sliders are a read-only display of what the sim is
    // doing on its own; while manual, they're live input controls. Either
    // way, don't stomp on a slider the user currently has grabbed.
    [sliderVx, sliderVy, sliderYaw].forEach((el) => { el.disabled = auto; });
  }
  // Only sync FROM the server when nothing is actively driving the command
  // right now — otherwise this would fight a held key, an in-progress drag,
  // or the mouse-look pad with a ~100ms-stale echo of what we just sent.
  const activelyDriving = draggingCommand || heldMoveKeys.size > 0 || mouseLookActive;
  if (status.command && !activelyDriving) {
    cruiseVx = status.command.vx;
    cruiseVy = status.command.vy;
    cruiseYaw = status.command.yaw;
    updateCommandUI();
  }
}

function updateCommandUI() {
  sliderVx.value = cruiseVx;
  sliderVy.value = cruiseVy;
  sliderYaw.value = cruiseYaw;
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
}

document.querySelectorAll('#tabs button').forEach((btn) => {
  btn.addEventListener('click', () => { if (!btn.disabled) selectView(btn.dataset.view); });
});

// ---- controls panel buttons ----

$('#btn-pause').addEventListener('click', () => {
  send(latestStatus?.paused ? 'resume' : 'pause');
});
$('#btn-restart').addEventListener('click', () => send('restart'));
$('#estop').addEventListener('click', () => send('estop'));

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

function rampAxisValue(current, dir, range) {
  if (dir === 0 || !range) return current; // not held on this axis — frozen, not zeroed
  const target = dir > 0 ? range[1] : range[0];
  const step = dir * RAMP_STEP_FRACTION * Math.abs(target);
  const next = current + step;
  return dir > 0 ? Math.min(next, range[1]) : Math.max(next, range[0]);
}

function rampTick() {
  const dirVx = heldDirection('vx');
  const dirVy = heldDirection('vy');
  const dirYaw = heldDirection('yaw');
  if (dirVx === 0 && dirVy === 0 && dirYaw === 0) return; // nothing held — stay frozen, nothing to send
  cruiseVx = rampAxisValue(cruiseVx, dirVx, commandRanges?.vx);
  cruiseVy = rampAxisValue(cruiseVy, dirVy, commandRanges?.vy);
  cruiseYaw = rampAxisValue(cruiseYaw, dirYaw, commandRanges?.yaw);
  sendCruiseCommand();
}

setInterval(rampTick, RAMP_INTERVAL_MS);

chkPush.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked });
});
chkAutoCmd.addEventListener('change', () => {
  send('set_random_events', { push_robots: chkPush.checked, auto_commands: chkAutoCmd.checked });
  // Going manual right now should take the sliders' CURRENT position as the
  // first command, rather than waiting for the user to nudge one first.
  if (!chkAutoCmd.checked) sendCruiseCommand();
});

const sliderAxis = { 'slider-vx': 'vx', 'slider-vy': 'vy', 'slider-yaw': 'yaw' };
[sliderVx, sliderVy, sliderYaw].forEach((el) => {
  el.addEventListener('pointerdown', () => { draggingCommand = true; });
  el.addEventListener('pointerup', () => { draggingCommand = false; });
  el.addEventListener('input', () => {
    const v = parseFloat(el.value);
    const axis = sliderAxis[el.id];
    if (axis === 'vx') cruiseVx = v;
    else if (axis === 'vy') cruiseVy = v;
    else if (axis === 'yaw') cruiseYaw = v;
    sendCruiseCommand();
  });
});

// ---- mouse look (yaw) — click-and-drag, NOT Pointer Lock ----
// Pointer Lock would hide the cursor and capture ALL mouse input, which
// breaks clicking every other button in this panel, doesn't work over the
// cross-origin Simulator iframe, and is force-released by the browser on
// Escape (our E-STOP key) in a way pages can't override. Pointer Capture on
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
  // Dragging right should turn right; yaw+ means "turn right" (matches
  // keymap.json's ArrowRight: sign +1), so drag-right adds.
  cruiseYaw = clampToRange(cruiseYaw + dx * MOUSE_YAW_SENSITIVITY, commandRanges?.yaw);
  sendCruiseCommand();
});
function endMouseLook() {
  mouseLookActive = false;
  lookPad.classList.remove('active');
}
lookPad.addEventListener('pointerup', endMouseLook);
lookPad.addEventListener('pointercancel', endMouseLook);

// ---- keyboard shortcuts ----
// Only bound while the controls panel (not the cross-origin Simulator
// iframe) has DOM focus — the iframe cannot forward keydown to us, and
// there's no way to force it to relinquish focus from here. The E-STOP
// *button* above is the one control that always works regardless of focus,
// since it's a click, not a keystroke — treat it as the primary estop path.

function setPanelFocused(focused) {
  panelFocused = focused;
  panel.classList.toggle('focused', focused);
  if (!focused && heldMoveKeys.size > 0) {
    // Focus can move to the iframe WHILE a key is physically still held —
    // the resulting keyup then goes to the iframe, not to us, so we'd never
    // find out it was released. Cruise mode freezes on a DELIBERATE
    // release, but this isn't one — we've lost the ability to track it, so
    // the safe move is to zero out here, not guess that "keep cruising" is
    // still what the user wants.
    heldMoveKeys.clear();
    cruiseVx = 0;
    cruiseVy = 0;
    cruiseYaw = 0;
    sendCruiseCommand();
  }
}

panel.addEventListener('click', () => { panel.focus(); setPanelFocused(true); });
panel.addEventListener('focusin', () => setPanelFocused(true));

const simIframeHolder = $('#view-sim');
window.addEventListener('blur', () => {
  // Standard trick for detecting "focus moved into an iframe" from the
  // parent page — see app.js module comment / HANDOFF_control_web.md §3-B
  // "Keyboard shortcuts" for why this is necessary and what it can't do.
  setTimeout(() => {
    const simIframe = simIframeHolder.querySelector('iframe');
    if (simIframe && document.activeElement === simIframe) setPanelFocused(false);
  }, 0);
});

document.addEventListener('keydown', (e) => {
  if (!panelFocused) return;
  const binding = keymap[e.key];
  if (!binding) return;
  e.preventDefault();
  if (binding.action === 'move') {
    if (heldMoveKeys.has(e.key)) return; // ignore OS key-repeat, already engaged
    heldMoveKeys.add(e.key);
    engageManualIfNeeded();
    rampTick(); // immediate feedback instead of waiting up to RAMP_INTERVAL_MS
  }
  else if (binding.action === 'switch') send('request_switch', { name: binding.policy });
  else if (binding.action === 'pause_toggle') send(latestStatus?.paused ? 'resume' : 'pause');
  else if (binding.action === 'restart') $('#btn-restart').click();
  else if (binding.action === 'estop') send('estop');
});

document.addEventListener('keyup', (e) => {
  const binding = keymap[e.key];
  if (!binding || binding.action !== 'move') return;
  // Deliberately no re-send here: releasing a key just stops ramping that
  // axis (it drops out of heldMoveKeys, so the next rampTick's
  // heldDirection() for it is 0) — cruise, not brake. The backend already
  // holds the last set_command until told otherwise.
  heldMoveKeys.delete(e.key);
});

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

  // Clamp the sliders (and, via commandRanges, the arrow keys) to the exact
  // velocity envelope this policy was trained across (env_cfg.commands.ranges
  // — see SimAdapter.set_command), not an arbitrary UI guess.
  if (config.command_ranges) {
    commandRanges = config.command_ranges;
    const setRange = (el, [lo, hi]) => { el.min = lo; el.max = hi; };
    setRange(sliderVx, config.command_ranges.vx);
    setRange(sliderVy, config.command_ranges.vy);
    setRange(sliderYaw, config.command_ranges.yaw);
  }

  connect();
  setPanelFocused(true);
}

boot();
