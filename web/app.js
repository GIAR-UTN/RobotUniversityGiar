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
$('#btn-restart').addEventListener('click', () => {
  // No ControlService.restart() exists yet — swap_experiment.py's own
  // "Restart" button resets the sim loop itself. The web panel exposes the
  // same intent as pause+estop-reset today; wiring a real restart() is a
  // small, separate follow-up once a networked reset call is needed.
  console.warn('Restart is not yet wired to ControlService over the network — see swap_experiment.py.');
});
$('#estop').addEventListener('click', () => send('estop'));

// ---- keyboard shortcuts ----
// Only bound while the controls panel (not the cross-origin Simulator
// iframe) has DOM focus — the iframe cannot forward keydown to us, and
// there's no way to force it to relinquish focus from here. The E-STOP
// *button* above is the one control that always works regardless of focus,
// since it's a click, not a keystroke — treat it as the primary estop path.

function setPanelFocused(focused) {
  panelFocused = focused;
  panel.classList.toggle('focused', focused);
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
  if (binding.action === 'switch') send('request_switch', { name: binding.policy });
  else if (binding.action === 'pause_toggle') send(latestStatus?.paused ? 'resume' : 'pause');
  else if (binding.action === 'restart') $('#btn-restart').click();
  else if (binding.action === 'estop') send('estop');
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

  connect();
  setPanelFocused(true);
}

boot();
