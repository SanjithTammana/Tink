// Same-origin Django API. Every mutation belongs to this page, including late requests.
const sid = window.crypto?.randomUUID?.() ?? String(Math.random()).slice(2);
let queue = Promise.resolve();
const enqueue = (task) => { queue = queue.then(task, task); return queue; };

async function request(path, body, timeoutMs = 35000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, body === undefined ? { signal: controller.signal } : {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, sid }), signal: controller.signal,
    });
    if (!response.ok) throw new Error(path + ' failed (' + response.status + ')');
    return await response.json();
  } finally { clearTimeout(timer); }
}

export const api = {
  state: () => request('/api/state'),
  ownsSession: (state) => state?.owner_sid === sid,
  readSpelling: async () => {
    await queue;
    const state = await api.state();
    if (!api.ownsSession(state) || state.mode !== 'asl' || state.phase !== 'running') {
      throw new Error('This page no longer owns the spelling session.');
    }
    return state.message ?? '';
  },
  trace: () => request('/api/trace'),
  flushMutations: () => queue,
  startSession: (mode, { isCurrent = () => true } = {}) => enqueue(() => isCurrent() ? request('/api/session/start', { mode }) : null),
  stopSession: () => enqueue(() => request('/api/session/stop', {})),
  setCamera: (on) => enqueue(() => request('/api/camera', { on })),
  normal: (action, body = {}) => enqueue(() => request('/api/normal/' + action, body, action === 'playback' ? 5000 : 35000)),
  asl: (action, body = {}) => enqueue(() => request('/api/asl/' + action, body)),
};

export async function confirmAndSpeak({ option, index, clipId, resultId, isCurrent = () => true,
  confirm = (body) => api.normal('confirm', body), play }) {
  if (!clipId || !resultId || !option?.id || !isCurrent()) throw new Error('The choice is stale.');
  const expected = { index, clip_id: clipId, option_id: option.id, result_id: resultId };
  const accepted = await confirm(expected);
  if (accepted?.ok !== true || accepted.text !== option.text || accepted.kind !== option.kind
      || accepted.clip_id !== clipId || accepted.option_id !== option.id || accepted.result_id !== resultId) {
    throw new Error('The confirmed choice no longer matches the displayed recording.');
  }
  if (!isCurrent()) return { accepted, spoken: false };
  return { accepted, spoken: await play(accepted.text) };
}

// Nonoverlapping polling. Failed starts retry with capped backoff; errors stay visible.
export function startSessionLoop(mode, { onState, onConnection, intervalMs = 250, client = api,
  schedule = setTimeout, cancel = clearTimeout } = {}) {
  let alive = true, timer, started = false, failures = 0;
  async function tick() {
    let delay = intervalMs;
    try {
      if (!started) {
        await client.startSession(mode, { isCurrent: () => alive });
        if (!alive) return;
        started = true;
      }
      const next = await client.state();
      if (!alive) return;
      onState(next.mode === mode ? next : { phase: 'connecting' });
      onConnection(false);
      failures = 0;
      if ((!next.mode || next.mode === mode) && ['idle', 'stopped', 'stopping'].includes(next.phase)) {
        started = false;
        delay = 500;
      }
    } catch {
      if (!alive) return;
      onConnection(true);
      delay = Math.min(8000, 500 * 2 ** Math.min(failures++, 4));
    }
    if (alive) timer = schedule(tick, delay);
  }
  tick();
  return () => { alive = false; cancel(timer); client.stopSession().catch(() => {}); };
}

window.addEventListener('pagehide', () => navigator.sendBeacon?.('/api/session/stop',
  new Blob([JSON.stringify({ sid })], { type: 'application/json' })));

export function aslShortcut(event) {
  const target = event.target;
  if (event.defaultPrevented || event.isComposing || event.metaKey || event.ctrlKey || event.altKey
      || target?.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target?.tagName)) return null;
  const key = event.key.toLowerCase();
  if (key === ' ' && /^(BUTTON|A|SUMMARY)$/.test(target?.tagName)) return null;
  let command = null;
  if (/^[1-8]$/.test(key)) command = { action: 'add', index: Number(key) - 1 };
  else if (key === ' ') command = { action: 'space' };
  else if (key === 'd' || key === 'backspace') command = { action: 'delete' };
  else if (key === 'c') command = { action: 'clear' };
  else if (key === 's') command = { action: 'speak' };
  return event.repeat ? null : command;
}

