import assert from 'node:assert/strict';
import { test } from 'node:test';

const utterances = [];
globalThis.window = {
  crypto: { randomUUID: () => 'test-page-owner' }, addEventListener() {},
  speechSynthesis: { cancel() {}, speak(value) { utterances.push(value); } },
  SpeechSynthesisUtterance: class { constructor(text) { this.text = text; } },
};
const { api, confirmAndSpeak, startSessionLoop, aslShortcut } = await import('./api.js');
const speech = await import('./speech.js');
const flush = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };
const option = { id: '2', text: 'Please stop.', kind: 'hypothesis' };
const accepted = () => ({ ok: true, ...option, clip_id: 'clip', option_id: '2', result_id: 'result' });

test('all mutations carry the page owner and ASL edits cannot overtake each other', async () => {
  const requests = [], first = deferred();
  globalThis.fetch = async (url, init) => {
    requests.push([url, JSON.parse(init.body)]);
    if (requests.length === 1) await first.promise;
    return { ok: true, json: async () => ({ ok: true }) };
  };
  const add = api.asl('add', { letter: 'A', sid: 'spoof' });
  const remove = api.asl('delete');
  await flush();
  assert.equal(requests.length, 1);
  first.resolve();
  await Promise.all([add, remove]);
  await api.setCamera(false);
  await api.normal('record');
  await api.normal('playback', { active: true });
  await api.stopSession();
  assert.deepEqual(requests.slice(0, 2).map(([url]) => url), ['/api/asl/add', '/api/asl/delete']);
  assert.ok(requests.every(([, body]) => body.sid === 'test-page-owner'));
});

test('accepted confirmation includes full identity before its wording can be spoken', async () => {
  const calls = [];
  const answer = await confirmAndSpeak({ option, index: 1, clipId: 'clip', resultId: 'result',
    confirm: async (body) => { calls.push(body); return accepted(); },
    play: async (text) => { calls.push(text); return true; } });
  assert.deepEqual(calls, [{ index: 1, clip_id: 'clip', option_id: '2', result_id: 'result' }, option.text]);
  assert.equal(answer.spoken, true);
});

test('failed, mismatched or stale confirmation never invokes speech', async () => {
  for (const delta of [{ ok: false }, { text: 'Another message' }, { kind: 'message' },
    { clip_id: 'other' }, { option_id: '1' }, { result_id: 'retry-result' }]) {
    let spoken = 0;
    await assert.rejects(confirmAndSpeak({ option, index: 1, clipId: 'clip', resultId: 'result',
      confirm: async () => ({ ...accepted(), ...delta }), play: () => { spoken++; } }));
    assert.equal(spoken, 0);
  }
  let live = true, spoken = 0;
  const delayed = deferred();
  const work = confirmAndSpeak({ option, index: 1, clipId: 'clip', resultId: 'result',
    isCurrent: () => live, confirm: () => delayed.promise, play: () => { spoken++; } });
  live = false; delayed.resolve(accepted());
  assert.equal((await work).spoken, false);
  assert.equal(spoken, 0);
  await assert.rejects(confirmAndSpeak({ option, index: 1, clipId: 'clip', resultId: 'result',
    confirm: async () => { throw new Error('offline'); }, play: () => { spoken++; } }));
  assert.equal(spoken, 0);
});

test('startup retries after disconnection, restarts stopped sessions, and cancels cleanly', async () => {
  let starts = 0, offline = true, phase = 'running';
  const scheduled = [], states = [], connections = [];
  const client = {
    startSession: async () => { starts++; if (offline) throw new Error('offline'); },
    state: async () => ({ mode: 'normal', phase }),
    stopSession: async () => {},
  };
  const stop = startSessionLoop('normal', { client, onState: (v) => states.push(v),
    onConnection: (v) => connections.push(v), schedule: (fn, delay) => { scheduled.push({ fn, delay }); return 1; }, cancel() {} });
  await flush();
  assert.equal(scheduled[0].delay, 500);
  offline = false;
  await scheduled.shift().fn();
  assert.equal(starts, 2);
  assert.equal(states.at(-1).phase, 'running');
  assert.equal(connections.at(-1), false);
  phase = 'stopped'; await scheduled.shift().fn();
  assert.equal(scheduled.at(-1).delay, 500);
  phase = 'running'; await scheduled.shift().fn();
  assert.equal(starts, 3);
  stop();
});

test('unmount during startup does not begin polling or update abandoned state', async () => {
  const waiting = deferred();
  let predicate, updates = 0, timers = 0, stops = 0;
  const stop = startSessionLoop('asl', { client: {
    startSession: (_, { isCurrent }) => { predicate = isCurrent; return waiting.promise; },
    state: async () => { throw new Error('must not poll'); },
    stopSession: async () => { stops++; },
  }, onState: () => updates++, onConnection: () => updates++, schedule: () => timers++, cancel() {} });
  stop();
  assert.equal(predicate(), false);
  waiting.resolve(); await flush();
  assert.equal(updates, 0); assert.equal(timers, 0); assert.equal(stops, 1);
});

test('ASL shortcuts work on focused buttons while preserving native activation and editing', () => {
  const key = (value, extra = {}) => aslShortcut({ key: value, target: { tagName: 'BUTTON' }, ...extra });
  assert.deepEqual(key('3'), { action: 'add', index: 2 });
  assert.deepEqual(key('s'), { action: 'speak' });
  assert.equal(key(' '), null);
  assert.equal(key('1', { repeat: true }), null);
  assert.equal(key('s', { repeat: true }), null);
  assert.equal(key('c', { ctrlKey: true }), null);
  assert.equal(key('1', { target: { tagName: 'INPUT' } }), null);
  assert.equal(key('1', { target: { isContentEditable: true } }), null);
  assert.deepEqual(key(' ', { target: { tagName: 'H1' } }), { action: 'space' });
});

test('browser playback reserves first, renews, and releases on end', async () => {
  const original = api.normal, oldInterval = globalThis.setInterval, oldClear = globalThis.clearInterval;
  const calls = [];
  let renew;
  api.normal = async (action, body) => { calls.push({ action, ...body }); return { ok: true }; };
  globalThis.setInterval = (fn, ms) => { assert.equal(ms, 2000); renew = fn; return 123; };
  globalThis.clearInterval = () => {};
  try {
    utterances.length = 0;
    assert.equal(await speech.speak('Confirmed wording'), true);
    assert.deepEqual(calls[0], { action: 'playback', active: true });
    assert.equal(utterances[0].text, 'Confirmed wording');
    assert.equal(speech.isSpeaking(), true);
    renew(); await flush();
    assert.equal(calls[1].active, true);
    utterances[0].onend();
    assert.equal(calls.at(-1).active, false);
  } finally { speech.stopSpeaking(); api.normal = original; globalThis.setInterval = oldInterval; globalThis.clearInterval = oldClear; }
});

test('cancelled pending reservation cannot start speech, and failed reservation fails closed', async () => {
  const original = api.normal, waiting = deferred(), calls = [];
  api.normal = (action, body) => { calls.push(body.active); return body.active ? waiting.promise : Promise.resolve({ ok: true }); };
  try {
    utterances.length = 0;
    const work = speech.speak('Do not speak after unmount');
    await flush(); speech.stopSpeaking(); waiting.resolve({ ok: true });
    assert.equal(await work, false);
    assert.equal(utterances.length, 0);
    assert.deepEqual(calls, [true, false]);
    api.normal = async () => { throw new Error('guard unavailable'); };
    assert.equal(await speech.speak('Must reserve first'), false);
    assert.equal(utterances.length, 0);
  } finally { speech.stopSpeaking(); api.normal = original; }
});

test('ASL playback uses shared state without a normal-mode lease', async () => {
  const original = api.normal;
  api.normal = async () => { throw new Error('ASL must not request a normal endpoint'); };
  try {
    utterances.length = 0;
    assert.equal(await speech.speak('ASL message', { guardRecording: false }), true);
    assert.equal(speech.isSpeaking(), true);
    utterances[0].onend();
  } finally { speech.stopSpeaking(); api.normal = original; }
});


test('renewal failure stops active playback, and speech errors release the lease', async () => {
  const original = api.normal, oldInterval = globalThis.setInterval, oldClear = globalThis.clearInterval;
  const oldCancel = window.speechSynthesis.cancel;
  let renew, failRenewal = false, cancelled = 0;
  const calls = [];
  api.normal = async (_, body) => {
    calls.push(body.active);
    if (body.active && failRenewal) throw new Error('lease renewal failed');
    return { ok: true };
  };
  globalThis.setInterval = (fn) => { renew = fn; return 456; };
  globalThis.clearInterval = () => {};
  window.speechSynthesis.cancel = () => cancelled++;
  try {
    utterances.length = 0;
    await speech.speak('Renew this playback');
    const before = cancelled;
    failRenewal = true; renew(); await flush();
    assert.equal(cancelled, before + 1);
    assert.equal(calls.at(-1), false);
    failRenewal = false;
    await speech.speak('Browser reports an error');
    utterances.at(-1).onerror();
    assert.equal(calls.at(-1), false);
  } finally {
    speech.stopSpeaking(); api.normal = original;
    globalThis.setInterval = oldInterval; globalThis.clearInterval = oldClear;
    window.speechSynthesis.cancel = oldCancel;
  }
});

test('startup backoff remains bounded during a sustained outage', async () => {
  const tasks = [];
  const stop = startSessionLoop('normal', {
    client: { startSession: async () => { throw new Error('offline'); }, stopSession: async () => {} },
    onState() {}, onConnection() {},
    schedule: (fn, delay) => { tasks.push({ fn, delay }); return 1; }, cancel() {},
  });
  await flush();
  const delays = [];
  for (let i = 0; i < 8; i++) { const task = tasks.shift(); delays.push(task.delay); await task.fn(); }
  stop();
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 8000, 8000, 8000, 8000]);
});


test('ASL speech reads current owned spelling and rejects another page or stopped session', async () => {
  let state = { mode: 'asl', phase: 'running', owner_sid: 'another-page', message: 'OTHER PERSON' };
  globalThis.fetch = async () => ({ ok: true, json: async () => state });
  await assert.rejects(api.readSpelling(), /no longer owns/);
  state = { ...state, owner_sid: 'test-page-owner', message: 'MY CURRENT SPELLING' };
  assert.equal(await api.readSpelling(), 'MY CURRENT SPELLING');
  state.phase = 'stopped';
  await assert.rejects(api.readSpelling(), /no longer owns/);
  state = { ...state, phase: 'running', owner_sid: undefined };
  await assert.rejects(api.readSpelling(), /no longer owns/);
});

