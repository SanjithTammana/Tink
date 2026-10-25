import { api } from './api.js';

export const canSpeak = typeof window !== 'undefined' && 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
let current = null, pending = Promise.resolve(), guardUntil = 0, guardTimer;
const listeners = new Set();
export const isSpeaking = () => current !== null || Date.now() < guardUntil;
const publish = () => listeners.forEach((listener) => listener(isSpeaking()));
export function subscribeSpeaking(listener) { listeners.add(listener); listener(isSpeaking()); return () => listeners.delete(listener); }

function release(token) {
  clearInterval(token.renew);
  const hadLease = token.reserved;
  if (!hadLease && !token.spoken) return;
  token.reserved = false;
  token.spoken = false;
  guardUntil = Date.now() + 700;
  clearTimeout(guardTimer);
  guardTimer = setTimeout(publish, 710);
  if (hadLease) api.normal('playback', { active: false }).catch(() => {});
}

function finish(token) {
  release(token);
  if (current === token) current = null;
  publish();
}

export function stopSpeaking() {
  const token = current;
  current = null;
  if (token) finish(token);
  if (canSpeak) window.speechSynthesis.cancel();
  publish();
}

// ASL has no microphone capture. Normal speech first reserves the server echo guard.
export function speak(text, { guardRecording = true } = {}) {
  if (!canSpeak || !text?.trim()) return Promise.resolve(false);
  stopSpeaking();
  const token = { reserved: false };
  current = token;
  publish();
  pending = pending.catch(() => false).then(async () => {
    if (current !== token) return false;
    try {
      if (guardRecording) {
        const reservation = await api.normal('playback', { active: true });
        if (reservation?.ok !== true) throw new Error('Playback guard unavailable.');
        token.reserved = true;
      }
      if (current !== token) { finish(token); return false; }
      const utterance = new window.SpeechSynthesisUtterance(text.trim());
      utterance.onend = utterance.onerror = () => finish(token);
      if (guardRecording) token.renew = setInterval(() => {
        if (current !== token) return;
        api.normal('playback', { active: true }).then((answer) => {
          if (answer?.ok !== true && current === token) stopSpeaking();
        }, () => { if (current === token) stopSpeaking(); });
      }, 2000);
      token.spoken = true;
      window.speechSynthesis.speak(utterance);
      return true;
    } catch { finish(token); return false; }
  });
  return pending;
}

