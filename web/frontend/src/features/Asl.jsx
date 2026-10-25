import { useEffect, useRef, useState } from 'react';
import Art from '../components/Art';
import CameraView from '../components/CameraView';
import SessionNotice from '../components/SessionNotice';
import { useSession } from '../hooks/useSession';
import { api, aslShortcut } from '../lib/api';
import { canSpeak, speak, stopSpeaking } from '../lib/speech';

const LIST_SIZE = 8;
const COMMIT_EVERY_MS = 450;   // the list reorders at most this often, so a button never moves under your cursor
const FADE_AFTER_MS = 1500;    // after the hand leaves, the last list stays (dimmed) so you can still pick from it
const CLEAR_AFTER_MS = 8000;

export default function Asl() {
  const { state, serverDown } = useSession('asl', 150);
  const [message, setMessage] = useState('');
  const [letters, setLetters] = useState([]);
  const [stale, setStale] = useState(false);
  const lastSeen = useRef(0);
  const lastCommit = useRef(0);
  const messageTimer = useRef(0);
  const speechLock = useRef(false);
  const mounted = useRef(true);
  const running = state.phase === 'running' && !serverDown;
  const text = state.message ?? '';
  const live = useRef({});
  live.current = { running, text, letters };

  // The ranked possibilities from the engine, steadied for clicking.
  useEffect(() => {
    const next = (state.ranked ?? []).slice(0, LIST_SIZE).map((item) => item.label);
    if (!next.length) return;
    const now = performance.now();
    lastSeen.current = now;
    setStale(false);
    if (now - lastCommit.current >= COMMIT_EVERY_MS || live.current.letters.length === 0) {
      lastCommit.current = now;
      setLetters((previous) => (previous.join() === next.join() ? previous : next));
    }
  }, [state.ranked]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const quiet = performance.now() - lastSeen.current;
      if (quiet > CLEAR_AFTER_MS) setLetters((previous) => (previous.length ? [] : previous));
      setStale(quiet > FADE_AFTER_MS);
    }, 400);
    return () => window.clearInterval(timer);
  }, []);

  // A one-off message shows for a few seconds, then the live status (e.g. "Show exactly one hand") returns.
  function flash(text) {
    setMessage(text);
    window.clearTimeout(messageTimer.current);
    messageTimer.current = window.setTimeout(() => setMessage(''), 3000);
  }

  async function add(letter) {
    try {
      const result = await api.asl('add', { letter });
      flash(result.added ? `Added ${result.added}` : 'That letter could not be added.');
      if (result.added) setStale(true);   // the engine starts reading the next letter fresh
    } catch { flash('Could not add that letter. Is the ASL page still running?'); }
  }

  async function act(action) {
    try { await api.asl(action); } catch { flash('That did not go through. Is the ASL page still running?'); }
  }

  async function speakText() {
    if (speechLock.current || !live.current.running) return;
    speechLock.current = true;
    try {
      const value = await api.readSpelling();
      if (!mounted.current) return;
      if (!value.trim()) { flash('Spell something first.'); return; }
      const spoken = await speak(value, { guardRecording: false });
      if (mounted.current) flash(spoken ? 'Speaking.' : 'Speech playback is unavailable.');
    } catch { if (mounted.current) flash('Could not check the current spelling. Nothing was spoken.'); }
    finally { speechLock.current = false; }
  }

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; stopSpeaking(); window.clearTimeout(messageTimer.current); };
  }, []);

  // Keys: 1-8 add that letter from the list; Space, D, C, S as in the desktop app.
  useEffect(() => {
    function onKey(event) {
      if (!live.current.running) return;
      const command = aslShortcut(event);
      if (!command) return;
      event.preventDefault();
      if (command.action === 'add') {
        const letter = live.current.letters[command.index];
        if (letter) add(letter);
      } else if (command.action === 'speak') speakText();
      else act(command.action);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <>
      <h1 className="sr-only" tabIndex={-1}>ASL</h1>
      <SessionNotice state={state} serverDown={serverDown} />

      <div className="asl-grid">
        <section className="camera-col" aria-label="Camera">
          <CameraView live={running} label="Live camera with your hand outlined" offTitle="Keep one hand in view" offText="Your camera preview appears here." />
          <p className="status-line" role="status" aria-live="polite">{message || state.status || ''}</p>
        </section>

        <section className="panel letters-panel" aria-labelledby="letters-title">
          <Art name="sprig" className="sprig sprig-corner" />
          <Art name="sprig" className="sprig sprig-corner flip" />
          <h2 id="letters-title" className="panel-title">Possible letters</h2>
          <p className="panel-sub">{letters.length ? 'Choose a letter to add to your message.' : 'Show one hand to the camera. Likely letters appear here.'}</p>
          <p className="rule-label"><span>Number keys 1 to 8 work too</span></p>
          {letters.length > 0 && (
            <ul className={stale ? 'tiles stale' : 'tiles'} aria-label="Possible letters, most likely first">
              {letters.map((letter, index) => (
                <li key={letter}>
                  <button type="button" className={index === 0 ? 'tile tile-hot' : 'tile'} onClick={() => add(letter)} aria-label={`Add ${letter}`}>
                    <span className="tile-letter">{letter}</span><small aria-hidden="true">{index === 0 ? 'Likeliest' : index + 1}</small>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="panel message-panel" aria-labelledby="message-title">
        <h2 id="message-title" className="panel-title small">Your message</h2>
        <div className="message-field">
          <Art name="sprig" className="sprig sprig-small" />
          <p className="message-text" aria-live="polite">{text ? <>{text}<span className="caret" aria-hidden="true" /></> : <span className="placeholder">Letters you add appear here<span className="caret" aria-hidden="true" /></span>}</p>
          <Art name="sprig" className="sprig sprig-small flip" />
        </div>
        <div className="message-actions">
          <button type="button" className="btn btn-outline" onClick={() => letters[0] && add(letters[0])} disabled={!running || !letters.length}>Add letter</button>
          <button type="button" className="btn btn-outline" onClick={() => act('space')} disabled={!running}><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="8" width="18" height="8" rx="2" /><path d="M7 13h10" /></svg>Space</button>
          <button type="button" className="btn btn-outline" onClick={() => act('delete')} disabled={!running || !text}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 7 4 12l5 5M4 12h11a5 5 0 0 1 5 5" /></svg>Delete last</button>
          <button type="button" className="btn btn-outline" onClick={() => act('clear')} disabled={!running || !text}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13" /></svg>Clear</button>
          <button type="button" className="btn btn-solid" onClick={() => speakText(text)} disabled={!canSpeak || !text.trim()}>Speak</button>
        </div>
        <p className="fine"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.500v.1" /></svg>Use Space between words. This reads fingerspelled letters one at a time; it does not translate signed language.</p>
      </section>
    </>
  );
}
