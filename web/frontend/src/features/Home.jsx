import { useEffect, useRef, useState } from 'react';
import Art from '../components/Art';
import CameraView from '../components/CameraView';
import SessionNotice from '../components/SessionNotice';
import { useSession } from '../hooks/useSession';
import { api, confirmAndSpeak } from '../lib/api';
import { canSpeak, speak, stopSpeaking, isSpeaking, subscribeSpeaking } from '../lib/speech';

function Row({ name, children }) {
  return <div className="trace-row"><dt>{name}</dt><dd>{children}</dd></div>;
}

function downloadJson(data, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const link = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

const KIND = { message: 'A message', hypothesis: 'A guess', clarify: 'A question' };
const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

export default function Home() {
  const { state, serverDown } = useSession('normal');
  const [pendingCamera, setPendingCamera] = useState(null);
  const [message, setMessage] = useState('');
  const [chosen, setChosen] = useState(null);
  const [working, setWorking] = useState(false);
  const [speaking, setSpeaking] = useState(isSpeaking);
  const actionLock = useRef(false);
  const mounted = useRef(true);
  const availability = useRef({});
  const chosenTimer = useRef(0);
  const messageTimer = useRef(0);
  const running = state.phase === 'running' && !serverDown;
  const camera = state.camera ?? { on: false, wanted: true };
  const wanted = pendingCamera ?? camera.wanted;
  const audio = state.audio ?? {};
  const recording = state.recording ?? { on: false, elapsed: 0, max: 30 };
  const result = state.result;
  const options = result?.options ?? [];
  const people = state.people ?? [];
  const busy = !!state.busy;
  availability.current = { running, recording: recording.on, busy };

  useEffect(() => { if (pendingCamera !== null && camera.wanted === pendingCamera) setPendingCamera(null); }, [camera.wanted, pendingCamera]);
  useEffect(() => {
    mounted.current = true;
    const unsubscribe = subscribeSpeaking(setSpeaking);
    return () => { mounted.current = false; unsubscribe(); stopSpeaking(); window.clearTimeout(chosenTimer.current); window.clearTimeout(messageTimer.current); };
  }, []);

  // A one-off message shows for a few seconds, then the live status returns.
  function flash(text) {
    setMessage(text);
    window.clearTimeout(messageTimer.current);
    messageTimer.current = window.setTimeout(() => setMessage(''), 6000);
  }

  async function act(action, body, failure) {
    if (actionLock.current || !availability.current.running) return null;
    actionLock.current = true;
    setWorking(true);
    try { return await api.normal(action, body); }
    catch { if (mounted.current) flash(failure); return null; }
    finally { actionLock.current = false; if (mounted.current) setWorking(false); }
  }

  async function toggleCamera() {
    const next = !wanted;
    setPendingCamera(next);
    flash(next ? 'Waking the mirror…' : 'The camera is off. Accuracy may now be significantly lower.');
    try { await api.setCamera(next); } catch { setPendingCamera(null); flash('Could not change the camera. Is the website server still running?'); }
  }

  const record = () => !isSpeaking() && act('record', {}, 'Could not start recording. Is the website server still running?');
  const stop = () => act('stop', {}, 'Could not stop the recording.');
  const retry = () => act('retry', {}, 'Could not retry the saved recording.');

  async function say(option, index) {
    const available = availability.current;
    if (actionLock.current || !available.running || available.recording || available.busy || isSpeaking()) return;
    actionLock.current = true;
    setWorking(true);
    try {
      const { accepted, spoken } = await confirmAndSpeak({
        option, index, clipId: result?.clip_id ?? state.clip?.clip_id, resultId: state.result_id ?? result?.result_id,
        isCurrent: () => mounted.current && availability.current.running && !availability.current.recording,
        play: speak,
      });
      if (!mounted.current) return;
      setChosen(accepted.text);
      window.clearTimeout(chosenTimer.current);
      chosenTimer.current = window.setTimeout(() => setChosen(null), 2500);
      flash(spoken ? 'Saying: ' + accepted.text : 'Choice confirmed; speech playback could not start.');
    } catch {
      if (mounted.current) flash('This choice could not be confirmed. Nothing was spoken; refresh the suggestions.');
    } finally { actionLock.current = false; if (mounted.current) setWorking(false); }
  }

  async function none() {
    flash('Noted. None of those fit. Record another turn to try again.');
    await act('reject', {}, 'Could not record that.');
  }

  async function exportTrace() {
    try {
      const trace = await api.trace();
      downloadJson(trace, `tink-trace-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`);
    } catch { flash('The trace could not be downloaded.'); }
  }

  const cameraOff = running && !camera.on;
  const heading = busy ? 'Tink is thinking…' : recording.on ? `Listening and watching · ${clock(recording.elapsed)}` : null;
  const micText = !audio.enabled ? 'Microphone off' : audio.ok ? (recording.on ? `Listening · ${audio.level_db} dB` : 'Microphone ready') : 'Microphone unavailable';

  return (
    <>
      <h1 className="sr-only" tabIndex={-1}>Home</h1>
      <SessionNotice state={state} serverDown={serverDown} />

      <div className="home-grid">
        <section className="camera-col" aria-label="Camera">
          <div className="col-head">
            <svg className="col-icon" viewBox="0 0 48 48" aria-hidden="true"><path d="M8 15h8l3-5h10l3 5h8a2 2 0 0 1 2 2v19a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V17a2 2 0 0 1 2-2Z" /><circle cx="24" cy="26" r="7.500" /></svg>
            <span className="col-label">Camera</span>
            <button type="button" role="switch" aria-checked={wanted} className="leaf-switch" onClick={toggleCamera} disabled={!running || recording.on}>
              <span className="leaf-track" aria-hidden="true"><span className="leaf-thumb" /></span>
              <span>{wanted ? 'On' : 'Off'}</span>
            </button>
          </div>
          <CameraView live={running && camera.on} label="Live camera with a box around your face"
            offTitle={running && !camera.on ? 'The camera is off' : 'Keep your face in view'}
            offText={running && !camera.on ? 'Tink is relying on the microphone.' : 'Your camera preview appears here.'} />
          <div className="camera-actions">
            {recording.on
              ? <button type="button" className="btn btn-solid" onClick={stop} disabled={!running || working}>Stop · {clock(recording.elapsed)}</button>
              : <button type="button" className="btn btn-solid" onClick={record} disabled={!running || busy || working || speaking}>{result || state.can_retry ? 'Record another turn' : 'Record'}</button>}
          </div>
          <p className="status-line" role="status" aria-live="polite">{heading || message || state.status || ''}</p>
          {(cameraOff || camera.error) && (
            <p className="note note-warn" role="alert">
              <strong>{camera.error ? 'The camera isn’t available.' : 'The camera is off.'}</strong>{' '}
              Accuracy might be significantly lower: without video, Tink can only use what it hears, so its suggestions will be less reliable.
              {camera.error && <> {camera.error}</>}
            </p>
          )}
        </section>

        <section className="voice-col" aria-label="Voice and sound">
          <div className="col-head">
            <svg className="col-icon" viewBox="0 0 48 48" aria-hidden="true"><rect x="17" y="5" width="14" height="25" rx="7" /><path d="M10 23c0 8 6 13 14 13s14-5 14-13M24 36v8M17 44h14" /></svg>
            <span className="col-label">Voice &amp; sound</span>
            <span className={audio.ok ? 'state state-on' : 'state'}>{audio.ok ? 'On' : 'Off'}</span>
          </div>
          <Art name="voice" className="voice-art" />
          <p className="voice-text">{micText}</p>
          {running && !audio.ok && <p className="note note-warn" role="status">{audio.error || 'Audio is off, so Tink will only see.'}</p>}
        </section>
      </div>

      <section className="wishes" aria-labelledby="wishes-title">
        <h2 id="wishes-title" className="section-title">{options.length ? 'Choose what to say' : 'Your three suggestions'}</h2>
        {result?.partner?.pending_question && <p className="hint" role="status">Tink thinks you were asked: “{result.partner.pending_question}” (a guess)</p>}
        {state.error_detail && (
          <p className="note note-problem" role="alert"><strong>Tink couldn’t get an answer.</strong> {state.error_detail}
            {state.can_retry && <> <button type="button" className="text-button" onClick={retry} disabled={working}>Try the same recording again</button></>}</p>
        )}
        <ul className={chosen ? 'cards has-chosen' : 'cards'}>
          {options.length > 0
            ? options.map((option, index) => {
              const isChosen = chosen === option.text;
              return (
                <li key={option.id}>
                  <button type="button" className={isChosen ? 'card chosen' : 'card'} onClick={() => say(option, index)} title={option.reason} disabled={!running || working || busy || recording.on || speaking}>
                    <Art name="sprig" className="sprig" />
                    <span className="card-text">{option.text}<small>{isChosen ? 'Confirmed' : KIND[option.kind] ?? ''}</small></span>
                    <Art name="sprig" className="sprig flip" />
                  </button>
                </li>
              );
            })
            : [0, 1, 2].map((i) => <li key={i}><div className="card card-empty" aria-hidden="true"><span className="card-text">{i === 0 && !busy ? 'Press Record' : '·'}<small>{busy ? 'Thinking…' : ''}</small></span></div></li>)}
        </ul>
        {running && !options.length && !busy && !recording.on && !state.error_detail && <p className="hint">Press Record, say or sign something, then Stop. Suggestions appear here.</p>}
        {busy && <p className="hint">Tink is studying the recording and asking Gemini…</p>}
        <div className="text-actions">
          {options.length > 0 && <button type="button" className="text-button" onClick={none} disabled={!running || working}>None of these</button>}
          {people.length > 1 && <button type="button" className="text-button" onClick={() => act('person', {}, 'Could not switch person.')}>Switch person</button>}
          {people.length > 0 && !state.owned && <button type="button" className="text-button" onClick={() => act('role', {}, 'Could not confirm.')}>That’s me</button>}
        </div>
        {!canSpeak && <p className="hint" role="status">Speech playback isn’t available in this browser. Choices are still recorded.</p>}
      </section>

      {!!state.said?.length && (
        <section className="said" aria-labelledby="said-title">
          <h2 id="said-title" className="section-title">What you said</h2>
          <ul>{state.said.map((turn, index) => <li key={index}>{turn.text}</li>)}</ul>
        </section>
      )}

      <details className="insight">
        <summary>How Tink decided</summary>
        <div className="insight-body">
          <p>Everything Tink used to reach those wishes, so each one can be traced back to what was recorded. Partner roles and transcripts are hypotheses, not facts.</p>
          <div className="trace-grid">
            <div>
              <h3>Recorded</h3>
              <dl>
                <Row name="People in view">{people.length ? `${people.length}${people.some((p) => p.selected) ? ' · communicator selected' : ''}` : camera.on ? 'none detected' : 'camera off'}</Row>
                <Row name="Communicator">{state.target ?? 'not yet'}{state.owned ? ' · confirmed' : ''}</Row>
                <Row name="Last clip">{state.clip ? `${state.clip.seconds}s · ${state.clip.frames} frames · clock ${state.clip.clock_valid ? 'ok' : 'uncertain'}${state.clip.audio_gaps ? ` · ${state.clip.audio_gaps} audio gaps` : ''}` : 'none yet'}</Row>
                <Row name="Microphone">{audio.enabled ? (audio.ok ? 'on' : audio.error || 'unavailable') : 'off'}</Row>
                {(state.warnings ?? []).map((warning) => <Row key={warning} name="Note">{warning}</Row>)}
              </dl>
            </div>
            <div>
              <h3>What Tink noticed</h3>
              {result ? (
                <>
                  <p className="quiet-detail">{result.summary}</p>
                  <dl>{(result.observations ?? []).map((text, index) => <Row key={index} name={`#${index + 1}`}>{text}</Row>)}</dl>
                </>
              ) : <p className="quiet-detail">Nothing yet</p>}
              <h3>Heard</h3>
              {result?.transcript?.length
                ? <dl>{result.transcript.map((turn, index) => <Row key={index} name={turn.role ?? 'speaker'}>{turn.text}</Row>)}</dl>
                : <p className="quiet-detail">No words recognised</p>}
            </div>
            <div>
              <h3>Context</h3>
              {result ? (
                <dl>
                  <Row name="Topic">{result.partner?.topic || 'unclear'}</Row>
                  <Row name="Partner asked">{result.partner?.pending_question || 'nothing'}</Row>
                  <Row name="Unsure about">{(result.partner?.uncertainties ?? []).join('; ') || '—'}</Row>
                </dl>
              ) : <p className="quiet-detail">Nothing yet</p>}
              <h3>Why each wish</h3>
              {options.length ? <dl>{options.map((option) => <Row key={option.id} name={option.id}>{option.reason}</Row>)}</dl> : <p className="quiet-detail">Nothing yet</p>}
              <h3>Limits</h3>
              {result?.limitations?.length ? <dl>{result.limitations.map((text, index) => <Row key={index} name="•">{text}</Row>)}</dl> : <p className="quiet-detail">None reported</p>}
            </div>
          </div>
          <div className="insight-actions"><button type="button" className="btn btn-outline" onClick={exportTrace} disabled={!running}>Download trace (JSON)</button></div>
        </div>
      </details>
    </>
  );
}
