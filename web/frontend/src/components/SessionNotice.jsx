// Explains, in plain words, why a page has no live data yet: server not running, models loading, or an error.
export default function SessionNotice({ state, serverDown }) {
  if (serverDown) {
    return (
      <div className="notices">
        <div className="whisper whisper-problem" role="alert">
          <strong>Tink can’t reach its website server.</strong> Start it in a terminal and this page connects by itself:
          <code className="command">cd Tink-web &amp;&amp; python start.py</code>
        </div>
      </div>
    );
  }
  if (state.phase === 'error') {
    return (
      <div className="notices">
        <div className="whisper whisper-problem" role="alert">
          <strong>This page couldn’t start.</strong> {state.error}
          <br />{/missing/i.test(state.error ?? '')
            ? <>A model file hasn’t been downloaded yet. Prepare the models using <code>docs/SESSION-SETUP.md</code>, then reload.</>
            : 'Check that the camera is free and the backend is set up (its setup script, run once), then reload.'}
          {state.engine?.log && <details><summary>Engine log</summary><pre className="engine-log">{state.engine.log}</pre></details>}
        </div>
      </div>
    );
  }
  const turnedOff = state.engine?.turned_off ?? [];
  const retried = turnedOff.length > 0 && (
    <p className="whisper" role="status">
      Tink’s engine stumbled while loading, so it restarted without {turnedOff.join(' and ')}. Everything else works; accuracy may be a little lower.
    </p>
  );
  if (state.phase !== 'running') {
    return (
      <div className="notices">
        {retried}
        <p className="whisper" role="status">
          Gathering pixie dust… the first run can take a few minutes while Tink’s models load. The camera wakes when they are ready.
        </p>
      </div>
    );
  }
  return retried ? <div className="notices">{retried}</div> : null;
}
