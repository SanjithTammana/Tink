import { useEffect, useState } from 'react';
import { startSessionLoop } from '../lib/api';

const CONNECTING = { phase: 'connecting' };
export function useSession(mode, intervalMs = 250) {
  const [state, setState] = useState(CONNECTING);
  const [serverDown, setServerDown] = useState(false);
  useEffect(() => {
    setState(CONNECTING);
    return startSessionLoop(mode, { onState: setState, onConnection: setServerDown, intervalMs });
  }, [mode, intervalMs]);
  return { state, serverDown, setState };
}

