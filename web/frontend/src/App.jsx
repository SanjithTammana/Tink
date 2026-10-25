import { useCallback, useEffect, useState } from 'react';
import AppShell from './components/AppShell';
import Asl from './features/Asl';
import Home from './features/Home';
import Intro from './features/Intro';

function useReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false);
  useEffect(() => {
    const media = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!media) return undefined;
    const onChange = (event) => setReduced(event.matches);
    media.addEventListener?.('change', onChange);
    return () => media.removeEventListener?.('change', onChange);
  }, []);
  return reduced;
}

export default function App() {
  const [view, setView] = useState('home');
  const [introActive, setIntroActive] = useState(true);
  const reducedMotion = useReducedMotion();
  const finishIntro = useCallback(() => setIntroActive(false), []);

  useEffect(() => {
    window.scrollTo(0, 0);
    if (!introActive) document.querySelector('#main-content h1')?.focus({ preventScroll: true });
  }, [view, introActive]);

  // The intro plays once over the Home page and lifts away to reveal it; the camera starts behind it.
  return (
    <>
      <AppShell view={view} onNavigate={setView} inert={introActive}>
        {view === 'home' && <Home />}
        {view === 'asl' && <Asl />}
      </AppShell>
      {introActive && <Intro onComplete={finishIntro} reducedMotion={reducedMotion} />}
    </>
  );
}
