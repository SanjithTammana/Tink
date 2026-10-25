import { useEffect, useRef, useState } from 'react';
import { useTheme } from '../hooks/useTheme';
import Art from './Art';

const icons = {
  home: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 11 12 3.8 20.5 11M6 9.5V20h4.5v-6h3v6H18V9.5" /></svg>,
  asl: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 12V5.5a1.5 1.5 0 0 1 3 0V11m0-1.5V4a1.5 1.5 0 0 1 3 0v6m0-4.5a1.5 1.5 0 0 1 3 0V13m0-3a1.5 1.5 0 0 1 3 0v4.5c0 4-2.5 6.5-6.500 6.500C9.500 20 8.500 18.500 6 15l-1.700-2.700a1.500 1.500 0 0 1 2.500-1.600L8 12.500" /></svg>,
  gear: <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.200" /><path d="M12 3v2.200M12 18.800V21M3 12h2.200M18.800 12H21M5.600 5.600l1.500 1.500M16.900 16.900l1.500 1.500M18.400 5.600l-1.500 1.500M7.100 16.900l-1.500 1.500" /></svg>,
};

const THEMES = [['light', 'Light'], ['dark', 'Dark'], ['system', 'Match my device']];

function Settings() {
  const { choice, choose } = useTheme();
  const [open, setOpen] = useState(false);
  const box = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => { if (event.type === 'keydown' ? event.key === 'Escape' : !box.current?.contains(event.target)) setOpen(false); };
    window.addEventListener('pointerdown', close);
    window.addEventListener('keydown', close);
    return () => { window.removeEventListener('pointerdown', close); window.removeEventListener('keydown', close); };
  }, [open]);
  return (
    <div className="settings" ref={box}>
      <button type="button" className="nav-item" aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(!open)}>{icons.gear}<span>Settings</span></button>
      {open && (
        <div className="settings-menu" role="group" aria-label="Appearance">
          <p>Appearance</p>
          {THEMES.map(([id, label]) => (
            <button key={id} type="button" role="radio" aria-checked={choice === id} className="theme-option" onClick={() => choose(id)}>{label}</button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function AppShell({ view, onNavigate, inert, children }) {
  return (
    <div className="shell" inert={inert}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="bar">
        <button type="button" className="wordmark" onClick={() => onNavigate('home')} aria-label="Tink, home"><Art name="wordmark" /></button>
        <nav className="nav" aria-label="Pages">
          <button type="button" className="nav-item" aria-current={view === 'home' ? 'page' : undefined} onClick={() => onNavigate('home')}>{icons.home}<span>Home</span></button>
          <button type="button" className="nav-item" aria-current={view === 'asl' ? 'page' : undefined} onClick={() => onNavigate('asl')}>{icons.asl}<span>ASL</span></button>
          <Settings />
        </nav>
      </header>
      <main id="main-content" className="stage" tabIndex={-1} key={view}>{children}</main>
    </div>
  );
}
