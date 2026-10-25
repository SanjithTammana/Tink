import { useCallback, useEffect, useState } from 'react';

// 'light' | 'dark' | 'system'. The page's data-theme is always light or dark; "system" follows the device.
const KEY = 'tink-theme';
const read = () => { try { return localStorage.getItem(KEY) || 'system'; } catch { return 'system'; } };

export function useTheme() {
  const [choice, setChoice] = useState(read);

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)');
    const apply = () => {
      const resolved = choice === 'system' ? (media?.matches ? 'dark' : 'light') : choice;
      document.documentElement.dataset.theme = resolved;
      document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#0b2620' : '#f3ecd6');
    };
    apply();
    if (choice !== 'system' || !media) return undefined;
    media.addEventListener?.('change', apply);
    return () => media.removeEventListener?.('change', apply);
  }, [choice]);

  const choose = useCallback((next) => {
    setChoice(next);
    try { localStorage.setItem(KEY, next); } catch { /* the choice just won't be remembered */ }
  }, []);

  return { choice, choose };
}
