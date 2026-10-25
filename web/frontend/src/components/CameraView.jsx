import { useMemo } from 'react';
import Art from './Art';

// The camera, seen through a wreath. The live stream sits in the oval; the wreath artwork is drawn over its edge.
export default function CameraView({ live, label, offTitle = 'Keep your face in view', offText = 'Your camera preview appears here.' }) {
  const src = useMemo(() => `/api/stream?k=${Date.now()}`, [live]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className={live ? 'wreath wreath-live' : 'wreath'}>
      <div className="wreath-view">
        {live
          ? <img key={src} src={src} alt={label} className="camera-stream" />
          : (
            <div className="wreath-empty">
              <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M8 15h8l3-5h10l3 5h8a2 2 0 0 1 2 2v19a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V17a2 2 0 0 1 2-2Z" /><circle cx="24" cy="26" r="7.5" /></svg>
              <strong>{offTitle}</strong>
              {offText && <span>{offText}</span>}
            </div>
          )}
      </div>
      <Art name="wreath" className="wreath-art" />
    </div>
  );
}
