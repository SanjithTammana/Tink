# Tink frontend

React and Vite frontend for Tink's communication workspace. Run from this folder:

```sh
npm install
npm run dev
npm run build
npm test
```

The local development URL is printed by Vite. The backend is not required to run the frontend.

## What works now

- Open on a brief procedural intro: the supplied silhouette sprite starts as the I in Tink, turns through standing frames, then flies along authored Bézier curves while a clean i appears in her place. Then enter the responsive Gilded Herbarium landing page. Skip the intro or replay it from the page header; reduced-motion settings show the completed static mark with the fairy as I.
- Compose text and read it aloud with the browser's speech synthesis; change rate or stop it.
- Speak one of three starter phrases, then add, edit, or remove phrases stored on this device.
- Use browser speech recognition where available. Interim words appear live; stop to edit, copy, or explicitly save a transcript in this browser.
- Start/stop a local camera preview with browser permission prompts.
- Read a sample conversation and fixed example clarification phrases. Locally saved transcripts are listed separately.
- Set system/light/dark appearance, high contrast, reduced motion, and text size.

Speech recognition depends on browser support and may use the browser provider's speech service. Camera access needs a secure context such as localhost or HTTPS. No camera frames are sent or stored. Saved transcripts and phrases use localStorage and are specific to this browser; users can delete transcripts in Conversation. The fixed conversation suggestions are examples, not personalized analysis.

See [INTEGRATION.md](INTEGRATION.md) for the backend handoff. The visual direction and accessibility target are in [`../design/README.md`](../design/README.md). WCAG 2.2 AA remains a target requiring manual and assistive technology testing.

The earlier ornate intro exploration is archived in [`../design/INTRO_ANIMATION.md`](../design/INTRO_ANIMATION.md). The implemented intro uses an original small fairy mark; it does not use the supplied Disney screenshots as production assets.
