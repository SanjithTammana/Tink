# Archived Tink cinematic intro exploration

Status: **superseded by the minimal wordmark intro implemented in the frontend.** This file preserves the earlier ornate exploration and asset analysis as historical context. The current direction and acceptance criteria are in [`README.md`](README.md). The supplied Disney screenshots were references, not a license for website reuse or animation; they are not shipped. The earlier concept images in [`concepts/`](concepts/) are not production assets.

## Visual idea

An illuminated page from a botanical field journal becomes the word **TINK**. T, N, and K are readable, custom branch-letter engravings. An original fairy occupies the I. She has dark auburn hair, warm brown skin, a long ivory-and-moss embroidered dress, and amber moth-like wings. This avoids the blonde bun, green petal dress, and wing design of the supplied character references. The first scene may be lush; the destination remains the restrained Gilded Herbarium page. One thin gold flight filament makes the visual handoff.

The generated [character study](concepts/original-fairy-study.png) and [wordmark keyframe](concepts/tink-intro-keyframe.png) were made with the built-in imagegen tool from this direction. The keyframe is a single raster image; it cannot independently grow letters or animate her body parts. It is not a substitute for the deliverables below.

## Sequence and timing

| Time | Scene | Motion and continuity |
| --- | --- | --- |
| 0–1.7 s | Reveal | Matte green woodland depth emerges. T, N, K grow along authored masks from trunk to fine branch tips, one after another. A soft gold pool exposes the fairy as the I. Keep the word legible throughout. |
| 1.7–3.6 s | Alive | Pause to read TINK. Wings breathe once, eyes glance, head tilts, then the fairy notices her place in the word. She turns through a graceful full spin; a single fine spiral follows her hands and wings. |
| 3.6–5.8 s | Flight | She leaves the I's position and arcs toward the upper right. Her narrow dust trail traces the same visual language as the landing page's botanical filament. It acts as a moving reveal mask for the cream or dark landing page beneath, depending on the saved theme. The green scene recedes behind the reveal. |
| 5.8–6.3 s | Home | The existing landing page is fully interactive. The final point of light resolves near its botanical illustration, then rests. Focus moves to the page heading only when the sequence completes or is skipped. |

The dust is a sparse sequence of authored glints that follows the flight path, never a random particle explosion. Use depth, restrained light and ease curves rather than screen-wide glow. On a phone, scale and recompose the word so TINK stays readable, then shorten the flight path to the landing illustration's position.

## Production assets required before animation code

1. **One original, consistent fairy rig.** Preferred: a transparent `.riv` file with independently controlled head/eyes, hair, torso/arms, skirt, and four wing parts. Include named idle, look, wing, spin and flight states; define anchor points and exact transition events. Keep the character's face and costume identical in every state. If Rive would flatten the painterly art too much, supply rendered transparent animation in browser-ready formats instead, with matching mobile framing and a still poster.
2. **Three separate T, N, K letter assets.** Deliver editable vector paths or layered illustration with path/mask order for branch growth, isolated alpha, and a shared baseline. A flat TINK image cannot produce the requested growth or let the fairy fly out of the I.
3. **A small scene package.** Two or three layered forest-depth planes, one gold light pass, and an authored spiral/flight path. Export at desktop and phone aspect ratios, with enough margin for responsive cropping. Avoid baking the landing page into the animation.
4. **A motion reference render.** A short low-resolution animatic showing wing motion, head glance, spin, flight, and the landing-page reveal. This lets us check timing and character consistency before wiring the site.

Artwork should be team-owned or otherwise cleared for use. The Disney images supplied with the brief are inspiration for personality and story context only; their character design is not a production asset.

## Frontend integration plan

React owns the intro overlay and exposes **Skip intro** immediately, **Replay intro** on the landing page, and Escape-to-skip. GSAP coordinates the timed letter masks, flight path, and reveal mask; Rive drives character motion if the `.riv` rig is supplied. The animation must never block the existing workspace after completion or skip. It plays on each landing-page load, not on internal tool navigation. Pause or remove hidden animation resources when the intro ends.

For `prefers-reduced-motion` or Tink's reduced-motion setting, show the landing page directly with a static, accessible Tink wordmark; do not require waiting through the sequence. High-contrast mode must preserve readable controls and avoid using light alone to convey the transition. The overlay has a visible focusable skip button, and the underlying page is unavailable to keyboard focus until the overlay is dismissed. Keep any introductory sound absent by default. Test at 390, 768, and 1440 px, with keyboard, screen reader, and slower devices. Use a low-memory still fallback if animation loading fails.

## Acceptance for implementation

The I reads as the fairy at a glance; TINK remains readable during formation; character features remain consistent through look, spin and flight; wing motion is independent of the body; the dust trail reveals the real landing page without a black cut; skip, replay, reduced motion and focus behavior work. Compare renders against the supplied keyframe and the actual landing page in a browser before calling the intro finished.
