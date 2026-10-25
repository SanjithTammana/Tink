# HackTX — Design exploration

Updated: October 24, 2026

**Historical exploration.** The latest request authorizes a React frontend; see [the current design contract](README.md). Peter Pan remains a theme. Earlier visual treatments were inspiration, and the following options were written before the user selected Tink and forest green, cream, and muted gold for the frontend foundation. Character art and further visual details remain open.

## What is confirmed

- You own UX/UI design and want help making a design document.
- You want a distinctive frontend and intend to use React when coding begins.
- This was written before frontend implementation was authorized; the current contract governs implementation.
- Your stated design target is 8 AM America/Chicago on October 24.

The project name, brand identity, colors, typography, layout, character use, illustration, copy, and screen details have not been finalized. The references' communication-assistance concept provides context; exact user needs and product flow still need confirmation.

## Inspiration library

The following descriptions identify what appears in the supplied materials. They are not instructions to adopt it.

| Reference | What we can examine | What remains open |
| --- | --- | --- |
| Dark screenshot | Forest tones, large serif headline, generous spacing, thin botanical drawing | Whether any of these belong in your design |
| Light screenshot | Warm ivory, restrained accent color, similar composition | Palette, material feel, contrast, and relationship to dark mode |
| Gilded Herbarium study | Botanical engraving and packaging-inspired restraint | Whether to use botanical imagery or this identity at all |
| Nocturne study | Restrained illumination and darker atmosphere | Whether light effects serve the product |
| Living Manuscript study | Editorial spacing and paper-like warmth | Whether the journal metaphor fits |
| Input/results prototype | One possible arrangement of conversation and output | Actual screens, interaction model, and content hierarchy |
| Name and copy in the package | `tink.`, headline, action labels, example conversation | Naming, voice, messaging, and sample scenario |

Original references are preserved under `source/Tink_Design_Handoff/`. Strong wording inside their guide describes that reference package; it does not establish your approval.

## Concept directions to explore

These concepts use the confirmed Peter Pan theme as a creative starting point. Each is a proposal, not an approved design. My recommendation for a hackathon presentation is **Neverland Field Atlas**: it gives the team a memorable visual thread that can connect the story to the communication task.

### A. Neverland Field Atlas — recommended to sketch first

A contemporary explorer's map at night: fine coastline contours, a few labeled stars, and one route that branches into several possible readings. The map is a framing device, not a claim that the system knows the correct meaning. Let the input remain the source text; place evidence, alternatives, uncertainty, and suggested replies as clear notes around it. Use one small map fragment as the visual signature, rather than putting a map behind every screen.

Why it could stand out: the forked route makes ambiguity visible and gives the Peter Pan setting a role in the interaction. Keep functional labels literal so the metaphor never obscures what the result means.

### B. The Second Star — warmer and more wondrous

Build the visual language around a night window and a single guiding star. Use a precise point of light and a hand drawn flight path as the signature. In the product, the light can mark where the user is in the flow; it must not suggest confidence or certainty in an interpretation.

Why it could stand out: the reference is immediately recognizable as a storybook journey while leaving the content calm. It may feel less distinctive if it becomes a generic starry background, so keep it sparse.

### C. Lost Things, Found Words — more editorial

Treat each conversation as a page in a collected field journal. Small annotations gather around the original words; a route line links the quote to a question the user might ask. Storybook atmosphere comes from paper, marginalia, and carefully drawn symbols rather than a large fantasy scene.

Why it could stand out: it keeps the communication task central and gives the team room to make a custom illustration and voice. It could drift away from Peter Pan unless a specific story cue is chosen.

**Recommendation:** sketch A and C with the same sample conversation. Compare which feels more like your team's idea and which explains the product fastest. Borrowing one detail from B is optional. Do not combine every motif into one screen.

## First: understand the task

Before drawing a screen, answer these together:

| Question | Current answer |
| --- | --- |
| Who is the primary user? | Open |
| What situation brings them here? | Open |
| What are they trying to accomplish? | Open |
| What input do they provide? | Open |
| What result should help them act? | Open |
| What does the team actually intend to implement? | Open |

A useful one-sentence brief: **A person in [situation] needs to [task], so they can [outcome].**

## Then: choose what feels like you

For each reference, record a specific detail you like, a detail you dislike, and why. A preference for one detail does not approve the entire image.

| Reference/detail | Keep exploring | Avoid | Reason |
| --- | --- | --- | --- |
| Typography | Open | Open | Open |
| Color and light | Open | Open | Open |
| Composition and spacing | Open | Open | Open |
| Illustration or graphic motif | Open | Open | Open |
| Controls and surfaces | Open | Open | Open |
| Language and tone | Open | Open | Open |

Choose three feeling words in your own language. Record what each means visually. For example, “calm” could mean fewer competing elements; “Peter Pan themed” could lean toward maps, shadows, flight, or storybook lettering. You decide which cues feel right.

## Screen planning — after the task is clear

Do not assume the final product needs the supplied landing hero, two-column workspace, or settings panel. Start with the user's task and establish the smallest useful set of screens or states.

For each proposed screen, document:

- Its purpose and the information needed first.
- The main action and what happens afterward.
- What the user needs to return, edit, or recover from a problem.
- Empty, working, error, and completed states that match actual functionality.
- Mobile behavior, readable text, keyboard/focus behavior, and accessibility considerations.

No screen map, route structure, layout, or component hierarchy is approved yet.

## Make the design distinctive through decisions

Pick a small number of details you can explain in relation to the product: a composition, type treatment, interaction, illustration, or voice. Sketch alternatives before selecting one. A custom filament is one possible idea from the references, not an assignment or requirement.

Compare alternatives using the same sample content. This helps distinguish a useful design choice from an attractive image that does not serve the task.

## Decision log

Only record a choice as approved after you explicitly select it. Reference wording and assistant suggestions are not approval.

| Area | Options under consideration | Your choice | Reason/date |
| --- | --- | --- | --- |
| Name | Supplied name or alternatives | Undecided | — |
| Visual character | To explore together | Undecided | — |
| Palette | To explore together | Undecided | — |
| Typography | To explore together | Undecided | — |
| Layout and screen map | Depends on the user task | Undecided | — |
| Motif or illustration | Neverland map / second star / field journal / your idea | Undecided | — |
| Interaction and sample | Depends on intended functionality | Undecided | — |

## Later design-document contents

As choices become clear, this same document can grow to include the product brief, selected references with reasons, screen sketches, final copy, visual styles, responsive behavior, interaction states, accessibility notes, and developer handoff details.

For now, the next step is to choose a concept to sketch or combine it with a detail you prefer. Frontend code now exists under `../frontend/`; these concepts remain reference options.
