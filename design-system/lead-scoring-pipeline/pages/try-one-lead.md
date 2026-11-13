# Page — Try one lead

Overrides `MASTER.md` for this page only. Built in T-601. **P2 on the cut list** — dropped before Rank leads or Model card.

**Job:** build trust. A rep changes one field, resubmits, and watches the score move in the direction the reasons said it would. That is the whole point of the page.

## Layout

Single column, max width ~640px, centred (variance 3/10 — centred and minimal). Form above, result panel below. **Never a dialog**: the inputs must stay visible beside the result so a value can be changed and compared.

## Field order and labelling

Order follows how a rep learns about a lead:

1. **Where it came from** — `lead_origin`, `lead_source`
2. **What they did** — `total_visits`, `total_time_spent_on_website`, `page_views_per_visit`
3. **Who they are** — `specialization`, `current_occupation`, `what_matters_most`, `city`, `country`
4. **Preferences** — `do_not_email`, `do_not_call`

Rules:

- Labels in **business language with units**: "Time on website (seconds)", not `total_time_spent_on_website`.
- Every field **prefilled** with the dataset median (numeric) or mode (categorical). The form is never blank, so a first-time visitor can submit immediately.
- Only admissible features appear. No field exists on this form that the model is not allowed to use — the form is itself a statement of the leakage rule.
- Numeric inputs enforce `≥ 0` client-side and still show the server's 422 message if it arrives.
- Submit button in accent `#D97706`, the single primary action.

## Result panel

Appears below the form, `--color-card`, `--shadow-md`, `--space-lg` padding:

- **Score** large, Fira Code, with the grade chip beside it.
- **Probability** as a percentage underneath in `muted-foreground` — "82 / 100 · about an 82% chance".
- **Three reasons** as three rows: an up/down arrow, the plain-English phrase, and the feature name in `muted-foreground` Fira Code. Direction carried by the **arrow and the word**, never by colour alone.
- Model version in small text at the panel's foot.

## States

| State | Screen |
| --- | --- |
| Initial | prefilled form, no result panel, one line explaining what the page does |
| Submitting | inline spinner on the button, form stays interactive-looking but disabled |
| Result | the panel above |
| Field invalid | message under that field in `--color-destructive`; all other entered values kept |
| API down | the same service-down message as Rank leads; form values kept |

## Do not

- No live scoring on every keystroke (it hides cause and effect, and hammers the API). Submit is explicit.
- No gauge, dial, speedometer or progress-ring for the score — a number and a grade chip.
- No colour-only direction indicators.
