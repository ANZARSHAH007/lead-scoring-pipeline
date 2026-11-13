# Page — Model card

Overrides `MASTER.md` for this page only. Built in T-601. Read-only.

**Job:** let a sceptic — a sales manager, or a reviewer of this portfolio — decide whether to believe the scores. Every figure is traceable to one real training run, and the page recomputes nothing.

## Order, and why it is this order

1. **Model version + training date** — so a screenshot is attributable.
2. **Results table** — all five rows from doc 02, including the labelled leakage row. The sceptic's first question is "compared to what?", so the baselines come before any chart.
3. **Reliability curve** — the second question is "does 70 mean 70%?".
4. **Cumulative gain curve** — with **the 20% mark annotated directly on the line**, because Capture@20% is the headline and it should be readable off the chart, not reconstructed.
5. **Score distribution by outcome** — two overlaid histograms, directly labelled.
6. **Feature importance** — horizontal bar, top 10, sorted, single colour.
7. **Grade bands** — the actual cut-offs, read from `/model-info`.
8. **What this model must not be used for** — last, never collapsed.

## Chart styling

All figures follow doc 08: tokens from `MASTER.md`, model line `#1E40AF`, comparison series `#D97706`, reference lines `#475569` dashed, axes labelled with units, one-line caption under each saying what it shows. No dual axes, no 3D, no gradients, no pie charts. Equal axes and 0–1 range on the reliability curve. Load the `dataviz` skill before writing the first figure.

Every chart's numbers also appear as text or a small table on this page, so the content does not depend on seeing the image.

## Results table

Fira Code numbers, right-aligned, 3 decimal places for metrics and one decimal for percentages. The shipped model's row is the only one with a `--color-card` highlight and a bold label. The leakage row is visually last and carries an inline caption — *"uses columns a rep fills in after contact; not shippable"* — in `muted-foreground`, not in red. It is an honest comparison, not a warning.

## Limits section

Plain prose, no bullets fewer than a full sentence, drawn from `model_card.json.limits`:

- it ranks leads; it does not decide whether to contact one
- trained on X Education web leads; untested on any other population
- excludes every column a rep fills in after contact
- `City`, `Country` and occupation are inputs, and no fairness audit has been run — stated, not glossed

## States

| State | Screen |
| --- | --- |
| Loaded | the eight sections above |
| `/model-info` unavailable | "model metrics unavailable" with the reason, **not** blank panels or zeros |
| A metric is `null` in the card | that row shows "not measured" — never a fabricated number |

## Do not

- No "accuracy: 94%" headline. Accuracy on a 38.5%-positive target is a misleading number and this page exists to not mislead.
- No hiding the leakage row, and no hiding the limits behind an expander.
- No metric on this page that did not come from a logged run.
