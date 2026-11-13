# Page — Rank leads

Overrides `MASTER.md` for this page only. Default landing page. Built in T-601.

**Job:** a rep uploads a CSV and leaves with a call list. The table is the hero; everything else orients.

## Layout, top to bottom

1. **Purpose line** — one sentence, `muted-foreground`, no heading above it.
2. **Upload control** — the only primary action on the page, in accent `#D97706`.
3. **Summary tiles** (only once results exist) — four, one row: `Leads scored` · `Grade A` · `Mean score` · `Model version`. Numbers in Fira Code, label beneath in `muted-foreground` at the smaller size. No icons, no sparklines, no coloured tile backgrounds — these orient, they are not the content.
4. **Results table** — full width, `--shadow-sm`, border `#DBEAFE`.
5. **Download CSV** — secondary styling, directly under the table, never floating.

## Table rules

| Rule | Value |
| --- | --- |
| Sort | `score` descending, always; other columns sortable |
| Column order | `score` · grade chip · `prospect_id` · reason 1–3 · then input columns |
| Row padding | `--space-md` (8px) — density 8/10, a rep scans 200 rows |
| Zebra | `--color-muted` on alternate rows |
| Header | sticky, Fira Code, `--color-card` background, bottom border |
| Numbers | Fira Code, tabular figures, right-aligned |
| Text | Fira Sans, left-aligned |
| Overflow | horizontal scroll **inside the table wrapper**; never on the page |
| Row height | fixed — no wrapping inside cells; truncate reason text with a title attribute |
| Hover | row background to `--color-muted`, 150 ms |

Grade chips exactly as `MASTER.md` and doc 08 define: A primary/white, B accent/black, C muted/muted-foreground, **each carrying its letter**.

## States

| State | Screen |
| --- | --- |
| Empty (default) | what to upload, the required column names as a list, and a **sample CSV download**. No bare upload box. |
| Loading | progress with "scoring N leads…", layout height held so nothing jumps |
| Success | tiles + table + download |
| Bad file — missing column | inline error naming the missing column; the file stays in session |
| Over the row cap | offer to score the first 1,000 and say so in plain words |
| API down | "the scoring service is not responding", a retry action, upload preserved |
| Header-only / empty file | the empty state, unchanged — not an error |

## Do not

- No modal dialogs, no toasts, no confetti, no animated counters on the tiles.
- No colour-only grade encoding.
- No red anywhere — nothing on this page is an error state except the error states.
- No second primary action competing with the upload.
