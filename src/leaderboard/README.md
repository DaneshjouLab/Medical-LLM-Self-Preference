# Self-Preference Leaderboard

Open `index.html` directly in a browser or serve this directory with any static
file server. The page has no build step and no JavaScript dependency.

## Result sources

- `docs/drafts/sep_3_pasted_revised.tex` supplies the current position-adjusted
  answer-rank self-preference results, aggregate generator performance, and
  experiment counts.
- `data/analysis/self_preference/` contains the generated model-, pairwise-, and
  question-level secondary score artifacts.

Charts are inline SVG and can be edited without adding a JavaScript dependency.
Shared colors, fonts, and dimensions are defined as custom properties at the top
of `styles.css`.

The page separates blinded single-turn results, blinded multi-turn results, and
the paired model-unblinded comparison into independent anchor sections.

The Google Fonts import is optional; system font fallbacks are already included.
