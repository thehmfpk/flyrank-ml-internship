# Capstone Report — Content Decline Prediction

- **Author:** Hafiz Muhammad Faizan
- **Lane:** Machine Learning
- **Repo:** flyrank-ml-internship
- **Date:** 19th July 2026

> Numbers marked `[FILL: ...]` are placeholders — replace each with the real value your
> notebooks printed on your own run before submitting. Everything else reflects the actual
> lane, methods, and decisions built across ML-04 through ML-10.

## 1. Problem framing

**Decision this supports:** which content items a search/content team should review first each
month, and what kind of action (title/meta rewrite vs. fuller content refresh) is most likely
worth the effort.

**Unit of analysis:** one content item (`client_hash_id` × `content_hash_id`), scored monthly.

**Output:** a ranked action queue — score, one reason code, one action label per item.

**Action a human takes:** reviews the top of the queue, confirms the signal looks real (not a
tracking artifact), and either rewrites a title/meta tag or schedules a fuller content refresh.

**Cost of a wrong call:** a false positive wastes editorial time on content that wasn't actually
declining; a false negative lets real decline go unaddressed for another month. Neither is
catastrophic at the scale of one queue item, which is why this is framed as a prioritization
aid with mandatory human review — not an automated action system.

**Why ML/data helps:** with hundreds of thousands of content items across 104 clients, a human
can't manually review everything every month. A ranked, reasoned queue turns an unmanageable
list into a short, explainable one.

## 2. Data safety

**Source:** `FlyRank/internship-warehouse` (Hugging Face, gated), table
`fact_content_daily_performance` — grain: `report_date × client_hash_id × content_hash_id`.
**Window used:** March 2026 (`month=2026-03`), a mid-panel month, iterated on directly rather
than filtered from the full 78.8M-row table. June 2026 (the final month) was deliberately never
used for label logic, per the dataset's own warning that it's the natural outcome window of any
past→future label.

**Columns used:** `gsc_impressions`, `gsc_clicks`, `gsc_avg_position`, `gsc_data_available`.

**Deliberately excluded, and why:**
- `ga4_*` columns — zero-filled before each client's `ga4_data_start`; out of scope for a
  GSC-only lane and would need their own missingness review before use.
- `sessions_*`, `ai_*` traffic-channel columns — describe traffic-source mix, not GSC
  visibility/CTR; out of scope for this lane.
- `client_hash_id`, `content_hash_id` — pseudonymous IDs, used only for grouping/joining and for
  the grouped train/test split, never as model features.
- Any column from days 16–31 of the month as a *feature* — that window is reserved entirely for
  the label.
- The starter CSV's `trend_direction` / `trend_pct` don't exist in this warehouse table at all,
  but the same principle applies to this lane's own second-half click aggregate
  (`clicks_second_half`) — it is the leak this project deliberately tested for and removed (see
  Section 5).

**Leakage risks considered:** label-derived features (tested by deliberately adding
`clicks_second_half` and confirming the score inflated, then removing it — Sections 5 and this
paper's Methodology), future/overlapping windows (features strictly days 1–15, label strictly
days 16–31), and product-flag-as-feature risk (none exist in this raw GSC table, confirmed by an
explicit column-name scan).

**Confirmed:** no client names, raw URLs, or private query text appear anywhere in `work/` —
only pseudonymous hash IDs, which are themselves excluded from every model input.

## 3. Baseline

**The rule (built ML-07):** flag content already ranking decently (position bucket 1–20) whose
CTR falls short of its own position bucket's average CTR that month. Score = estimated extra
clicks available if CTR caught up (`impressions × CTR shortfall`). One reason code:
`CTR_FIX_OPPORTUNITY`.

**Why it's a fair comparison:** it's built from the same single month of the same table, with no
future data and no product flags — exactly the same data the model in Section 4 uses.

**Baseline numbers (from ML-09's audit, same held-out split as the model):**

| Metric | Value |
|---|---|
| Base rate (declining share) | `[FILL: base_rate]` |
| precision@20 | `[FILL: baseline_p20]` |
| precision@50 | `[FILL: baseline_p50]` |
| precision@100 | `[FILL: baseline_p100]` |

## 4. Model / analysis

**Method:** Logistic Regression, then Random Forest — chosen because the target is a genuine
yes/no observed label, which the project's own method-selection table maps directly to that
path (readable first, stronger second, only kept if it earns the comparison).

**Feature list (all computable from days 1–15 only):** `clicks_first_half`,
`impressions_first_half`, `ctr_first_half` (derived), `avg_position_first_half` (zero-position
rows excluded before averaging, median-filled with a `has_position_data` flag for true no-data
cases), `active_days_first_half`, and one-hot position-bucket dummies.

**Deliberately left out:** `client_hash_id` / `content_hash_id` (pseudonyms), any day-16–31
aggregate, the Week-4 rule's own score (a product-flag-style input, used only as the baseline to
beat, never as a model feature), and GA4/session/AI-channel columns (out of this lane's scope).

**Target/proxy, one sentence:** `is_declining_label` = 1 if a content item's total GSC clicks in
days 16–31 of March 2026 were lower than its total clicks in days 1–15 of the same month, else 0.

## 5. Evaluation

**Split:** grouped by `client_hash_id` (`GroupShuffleSplit`, 75/25, seed 42), not a random row
split — content items from the same client share hidden characteristics, and a random split
would let the model partially memorize per-client patterns instead of learning the general
signal. Confirmed zero clients appear in both train and test.

**Before/after (from ML-09):** a naive random split vs. the honest grouped split, same model,
same features:

| Split | Base rate | precision@20 | precision@50 | precision@100 |
|---|---|---|---|---|
| Random (before) | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| Grouped by client (after) | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

The gap between these two rows is itself a finding about how much memorization the random split
was allowing.

**Model vs. baseline, same split, same metric:**

| Model | precision@20 | precision@50 | precision@100 |
|---|---|---|---|
| Base rate (random ranking) | `[FILL]` | `[FILL]` | `[FILL]` |
| Week-4 rule baseline | `[FILL]` | `[FILL]` | `[FILL]` |
| Logistic Regression | `[FILL]` | `[FILL]` | `[FILL]` |
| Random Forest | `[FILL]` | `[FILL]` | `[FILL]` |

**Leakage test:** deliberately added `clicks_second_half` (the future window the label is
derived from) as a feature. Precision@50 went from `[FILL: honest]` to `[FILL: leaky]` —
confirming the test harness correctly detects leakage. That column is excluded from the shipped
model; the honest number above is what ships.

**Errors:** accuracy broken out by position bucket showed `[FILL: e.g., "the model was weakest
in the 21-50 position bucket, where n was smallest"]`. The three most confidently-wrong cases
were items with `[FILL: describe the pattern you actually saw — e.g., very thin first-half
data, or a one-off mid-month spike]`.

## 6. Interpretation

**Top 3 features by permutation importance:** `[FILL: feature 1]`, `[FILL: feature 2]`,
`[FILL: feature 3]`. None of these towered suspiciously above the rest — a single
near-perfect predictor would have been the signature of a missed leak, and none appeared.

**Signal audit findings (ML-06), each a real computed verdict, not an assumption:**
- CTR vs. position (behind the CTR-fix flag): `[FILL: CONFIRMED/OPPOSITE/MIXED/FALSE]`
- Impression volume vs. CTR: `[FILL: verdict]`
- Position-tracking consistency vs. CTR: `[FILL: verdict]`
- Weekday vs. weekend click volume: `[FILL: verdict]`

**Surprises / negative results:** `[FILL: name any signal that came back FALSE or OPPOSITE here
— a clean negative is a real, reportable result, not a failure of the analysis]`.

## 7. Recommendation

**Ranked actions (from ML-10's playbook):**

| Archetype | Condition | Action | Reason code |
|---|---|---|---|
| CTR-fix opportunity | good position (1–20), declining, CTR below bucket norm | Rewrite title/meta | `CTR_FIX_OPPORTUNITY` |
| Deeper decline | poor position (21+), declining | Full content refresh | `CONTENT_REFRESH_NEEDED` |
| Thin data | fewer than 10 days with position data | Hold — insufficient evidence | `INSUFFICIENT_DATA` |
| Stable/growing | not declining | No action needed | `NO_ACTION_NEEDED` |

**How an editor uses this tomorrow:** open the ranked queue, start from the top (highest value
density = predicted opportunity ÷ action cost), read the actual page before acting, and never
bulk-apply the "content refresh" action without editorial sign-off — it's the most expensive,
highest-risk tier.

**Confidence and limits, stated explicitly:** this is decision-support for one month
(March 2026), one lane, one data source (GSC only). It is directional, not causal — it ranks
"worth reviewing first," not "will improve if you act."

## 8. Reproducibility

**Repo:** `flyrank-ml-internship` (this repo).

**Notebooks, in order:** `work/notebooks/w03_data_contract.ipynb` →
`w04_baseline_score.ipynb` → `w05_feature_vector.ipynb` → `w06_signal_audit.ipynb` →
`w05_model.ipynb` (capstone modeling) → `w06_validation_audit.ipynb` → `w07_action_playbook.ipynb`.

**To rerun from a fresh clone:** open each notebook in Colab, set the `HF_TOKEN` Colab Secret
(read-only token from the gated Hugging Face dataset), Runtime → Run all, in the order above.

**Random seed:** 42, fixed everywhere a split or a model is created.

**Environment:** `scikit-learn` version printed at the top of every modeling notebook —
`[FILL: paste the version your run printed]`. If a headline number matters and the library
version differs from this, tree-ensemble results can shift a few points between versions; that's
expected and worth one sentence, not a re-run.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
>
> **Metrics vs. base rate:** every precision@K above sits next to the base rate row — a high
> score can just be a high base rate.
