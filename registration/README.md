# Registered analysis code and what changed since registration

The human audit was registered on OSF Registries on 2026-09-23 (10:01 UTC), before any rating. The paper's
appendix gives an anonymized view-only link to the registration, its two updates and its attachments.

* `registered_anno_analysis.py` — the analysis script exactly as attached to the registration (attachment 02;
  SHA-256 `3eb0e52435ca488e…`, matching the registration's `SHA256SUMS`). Its comments are in Korean, as
  registered.
* `registered_vs_current.diff` — the functional difference between that script and `scripts/anno_analysis.py`,
  computed on the parsed code with comments and docstrings removed (so translated comments do not appear).

What the diff contains:

| Change | Effect on any estimate |
| --- | --- |
| Restrict the analysis to the shared subset of 220 main items (`outputs/anno/subset_v2.json`); count, but never read, ratings outside it | Yes — this is the sample reduction filed as a registration update on 2026-10-06 and described in the paper |
| Drop repeated rows of the same persona in the model judge's bank file (keep the last) | None — the bank file has no repeated rows (44,821 rows, 44,821 personas) |
| Refuse the bank-wide prediction-powered estimate until every bank persona has been judged | None — all 44,821 bank personas are judged |
| English log and help messages; a claim label that prints the actual number of planted items | None |

Claims, thresholds, tests, the Holm family and the handling of unresolved disagreements are unchanged.
