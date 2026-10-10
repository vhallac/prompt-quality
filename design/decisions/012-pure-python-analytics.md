# 012 — Pure-Python analytics (no numpy/scipy)

**Status:** accepted

## Context
The baseline stage (C3) needs AUC computation (Mann-Whitney with ties), FNR at operating-point thresholds, and bootstrap confidence intervals. The natural choice in a data-science project would be numpy + scipy + sklearn, but the project's AGENTS.md says "the standard library is preferred" and numpy/scipy are not in the system interpreter.

Adding a numpy dependency would require `uv add numpy scipy scikit-learn`, pinning versions, and would pull in a large dependency tree for three statistical functions that are straightforward to implement in pure Python.

## Decision
**All analytics are pure Python — no numpy, scipy, or sklearn.** AUC is computed as the Mann-Whitney U statistic with ties handled as 0.5 contributions (`auc_from_scores`). Bootstrap CIs use `random.choices` with a recorded seed. FNR is the fraction of positives below the threshold. All functions work on plain lists of floats.

## Consequences
- Zero new dependencies beyond the Python standard library.
- AUC and CI functions are small (~30 lines each) and fully testable with hand-computed values.
- Performance is adequate for the gold set (242 rows) and corpus (5430 rows); the experiment scale does not need vectorized operations.
- A future stage that needs large-scale computation can revisit this decision and add numpy as an explicit dependency, justified by scale.