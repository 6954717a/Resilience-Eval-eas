# Evidence Catalog

| Evidence | Status | Relevance | Limitation or next check |
|:--|:--|:--|:--|
| `README.md` | Current | Public framing, metrics, results, and analysis commands | Clone owner remains a publication-time placeholder. |
| `INSTALLATION.md` | Partial | Python, CUDA, Habitat-Sim, and LLM setup | Requires target-machine dependency validation. |
| `requirements.txt` | Current inventory | Python dependency names and selected pins | No lockfile or package manifest is available. |
| `ResilienceEvaluationLayer/evaluation/` | Current source | Metrics, monitors, perturbations, baselines, and aggregation | Full execution depends on Habitat/PARTNR assets and endpoints. |
| `ResilienceEvaluationLayer/tests/` | Discovered | Fifteen main-package pytest files | Not all dependencies are available in this Windows checkout. |
| `Analysis_Code/` | Current source | Plotting, table construction, and legacy metric extraction | Some scripts consume packaged paths directly. |
| `Analysis_Data/Aggregated_Tables/` | Packaged evidence | Inputs for manuscript-facing benchmark tables and figures | Provenance should remain paired with the generating configs. |
| `Analysis_Data/Exp1_*` | Restored from legacy HEAD | Mini runtime evidence for Rebound, Stability, and GE | Deep paths require a short Windows checkout or long-path support. |
| `Analysis_Results/` | Derived artifact | Precomputed tables, images, and videos | Regeneration coverage varies by artifact. |
| Legacy Git object store | Verified | Recovery source for the complete 2,239-file tree | Kept outside the fresh repository; do not publish its credential-bearing history. |
| Fresh-repository secret scan | Verified before initial commit | Zero staged files matched known token or private-key patterns; three local API-key files are ignored | Provider-side rotation of legacy keys remains external work. |
| Fresh-repository large-file scan | Verified before initial commit | Zero staged files are 50 MiB or larger; the largest is 12.87 MiB | Reassess if future binary outputs exceed normal Git limits. |
| Scoped whitespace check | Verified before initial commit | Migration-authored files pass `git diff --cached --check` | Imported legacy files retain pre-existing trailing-whitespace debt. |
