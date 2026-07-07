# Post-run documentation patches

Date: 2026-06-30

No experiments were re-run for this documentation patch. No full run, CTGAN run, LLM/Ollama run, notebook execution, or output regeneration was performed.

The archived result run `20260629T202543Z`, synthetic data, splits, result tables, generator parameters, prompts, temperatures, seeds, fallback logic, metrics, validation logic, raw LLM outputs, generated code, and environment snapshot were not changed.

Patches applied:

1. README/main safety correction: `main()` is documented and configured as a non-experiment preflight entry point by default.
2. Partial split-regeneration guard: if archived split files are only partially present, regeneration aborts with a clear error instead of reconstructing missing train splits from an existing test split.
3. Semantic plausibility evidence: `results/tables/semantic_plausibility.csv` was added as a post-hoc diagnostic table computed from archived final synthetic CSVs and archived real training splits.
4. README notes were added for reproduction entry points, processed CSVs, the archived result run, the environment snapshot, and post-hoc semantic plausibility diagnostics.

Because documentation/proof files were added or changed, the ZIP SHA-256 changes. Archived result artifacts remain unchanged except for the manifest entry update needed to record the documentation/proof patch.
