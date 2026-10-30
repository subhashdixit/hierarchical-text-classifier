# Model card

- Intended use: educational product-text classification in the supplied English taxonomy.
- Artifact: `constrained_lr_c4`, model version `1.0.0`.
- Data: preserved 10,000-record CSV. External provenance and redistribution permission not verified.
- Preprocessing: shared Unicode/case/whitespace normalization; title + body; training-only word/bigram TF-IDF vocabulary.
- Selection: three candidates on validation exact-path accuracy; no test-driven selection.
- Split: connected product/text groups, seed 42; 5810 train / 2091 validation / 1990 test rows.
- Exact-path test accuracy: 0.4663. Top-level accuracy: 0.8377.
- Macro/weighted F1, unseen support, timing and rare/common results: [metrics](../reports/metrics.json).
- Valid paths: 404, learned only from training. New true paths cannot be generated and count as evaluation errors.
- Scores: uncalibrated path rankings. Review threshold 0.45 is an illustrative heuristic.
- Limitations: severe rare-label weakness, ambiguous/short reviews, unknown labels, unsupported domains, single fixed holdout and no production feedback study.
- Not suitable for consequential automated decisions. Human review is required for uncertainty.
- Reproducibility: [metadata](../models/metadata.json) records model, dataset and config hashes and sklearn version. The source revision contains the matching implementation. Exact floating-point values may vary across CPU/BLAS environments.
- Artifact trust: load only approved repository artifacts; hashes alone are not a signature.
