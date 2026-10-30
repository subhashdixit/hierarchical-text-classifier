# Dataset documentation

`raw/data.csv` is the original dataset preserved unchanged from this repository before the Flask refactor. Its external source/license were not recorded clearly enough to establish provenance or commercial redistribution rights. Do not infer a dataset license from the repository's Apache software license.

10,000 rows; Title/Text are features, Cat1/Cat2/Cat3 targets. productId is used only for grouping splits; userId and Time are excluded from model input. Do not display or use identifiers as predictive shortcuts.

Training removes blank required fields and exact normalized text+label duplicates. Conflicting-label repeated text is retained and grouped; products and repeated text form connected components. See reports/metrics.json for counts. The raw file is never edited by training.

`examples/sample_batch.csv` contains synthetic demonstration text without identifiers. Deployment images exclude the raw dataset. No new uploaded data is saved.
