# Historical comparison

Mode: `stage_a_train`; smoke: `true`.

Reproducibility classification: `current_implementation_baseline`.

Exact reproduction is not claimed: historical manifests came from different dirty revisions and do not preserve their patches.

Current chunks: 28; tokens: 6840.
Historical structure chunks: 6007; tokens: 1316605.
Segmenter: `structure`. Structure historical chunk-hash match: `false`.
Current train Hybrid EvidenceCoverage@5 (structure): 1.000000000000.
Stored historical structure train Hybrid EvidenceCoverage@5: 0.828070663854.
Historical metric is context only; fixed-versus-structure inference requires paired current runs.

The historical 30-row test split was not loaded in stage_a_train mode. Historical regression remains an explicit opt-in technical mode.
