# Study-fork modifications

This is a public, unofficial fork of
<https://github.com/vanderschaarlab/synthcity> for the knee OA synthetic-data
study.

- Upstream base commit: `bc228832262ff67e51efee5b66bd364f944d5ab7`
- Study baseline before this disclosure update:
  `c704189bc159d5ccea814d6ace9079b324532ee2`
- Licence retained from upstream: Apache License 2.0
- Repository visibility: public

The fork changes dependency constraints, dataset loading and dataframe
handling, performance and privacy metric evaluation, metric score aggregation,
multiclass SHAP layout handling, and associated tests used by the study
pipeline.

The exact changed-file inventory relative to the upstream base can be checked
with:

```bash
git diff --name-status bc228832262ff67e51efee5b66bd364f944d5ab7..HEAD
```

Modified Apache-licensed source, configuration, documentation, and test files
carry a prominent change notice and retain applicable upstream attribution.
The upstream `LICENSE` is preserved unchanged. No top-level upstream `NOTICE`
file was present at the recorded base commit.
