# Project 6 Spine Pipeline

Run:

```bash
cd /Users/jcbn/Desktop/CODEX/SPINE
python run_project6_spine_pipeline.py
```

All-in-one single-file version (for copy/paste workflows):

```bash
cd /Users/jcbn/Desktop/CODEX/SPINE
python3 PROJECT6_SPINE_ALL_IN_ONE.py
```

Note: the canonical implementation is now the modular package under `project6_spine_pipeline/`; use
`run_project6_spine_pipeline.py` for current methods (MI-first missingness handling, overlap/double weighting, and
updated diagnostics).

Required env:

- `WORKSPACE_CDR`
- BigQuery credentials in All of Us Workbench session

Outputs are written to `SPINE/project6_outputs`.

Notable diagnostics outputs:

- `logit_model_diagnostics.csv` (overall/by-exposure event counts, rates, EPV context)
- `cox_model_diagnostics.csv` (2-year-censored Cox analysis population and event totals)
