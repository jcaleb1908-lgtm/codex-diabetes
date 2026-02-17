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

`PROJECT6_SPINE_ALL_IN_ONE.py` is now a true standalone monolithic script (no package-relative imports) generated
from the same modular source, so both entrypoints run the same MI-first and overlap/weight-diagnostics methodology.

Required env:

- `WORKSPACE_CDR`
- BigQuery credentials in All of Us Workbench session

Outputs are written to `SPINE/project6_outputs`.

Notable diagnostics outputs:

- `logit_model_diagnostics.csv` (overall/by-exposure event counts, rates, EPV context)
- `cox_model_diagnostics.csv` (2-year-censored Cox analysis population and event totals)
