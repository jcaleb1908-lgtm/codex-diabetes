# codex-diabetes — ARCHIVED SANDBOX DUMP (soft-deprecated)

> **Do not use this repository as a study home.**  
> This was a Feb 2026 Codex workspace dump mixing **multiple All of Us studies**
> plus GLP-1 advanced-analysis experiments. Prefer the dedicated study repos below.
> The GitHub repo is intentionally **not deleted**.

## Salvaged / relocated

| Content | Destination |
|---------|-------------|
| Unique GLP-1 CSV/HTML/txt diagnostics (`index.html`, FDR, immortal time, ANCOVA, etc.) | `GLP1-EFFECT-ON-CARDIOMETABOLIC-RISK-FACTORS` → `results/codex-diabetes-salvage/` (PR branch `organize/import-codex-diabetes-unique`) |
| Advanced-analysis / mock-run variants | Same GLP1 repo → `archive/from-codex-diabetes/` |
| `concept_sets.py` (exact SHA) | Already in GLP1 root; copy moved to `archive/duplicates/` here |
| `PCOS GENERAL`, `PCOS POLYMORPH` | Exact older dups of PCOS — moved to `archive/duplicates/` |
| `GEST DIAB` | Exact older dup of GD-and-CVR `MASTER CODE` — `archive/duplicates/` |
| `CANCER RISK` | Exact older dup of CANCER-BMI `MASTER CODE` — `archive/duplicates/` |

## Canonical study homes

- **GLP-1 / DM cardiometabolic:** `GLP1-EFFECT-ON-CARDIOMETABOLIC-RISK-FACTORS`
- **PCOS:** `PCOS`
- **GDM→CVD:** `GD-and-CVR`
- **Cancer–BMI:** `CANCER-BMI`

## Remaining root artifacts

Diagnostic CSVs/txt/`index.html` remain at the root as historical copies of what was
salvaged into GLP1. Cross-study code dumps live under `archive/duplicates/`.

Soft-deprecated 2026-09-14 per `duplicate-merge-plan.md`. Optional next step:
GitHub “Archive repository” setting (admin), without deleting.
