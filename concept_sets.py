"""
================================================================================
CONCEPT SETS - Centralized Definitions for All Measured Variables
================================================================================
GLP-1 Effect on Cardiometabolic Risk Factors
All of Us Registered Tier Dataset v8

This module centralizes all concept IDs used in the analysis.
All queries and filters must reference these definitions -- no scattered
hard-coded IDs elsewhere in the codebase.

NOTE: Inflammatory biomarkers (CRP, ESR, Fibrinogen) are intentionally
excluded from this concept set inventory per study protocol.
================================================================================
"""

import pandas as pd

# ============================================================================
# INCLUSION CRITERIA
# ============================================================================

# Type 2 Diabetes Mellitus (OMOP standard concept hierarchy root)
TYPE_2_DIABETES = {
    'name': 'Type 2 Diabetes Mellitus',
    'domain': 'Condition',
    'concept_ids': (201826,),
    'description': 'T2DM inclusion criterion; uses cb_criteria hierarchy',
    'is_standard': True,
}

# BMI >= 25 (overweight/obese inclusion gate)
BMI_INCLUSION = {
    'name': 'BMI >= 25 (Overweight/Obese)',
    'domain': 'Measurement',
    'concept_ids': (903124,),
    'description': 'Non-standard concept used in cb_search_all_events for BMI >= 25',
    'is_standard': False,
}

# ============================================================================
# EXCLUSION CRITERIA
# ============================================================================

# Cardiovascular / severe conditions exclusion
CV_EXCLUSION = {
    'name': 'Cardiovascular / Severe Conditions',
    'domain': 'Condition',
    'concept_ids': (316139, 314059, 316998, 46271022, 4299535, 444031, 4329847, 319835),
    'description': (
        'Heart failure, atrial fibrillation, cardiac arrest, '
        'cerebrovascular disease, peripheral vascular disease, '
        'cardiomyopathy, acute MI, cardiogenic shock'
    ),
    'is_standard': True,
}

# Pregnancy-related exclusion
PREGNANCY_EXCLUSION = {
    'name': 'Pregnancy-Related Conditions',
    'domain': 'Condition',
    'concept_ids': (24612, 313217, 199074),
    'description': 'Malignant neoplasm of breast, pregnancy, endometriosis',
    'is_standard': True,
}

# Demographic exclusions (missing / unknown values)
ETHNICITY_EXCLUSION = {
    'name': 'Ethnicity Exclusion (Missing/Unknown)',
    'domain': 'Person',
    'concept_ids': (903096, 0, 903079, 1586148),
    'description': 'PMI: Skip, No matching concept, PMI: Prefer Not To Answer, free text',
}

RACE_EXCLUSION = {
    'name': 'Race Exclusion (Missing/Unknown)',
    'domain': 'Person',
    'concept_ids': (2100000001, 903096, 45882607, 1177221),
    'description': 'AoU custom: None Indicated, PMI: Skip, None of these, Missing',
}

GENDER_EXCLUSION = {
    'name': 'Gender Exclusion (Missing/Unknown)',
    'domain': 'Person',
    'concept_ids': (2000000002, 0),
    'description': 'AoU custom: Gender Identity: Non Binary, No matching concept',
}

SEX_AT_BIRTH_EXCLUSION = {
    'name': 'Sex at Birth Exclusion (Missing/Unknown)',
    'domain': 'Person',
    'concept_ids': (2000000009, 0),
    'description': 'AoU custom: PMI: Skip, No matching concept',
}

# Insulin exclusion (ancestor-based)
INSULIN_EXCLUSION = {
    'name': 'Insulin Therapy',
    'domain': 'Drug',
    'concept_ids': (1539403,),
    'description': 'Insulin ancestor concept; uses cb_criteria_ancestor for descendant lookup',
    'is_standard': True,
}

# ============================================================================
# EXPOSURE / DRUG CONCEPTS
# ============================================================================

# Metformin
METFORMIN = {
    'name': 'Metformin',
    'domain': 'Drug',
    'concept_ids': (1123618,),
    'description': 'Metformin; also matched by string pattern on standard_concept_name',
    'string_pattern': 'metformin',
    'is_standard': True,
}

# GLP-1 Receptor Agonists
GLP1_AGONISTS = {
    'name': 'GLP-1 Receptor Agonists',
    'domain': 'Drug',
    'concept_ids': (1123627, 21600745, 21600749, 21600765, 21600775,
                    21600779, 21600783, 21600788, 21601855),
    'description': (
        'Dulaglutide, liraglutide, semaglutide, exenatide, tirzepatide; '
        'also matched by brand names'
    ),
    'string_pattern': (
        'dulaglutide|liraglutide|semaglutide|exenatide|'
        'trulicity|victoza|ozempic|saxenda|wegovy|mounjaro|tirzepatide'
    ),
    'is_standard': True,
}

# Statins (HMG-CoA reductase inhibitors)
STATINS = {
    'name': 'Statins (HMG-CoA Reductase Inhibitors)',
    'domain': 'Drug',
    'concept_ids': (83367, 596723, 41127, 6472, 861634, 42463, 301542, 36567),
    'description': (
        'Atorvastatin, cerivastatin, fluvastatin, lovastatin, '
        'pitavastatin, pravastatin, rosuvastatin, simvastatin'
    ),
    'string_pattern': (
        'atorvastatin|cerivastatin|fluvastatin|lovastatin|'
        'pitavastatin|pravastatin|rosuvastatin|simvastatin|'
        'lipitor|crestor|zocor|lescol|mevacor|livalo|pravachol'
    ),
    'is_standard': True,
}

# Combined drug concept IDs for the SQL query
ALL_DRUG_CONCEPT_IDS = (
    METFORMIN['concept_ids']
    + GLP1_AGONISTS['concept_ids']
    + STATINS['concept_ids']
)

# ============================================================================
# OUTCOME MEASUREMENTS (excluding inflammatory biomarkers)
# ============================================================================

# Standard measurement concept IDs used in the measurement SQL query
MEASUREMENT_STANDARD_CONCEPTS = {
    'name': 'Cardiometabolic Measurements (Standard)',
    'domain': 'Measurement',
    'concept_ids': (
        3004249,   # HbA1c (Hemoglobin A1c/Hemoglobin.total in Blood)
        3004410,   # HbA1c (variant)
        3025315,   # HbA1c (variant)
        40789535,  # HbA1c (variant)
        40782589,  # HbA1c (variant)
        40782590,  # HbA1c (variant)
        3037110,   # Glucose
        37027885,  # Glucose (variant)
        37045117,  # Glucose (variant)
        40772572,  # Glucose (variant)
        40779160,  # LDL Cholesterol
        40795740,  # HDL Cholesterol
        40795800,  # Total Cholesterol
        3012888,   # Triglycerides
        3031203,   # Triglycerides (variant)
        1002597,   # Triglycerides (variant)
    ),
    'description': 'Standard concept IDs for glycemic and lipid measurements',
    'is_standard': True,
}

# Non-standard / source measurement concept IDs
MEASUREMENT_SOURCE_CONCEPTS = {
    'name': 'Cardiometabolic Measurements (Source/Non-Standard)',
    'domain': 'Measurement',
    'concept_ids': (903107, 903121, 903124, 903135),
    'description': 'Non-standard source concept IDs for measurement lookup',
    'is_standard': False,
}

# BMI-specific concepts for extraction
BMI_MEASUREMENT = {
    'name': 'Body Mass Index',
    'domain': 'Measurement',
    'concept_ids': (903124, 3038553, 40762636),
    'description': (
        '903124 = AoU non-standard BMI; 3038553 = Body mass index (BMI) [Ratio]; '
        '40762636 = Body mass index. '
        'Also matched by string pattern on standard_concept_name.'
    ),
    'string_pattern': 'body mass index|bmi',
    'is_standard': False,
    'plausible_range': (12.0, 80.0),
    'expected_unit': 'kg/m2',
}

# ============================================================================
# GLYCEMIC OUTCOME LABELS (used in categorize_measurement)
# ============================================================================

GLYCEMIC_OUTCOMES = ['Glucose', 'HbA1c']

# ============================================================================
# LIPID OUTCOME LABELS
# ============================================================================

LIPID_OUTCOMES = ['HDL', 'LDL', 'Total_Cholesterol', 'Triglycerides']

# ============================================================================
# ALL OUTCOMES (excluding inflammatory)
# ============================================================================

ALL_OUTCOMES = GLYCEMIC_OUTCOMES + LIPID_OUTCOMES

# ============================================================================
# PLAUSIBLE RANGES FOR DATA CLEANING (excluding inflammatory biomarkers)
# ============================================================================

PLAUSIBLE_RANGES = {
    'Glucose': (20, 800),
    'HbA1c': (3, 20),
    'HDL': (5, 150),
    'LDL': (10, 400),
    'Total_Cholesterol': (50, 500),
    'Triglycerides': (20, 2000),
    'BMI': (12.0, 80.0),
}

# ============================================================================
# CONDITION CONCEPTS (hypertension / essential hypertension for condition query)
# ============================================================================

HYPERTENSION_CONDITIONS = {
    'name': 'Essential Hypertension',
    'domain': 'Condition',
    'concept_ids': (320128, 444208),
    'description': 'Essential hypertension and generalized ischemic cerebrovascular disease',
    'is_standard': True,
}

# ============================================================================
# SURVEY CONCEPTS (Smoking and Lifestyle)
# ============================================================================

SMOKING_SURVEY_QUESTIONS = {
    'name': 'Smoking Survey Questions',
    'domain': 'Survey',
    'concept_ids': (1585857, 1585860, 1586177, 1585855),
    'description': (
        '1585857 = Smoking status (100 cigarettes); '
        '1585860 = Current smoking frequency; '
        '1586177 = Cigarette smoking status; '
        '1585855 = Tobacco use parent question (hierarchy root for sub-questions)'
    ),
    'non_smoker_answers': (
        'No',
        'Not at all',
        'PMI: No',
    ),
    'unknown_skip_answers': (
        'PMI: Skip',
        'PMI: Prefer Not To Answer',
        'PMI: Dont Know',
        'Dont Know',
        'Skip',
        'Prefer Not To Answer',
    ),
}

# ============================================================================
# NEW COHORT MEASUREMENT CONCEPTS (from user-provided queries dataset 78271067)
# ============================================================================

NEW_COHORT_MEASUREMENT_CONCEPTS = {
    'name': 'New Cohort Cardiometabolic Measurements',
    'domain': 'Measurement',
    'concept_ids': (
        1002597, 3004249, 3004410, 3012888, 3025315, 3031203,
        3037110, 37027885, 37045117, 40772572, 40779160,
        40782589, 40782590, 40789535, 40795740, 40795800,
        4087496, 4093979, 4099154, 4245997,
    ),
    'description': 'Extended set of measurement concepts from new cohort (dataset 78271067)',
    'is_standard': True,
}

# New cohort drug concepts (Metformin, GLP-1, Statins)
NEW_COHORT_DRUG_CONCEPTS = {
    'name': 'New Cohort Drug Exposure Concepts',
    'domain': 'Drug',
    'concept_ids': (1123618, 1503297, 21601855),
    'description': 'Drug ancestor concepts for new cohort: metformin (1123618), GLP-1 (1503297, 21601855)',
    'is_standard': True,
}

# New cohort CV exclusion concepts (same structure, slightly different set)
NEW_COHORT_CV_EXCLUSION = {
    'name': 'New Cohort CV Exclusion',
    'domain': 'Condition',
    'concept_ids': (316139, 4299535, 46271022, 444031, 4329847, 319835),
    'description': 'Cardiovascular exclusions in new cohort definition',
    'is_standard': True,
}

# ============================================================================
# TREATMENT GROUPS
# ============================================================================

TREATMENT_GROUPS = [
    'Statin_mono',
    'Metformin_mono',
    'GLP1_mono',
    'Statin_GLP1',
    'Statin_Metformin',
    'Metformin_GLP1',
    'Statin_GLP1_Metformin',
]

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

PRE_WINDOW_DAYS = 365
POST_WINDOW_DAYS = 365
MIN_PRE_POST_GAP_DAYS = 30  # Minimum separation between pre and post measurements

# Measure-specific minimum gaps (override the default 30-day rule where needed)
MEASURE_SPECIFIC_MIN_GAP = {
    'HbA1c': 90,           # HbA1c reflects ~3 months; need >= 90 days separation
    'Glucose': 30,         # Fasting glucose can change quickly
    'HDL': 30,             # Lipids can change within weeks on therapy
    'LDL': 30,
    'Total_Cholesterol': 30,
    'Triglycerides': 30,
    'BMI': 30,             # BMI changes are gradual but 30 days is a minimum
}

HBA1C_TARGET = 7.0
GLYCEMIC_RESPONSE_THRESHOLD = -0.5   # HbA1c decrease >= 0.5%
LIPID_RESPONSE_LDL_TARGET = 100      # LDL < 100 mg/dL post-treatment
LIPID_RESPONSE_LDL_CHANGE = -20      # OR LDL decrease >= 20 mg/dL

AGE_RANGE = (40, 65)

# ============================================================================
# CONCEPT SET INVENTORY TABLE
# ============================================================================

def get_concept_set_inventory():
    """
    Return a DataFrame listing every concept set used in the analysis.

    Columns: name, domain, n_concepts, concept_ids, used_in
    """
    inventory = [
        {
            'name': TYPE_2_DIABETES['name'],
            'domain': TYPE_2_DIABETES['domain'],
            'n_concepts': len(TYPE_2_DIABETES['concept_ids']),
            'concept_ids': str(TYPE_2_DIABETES['concept_ids']),
            'used_in': 'Cohort inclusion (person, measurement, drug, condition, survey SQL)',
        },
        {
            'name': BMI_INCLUSION['name'],
            'domain': BMI_INCLUSION['domain'],
            'n_concepts': len(BMI_INCLUSION['concept_ids']),
            'concept_ids': str(BMI_INCLUSION['concept_ids']),
            'used_in': 'Cohort inclusion gate (BMI >= 25)',
        },
        {
            'name': CV_EXCLUSION['name'],
            'domain': CV_EXCLUSION['domain'],
            'n_concepts': len(CV_EXCLUSION['concept_ids']),
            'concept_ids': str(CV_EXCLUSION['concept_ids']),
            'used_in': 'Cohort exclusion (all SQL queries)',
        },
        {
            'name': PREGNANCY_EXCLUSION['name'],
            'domain': PREGNANCY_EXCLUSION['domain'],
            'n_concepts': len(PREGNANCY_EXCLUSION['concept_ids']),
            'concept_ids': str(PREGNANCY_EXCLUSION['concept_ids']),
            'used_in': 'Cohort exclusion (all SQL queries)',
        },
        {
            'name': ETHNICITY_EXCLUSION['name'],
            'domain': ETHNICITY_EXCLUSION['domain'],
            'n_concepts': len(ETHNICITY_EXCLUSION['concept_ids']),
            'concept_ids': str(ETHNICITY_EXCLUSION['concept_ids']),
            'used_in': 'Demographic exclusion (all SQL queries)',
        },
        {
            'name': RACE_EXCLUSION['name'],
            'domain': RACE_EXCLUSION['domain'],
            'n_concepts': len(RACE_EXCLUSION['concept_ids']),
            'concept_ids': str(RACE_EXCLUSION['concept_ids']),
            'used_in': 'Demographic exclusion (all SQL queries)',
        },
        {
            'name': GENDER_EXCLUSION['name'],
            'domain': GENDER_EXCLUSION['domain'],
            'n_concepts': len(GENDER_EXCLUSION['concept_ids']),
            'concept_ids': str(GENDER_EXCLUSION['concept_ids']),
            'used_in': 'Demographic exclusion (all SQL queries)',
        },
        {
            'name': SEX_AT_BIRTH_EXCLUSION['name'],
            'domain': SEX_AT_BIRTH_EXCLUSION['domain'],
            'n_concepts': len(SEX_AT_BIRTH_EXCLUSION['concept_ids']),
            'concept_ids': str(SEX_AT_BIRTH_EXCLUSION['concept_ids']),
            'used_in': 'Demographic exclusion (all SQL queries)',
        },
        {
            'name': INSULIN_EXCLUSION['name'],
            'domain': INSULIN_EXCLUSION['domain'],
            'n_concepts': len(INSULIN_EXCLUSION['concept_ids']),
            'concept_ids': str(INSULIN_EXCLUSION['concept_ids']),
            'used_in': 'Drug exclusion (person, measurement, drug SQL)',
        },
        {
            'name': METFORMIN['name'],
            'domain': METFORMIN['domain'],
            'n_concepts': len(METFORMIN['concept_ids']),
            'concept_ids': str(METFORMIN['concept_ids']),
            'used_in': 'Exposure classification (drug SQL, treatment groups)',
        },
        {
            'name': GLP1_AGONISTS['name'],
            'domain': GLP1_AGONISTS['domain'],
            'n_concepts': len(GLP1_AGONISTS['concept_ids']),
            'concept_ids': str(GLP1_AGONISTS['concept_ids']),
            'used_in': 'Exposure classification (drug SQL, treatment groups)',
        },
        {
            'name': STATINS['name'],
            'domain': STATINS['domain'],
            'n_concepts': len(STATINS['concept_ids']),
            'concept_ids': str(STATINS['concept_ids']),
            'used_in': 'Exposure classification (drug SQL, treatment groups)',
        },
        {
            'name': MEASUREMENT_STANDARD_CONCEPTS['name'],
            'domain': MEASUREMENT_STANDARD_CONCEPTS['domain'],
            'n_concepts': len(MEASUREMENT_STANDARD_CONCEPTS['concept_ids']),
            'concept_ids': str(MEASUREMENT_STANDARD_CONCEPTS['concept_ids']),
            'used_in': 'Outcome extraction (measurement SQL)',
        },
        {
            'name': MEASUREMENT_SOURCE_CONCEPTS['name'],
            'domain': MEASUREMENT_SOURCE_CONCEPTS['domain'],
            'n_concepts': len(MEASUREMENT_SOURCE_CONCEPTS['concept_ids']),
            'concept_ids': str(MEASUREMENT_SOURCE_CONCEPTS['concept_ids']),
            'used_in': 'Outcome extraction fallback (measurement SQL, source concepts)',
        },
        {
            'name': BMI_MEASUREMENT['name'],
            'domain': BMI_MEASUREMENT['domain'],
            'n_concepts': len(BMI_MEASUREMENT['concept_ids']),
            'concept_ids': str(BMI_MEASUREMENT['concept_ids']),
            'used_in': 'BMI extraction for baseline features and clustering',
        },
        {
            'name': HYPERTENSION_CONDITIONS['name'],
            'domain': HYPERTENSION_CONDITIONS['domain'],
            'n_concepts': len(HYPERTENSION_CONDITIONS['concept_ids']),
            'concept_ids': str(HYPERTENSION_CONDITIONS['concept_ids']),
            'used_in': 'Condition query (comorbidity identification)',
        },
        {
            'name': SMOKING_SURVEY_QUESTIONS['name'],
            'domain': SMOKING_SURVEY_QUESTIONS['domain'],
            'n_concepts': len(SMOKING_SURVEY_QUESTIONS['concept_ids']),
            'concept_ids': str(SMOKING_SURVEY_QUESTIONS['concept_ids']),
            'used_in': 'Smoking status derivation (survey SQL, binary covariate)',
        },
        {
            'name': NEW_COHORT_MEASUREMENT_CONCEPTS['name'],
            'domain': NEW_COHORT_MEASUREMENT_CONCEPTS['domain'],
            'n_concepts': len(NEW_COHORT_MEASUREMENT_CONCEPTS['concept_ids']),
            'concept_ids': str(NEW_COHORT_MEASUREMENT_CONCEPTS['concept_ids']),
            'used_in': 'Extended measurement extraction (new cohort dataset 78271067)',
        },
        {
            'name': NEW_COHORT_DRUG_CONCEPTS['name'],
            'domain': NEW_COHORT_DRUG_CONCEPTS['domain'],
            'n_concepts': len(NEW_COHORT_DRUG_CONCEPTS['concept_ids']),
            'concept_ids': str(NEW_COHORT_DRUG_CONCEPTS['concept_ids']),
            'used_in': 'Drug exposure extraction (new cohort dataset 78271067)',
        },
    ]
    return pd.DataFrame(inventory)


def print_concept_set_inventory():
    """Print a formatted concept set inventory table."""
    df = get_concept_set_inventory()
    print("\n" + "=" * 120)
    print("CONCEPT SET INVENTORY")
    print("=" * 120)
    print(f"\n{'#':<4} {'Name':<45} {'Domain':<15} {'#IDs':<8} {'Used In'}")
    print("-" * 120)
    for i, row in df.iterrows():
        print(f"{i+1:<4} {row['name']:<45} {row['domain']:<15} {row['n_concepts']:<8} {row['used_in']}")
    print("-" * 120)
    print(f"Total concept sets: {len(df)}")
    print(f"Total unique concept IDs across all sets: see concept_ids column for details")
    print("=" * 120)
    return df


# ============================================================================
# SQL HELPER: format concept_ids tuple for SQL IN clauses
# ============================================================================

def ids_for_sql(concept_set):
    """
    Return a comma-separated string of concept IDs suitable for SQL IN(...).
    Accepts either a dict with 'concept_ids' key or a plain tuple.
    """
    if isinstance(concept_set, dict):
        ids = concept_set['concept_ids']
    else:
        ids = concept_set
    return ', '.join(str(c) for c in ids)
