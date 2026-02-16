import os
import sys
import runpy
import numpy as np
import pandas as pd

np.random.seed(42)
os.environ.setdefault("WORKSPACE_CDR", "dummy")

GROUPS = [
    'Statin_mono',
    'Metformin_mono',
    'GLP1_mono',
    'Statin_GLP1',
    'Statin_Metformin',
    'Metformin_GLP1',
    'Statin_GLP1_Metformin',
]

N_PER_GROUP = 36
person_rows = []
drug_rows = []
measurement_rows = []
survey_rows = []

races = ['White', 'Black or African American', 'Asian', 'Another single population']
eths = ['Not Hispanic or Latino', 'Hispanic or Latino']
sexes = ['Male', 'Female']

pid = 1
for g in GROUPS:
    for i in range(N_PER_GROUP):
        person_id = pid
        pid += 1

        sex = np.random.choice(sexes)
        race = np.random.choice(races, p=[0.45, 0.25, 0.15, 0.15])
        eth = np.random.choice(eths, p=[0.8, 0.2])

        dob = pd.Timestamp('1962-01-01') + pd.to_timedelta(np.random.randint(0, 3650), unit='D')
        base = pd.Timestamp('2021-01-01') + pd.to_timedelta(np.random.randint(0, 730), unit='D')

        statin_start = pd.NaT
        met_start = pd.NaT
        glp1_start = pd.NaT

        if g == 'Statin_mono':
            statin_start = base
        elif g == 'Metformin_mono':
            met_start = base
        elif g == 'GLP1_mono':
            glp1_start = base
        elif g == 'Statin_GLP1':
            statin_start = base - pd.Timedelta(days=np.random.randint(40, 180))
            glp1_start = base
        elif g == 'Statin_Metformin':
            statin_start = base - pd.Timedelta(days=np.random.randint(40, 180))
            met_start = base
        elif g == 'Metformin_GLP1':
            met_start = base - pd.Timedelta(days=np.random.randint(40, 180))
            glp1_start = base
        elif g == 'Statin_GLP1_Metformin':
            statin_start = base - pd.Timedelta(days=np.random.randint(120, 260))
            met_start = base - pd.Timedelta(days=np.random.randint(50, 150))
            glp1_start = base

        starts = [x for x in [statin_start, met_start, glp1_start] if pd.notna(x)]
        index_date = max(starts)

        person_rows.append({
            'person_id': person_id,
            'gender_concept_id': 8507 if sex == 'Male' else 8532,
            'gender': sex,
            'date_of_birth': dob,
            'race_concept_id': 1,
            'race': race,
            'ethnicity_concept_id': 1,
            'ethnicity': eth,
            'sex_at_birth_concept_id': 1,
            'sex_at_birth': sex,
        })

        if pd.notna(statin_start):
            drug_rows.append({
                'person_id': person_id,
                'drug_concept_id': 41127,
                'standard_concept_name': 'atorvastatin 20 MG Oral Tablet',
                'drug_exposure_start_datetime': statin_start,
            })
        if pd.notna(met_start):
            drug_rows.append({
                'person_id': person_id,
                'drug_concept_id': 1123618,
                'standard_concept_name': 'metformin hydrochloride 500 MG Oral Tablet',
                'drug_exposure_start_datetime': met_start,
            })
        if pd.notna(glp1_start):
            glp1_name = np.random.choice(['semaglutide 1 MG/DOSE Injectable', 'liraglutide Injectable', 'dulaglutide Injectable'])
            drug_rows.append({
                'person_id': person_id,
                'drug_concept_id': 21600749,
                'standard_concept_name': glp1_name,
                'drug_exposure_start_datetime': glp1_start,
            })

        glp1_flag = int('GLP1' in g)
        statin_flag = int('Statin' in g)

        hba1c_pre = np.random.uniform(7.3, 10.5)
        hba1c_delta = np.random.normal(-0.9 if glp1_flag else -0.45, 0.35)
        hba1c_post = np.clip(hba1c_pre + hba1c_delta, 4.8, 15)

        ldl_pre = np.random.uniform(95, 185)
        ldl_delta = np.random.normal(-34 if statin_flag else -10, 12)
        if glp1_flag:
            ldl_delta -= 4
        ldl_post = np.clip(ldl_pre + ldl_delta, 30, 320)

        hdl_pre = np.random.uniform(35, 65)
        hdl_post = np.clip(hdl_pre + np.random.normal(2.5, 4), 5, 150)

        tg_pre = np.random.uniform(120, 320)
        tg_post = np.clip(tg_pre + np.random.normal(-35 if glp1_flag else -15, 35), 20, 1800)

        tc_pre = np.random.uniform(170, 280)
        tc_post = np.clip(tc_pre + np.random.normal(-22 if statin_flag else -8, 18), 60, 480)

        glu_pre = np.random.uniform(120, 230)
        glu_post = np.clip(glu_pre + np.random.normal(-30 if glp1_flag else -18, 18), 30, 700)

        bmi_pre = np.random.uniform(27, 43)
        bmi_post = np.clip(bmi_pre + np.random.normal(-1.5 if glp1_flag else -0.4, 1.2), 16, 70)

        pre_dt = index_date - pd.Timedelta(days=np.random.randint(40, 180))
        post_dt = index_date + pd.Timedelta(days=np.random.randint(95, 260))

        measures = [
            (3004249, 'Hemoglobin A1c/Hemoglobin.total in Blood', hba1c_pre, hba1c_post, '%', 1),
            (3037110, 'Glucose [Mass/volume] in Blood', glu_pre, glu_post, 'mg/dL', 2),
            (40779160, 'LDL Cholesterol', ldl_pre, ldl_post, 'mg/dL', 2),
            (40795740, 'HDL Cholesterol', hdl_pre, hdl_post, 'mg/dL', 2),
            (40795800, 'Total Cholesterol', tc_pre, tc_post, 'mg/dL', 2),
            (3012888, 'Triglyceride [Mass/volume] in Serum or Plasma', tg_pre, tg_post, 'mg/dL', 2),
            (903124, 'Body mass index (BMI)', bmi_pre, bmi_post, 'kg/m2', 3),
        ]

        for cid, cname, pre_v, post_v, unit_name, unit_cid in measures:
            measurement_rows.append({
                'person_id': person_id,
                'measurement_concept_id': cid,
                'standard_concept_name': cname,
                'measurement_datetime': pre_dt,
                'value_as_number': float(pre_v),
                'unit_concept_id': unit_cid,
                'unit_concept_name': unit_name,
            })
            measurement_rows.append({
                'person_id': person_id,
                'measurement_concept_id': cid,
                'standard_concept_name': cname,
                'measurement_datetime': post_dt,
                'value_as_number': float(post_v),
                'unit_concept_id': unit_cid,
                'unit_concept_name': unit_name,
            })

        # Extra HbA1c measurements to create intensity differences by exposure group
        n_extra = np.random.randint(2, 5) if glp1_flag else np.random.randint(0, 2)
        for _ in range(n_extra):
            extra_day = np.random.randint(20, 340)
            measurement_rows.append({
                'person_id': person_id,
                'measurement_concept_id': 3004249,
                'standard_concept_name': 'Hemoglobin A1c/Hemoglobin.total in Blood',
                'measurement_datetime': index_date + pd.Timedelta(days=int(extra_day)),
                'value_as_number': float(np.clip(hba1c_post + np.random.normal(0, 0.35), 4.8, 14)),
                'unit_concept_id': 1,
                'unit_concept_name': '%',
            })

        is_smoker = np.random.rand() < (0.30 if glp1_flag else 0.22)
        answer = 'Every day' if is_smoker else 'No'
        survey_rows.append({
            'person_id': person_id,
            'survey_datetime': index_date - pd.Timedelta(days=np.random.randint(30, 300)),
            'survey': 'Lifestyle',
            'question_concept_id': 1585860,
            'question': 'Do you now smoke cigarettes every day, some days, or not at all?',
            'answer_concept_id': 1,
            'answer': answer,
            'survey_version_concept_id': 1,
            'survey_version_name': 'v1',
        })

person_df = pd.DataFrame(person_rows)
measurement_df = pd.DataFrame(measurement_rows)
drug_df = pd.DataFrame(drug_rows)
survey_df = pd.DataFrame(survey_rows)

calls = [person_df, measurement_df, drug_df, survey_df]

def fake_read_gbq(sql, *args, **kwargs):
    idx = fake_read_gbq.idx
    if idx >= len(calls):
        raise RuntimeError(f"Unexpected read_gbq call #{idx + 1}")
    fake_read_gbq.idx += 1
    return calls[idx].copy()

fake_read_gbq.idx = 0

import pandas as pandas_module
pandas_module.read_gbq = fake_read_gbq
pd.read_gbq = fake_read_gbq

sys.path.insert(0, '/Users/jcbn/Desktop/CODEX')

runpy.run_path('/Users/jcbn/Desktop/CODEX/MASTES CODE COPY', run_name='__main__')
print('MOCK_RUN_SUCCESS')
