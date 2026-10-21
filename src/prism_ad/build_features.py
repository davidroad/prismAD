#!/usr/bin/env python3
"""Build restricted development-only data; exclude fold 0 from the present analysis."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from .runtime import workspace,sha
from .input_io import FOLDS,FOLD_HASH,BASELINE,COGNITION,filtered_csv,development_folds
from .transitions import normalize_visits
from .features import STUDIES,DOMAINS,CURRENT,HISTORY,endpoint,history_features
HERE=workspace()


def build():
    os.umask(0o077)
    if (HERE/'DATA_RECEIPT.json').exists():
        raise RuntimeError('Refusing to overwrite a completed feature build')
    if FOLD_HASH: assert sha(FOLDS) == FOLD_HASH
    folds = development_folds()
    b = filtered_csv(BASELINE, ['subject_id','baseline_diagnosis_code','index_visit','PRS'],
                     'subject_id', folds.subject_id)
    b = b.merge(folds, on='subject_id', validate='one_to_one')
    for c in ['baseline_diagnosis_code', 'index_visit', 'PRS']:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    columns = ['Study','SUBJID','PHC_Visit','PHC_Age_Cognition','PHC_Diagnosis',
               'PHC_Sex','PHC_Education'] + [f'PHC_{d}{suffix}' for d in DOMAINS
                                                          for suffix in ('','_SE','_PreciseFilter')]
    raw = filtered_csv(COGNITION, columns, 'SUBJID', b.subject_id)
    visits = normalize_visits(raw)
    assert not visits.duplicated(['subject_id','visit']).any() and visits.visit.notna().all()
    assert visits.groupby('subject_id').Study.nunique().max() == 1
    assert set(visits.subject_id) <= set(folds.subject_id)
    for c in ['PHC_Sex','PHC_Education'] + [c for c in columns if c.startswith(tuple('PHC_'+d for d in DOMAINS))]:
        visits[c] = pd.to_numeric(visits[c], errors='coerce')
    assert visits.PHC_Sex.dropna().isin([1, 2]).all()
    assert visits.PHC_Education.dropna().between(0, 40).all()
    for d in DOMAINS:
        assert visits[f'PHC_{d}_PreciseFilter'].dropna().isin([0, 1]).all()
        assert visits[f'PHC_{d}_SE'].dropna().ge(0).all()
    groups = {s: g for s, g in visits.groupby('subject_id', sort=False)}
    exclusion = Counter(); records = []; current = []; history = []; events = []; masks = []; prs = []
    for base in b.itertuples(index=False):
        g = groups.get(base.subject_id)
        if g is None:
            exclusion[('roster', 'unknown', 'no_visits')] += 1; continue
        study = str(g.Study.iloc[0])
        if study not in STUDIES:
            exclusion[('roster', study, 'study_not_in_pilot')] += 1; continue
        if base.baseline_diagnosis_code not in (1, 2) or not np.isfinite(base.PRS):
            exclusion[('roster', study, 'baseline_not_CN_MCI_or_no_PRS')] += 1; continue
        age = g.age.to_numpy(float); vnum = g.visit.to_numpy(float); dx = g.diagnosis.to_numpy(float)
        baseline_ix = np.flatnonzero(vnum == base.index_visit)
        assert len(baseline_ix) == 1 and dx[baseline_ix[0]] == base.baseline_diagnosis_code
        values = g[['PHC_'+d for d in DOMAINS]].to_numpy(float)
        se = g[['PHC_'+d+'_SE' for d in DOMAINS]].to_numpy(float)
        precise = g[['PHC_'+d+'_PreciseFilter' for d in DOMAINS]].to_numpy(float)
        sex, education = g.PHC_Sex.to_numpy(float), g.PHC_Education.to_numpy(float)
        for j in range(int(baseline_ix[0]), len(g)):
            if dx[j] == 1 and base.baseline_diagnosis_code == 1:
                task, state = 'CN_MCI', 1
            elif dx[j] == 2:
                task, state = 'MCI_AD', 2
            else:
                continue
            rec, reason = endpoint(age, dx, j, state)
            if rec is not None and np.isfinite(values[j]).sum() < 2:
                rec, reason = None, 'fewer_than_two_current_domains'
            exclusion[(task, study, reason)] += 1
            if rec is None:
                continue
            h, ev, mask = history_features(age, vnum, values, se, precise, j)
            rec.update(subject_id=str(base.subject_id), fold=int(base.fold), task=task,
                       study=study, age=float(age[j]), visit=float(vnum[j]),
                       origin='baseline_MCI' if base.baseline_diagnosis_code == 2 else 'original_CN',
                       has_history=bool(mask.any()), is_original_index=bool(vnum[j] == base.index_visit))
            records.append(rec)
            current.append([age[j], sex[j], education[j], np.nanmean(values[j]),
                            *values[j], np.isfinite(values[j]).sum()])
            history.append(h); events.append(ev); masks.append(mask); prs.append(base.PRS)
    m = pd.DataFrame(records)
    assert m.fold.isin(range(1, 10)).all() and m.groupby('subject_id').fold.nunique().max() == 1
    assert not m.duplicated(['task','subject_id','visit']).any()
    m['first_eligible'] = ~m.duplicated(['task','subject_id'])
    private = HERE/'LOCAL_RESTRICTED'; private.mkdir(mode=0o700, exist_ok=True)
    m.to_csv(private/'metadata.csv', index=False)
    np.savez_compressed(private/'features.npz', current=np.asarray(current, dtype=np.float32),
        history=np.asarray(history, dtype=np.float32), events=np.asarray(events, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool), prs=np.asarray(prs, dtype=float))
    ex = pd.DataFrame([dict(task=t, study=s, reason=r, n=n) for (t,s,r),n in sorted(exclusion.items())])
    ex.to_csv(HERE/'exclusions.csv', index=False)
    out = []
    for (task, study, fold), q in m.groupby(['task','study','fold']):
        out.append(dict(task=task, study=study, fold=int(fold), n_subjects=q.subject_id.nunique(),
            n_landmarks=len(q), n_history_landmarks=int(q.has_history.sum()),
            n_target_landmarks=int(q.event.eq(1).sum()),
            n_target_subjects=q.loc[q.event.eq(1),'subject_id'].nunique(),
            n_competing_landmarks=int(q.event.eq(2).sum()),
            n_uncertain_event_censored=int(q.uncertain_event.sum())))
    pd.DataFrame(out).to_csv(HERE/'cohort_support.csv', index=False)
    receipt = dict(status='BUILT_NOT_FITTED', input_sha256={str(p):sha(p) for p in (FOLDS,BASELINE,COGNITION)},
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('transitions.py'))},
        output_sha256={str(p.relative_to(HERE)):sha(p) for p in
                      (private/'metadata.csv',private/'features.npz',HERE/'exclusions.csv',HERE/'cohort_support.csv')},
        n_roster_development_subjects=len(b), n_risk_set_rows=len(m), n_subjects=m.subject_id.nunique(),
        task_subjects=m.groupby('task').subject_id.nunique().to_dict(),
        task_landmarks=m.groupby('task').size().to_dict(), current_names=list(CURRENT), history_names=list(HISTORY),
        reserved_fold_records_materialized=False, no_byte_isolation_claim=True,
        phenotype='PHC v18 local cognition DS 2024.12.16.v2', genotype='processed EUR ADSP NG00067.v17; original GRPa-PRS QC')
    (HERE/'DATA_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','n_risk_set_rows','n_subjects','task_subjects','task_landmarks')}), flush=True)


if __name__ == '__main__':
    build()
