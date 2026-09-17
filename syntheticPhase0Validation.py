import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Generate the synthetic Phase 0 note-validation outputs.')
parser.add_argument('--outdir', default='phase0SyntheticOutputs', help='Output directory relative to the current working directory unless an absolute path is supplied.')
parser.add_argument('--seed', type=int, default=20260609, help='Random seed for the synthetic note-validation generator.')
args = parser.parse_args()

OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

roles_all = ['B','D','H','U','M','Y','Obs','I','excluded']
Apre = {'B','D','H','U'}
Adown = {'M','Y','Obs','I','excluded'}

# ---- Structural population for distinct analyses, not a common-law fiber ----
Npop = 200000
X = rng.normal(size=Npop)
J = 0.55*X + rng.normal(size=Npop)
O = 0.45*X + 0.55*J + rng.normal(size=Npop)
D = (rng.random(Npop) < 1/(1+np.exp(-(-0.2 + 0.45*X + 0.50*J - 0.25*O)))).astype(float)
M0 = 0.40*J + 0.20*X + 0.10*D + rng.normal(scale=0.7, size=Npop)
M1 = M0 + 0.65
base = 0.30 + 0.70*X + 0.65*D + 0.80*J + 0.35*O
# Y(a, m) = base + direct(a) + 0.45*m + error; direct(a)=a*(0.55+0.25 J)
total_effect = (0.55 + 0.25*J) + 0.45*(M1-M0)
direct_effect = 0.55 + 0.25*J
# observation process target weights
q = 1/(1+np.exp(-(0.25 + 0.65*O + 0.25*J + 0.20*D)))
psi_total = float(np.mean(total_effect))
psi_direct = float(np.mean(direct_effect))
psi_obs = float(np.sum(q*total_effect)/np.sum(q))
# Outcome proxy adjustment benchmark: large-population regression coefficient if conditioning on post-outcome proxy.
A = (rng.random(Npop) < 1/(1+np.exp(-(-0.15 + 0.45*X + 0.65*D + 0.75*J + 0.35*O)))).astype(float)
M = np.where(A==1, M1, M0)
Y = base + A*(0.55 + 0.25*J) + 0.45*M + rng.normal(scale=1.0, size=Npop)
Yproxy = Y + rng.normal(scale=0.4, size=Npop)
# Regression Y ~ A + X + D + O + Yproxy; standardize A coefficient via prediction contrast.
Xmat = np.column_stack([np.ones(Npop), A, X, D, O, Yproxy])
beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0]
X1 = Xmat.copy(); X1[:,1]=1
X0 = Xmat.copy(); X0[:,1]=0
psi_yproxy = float(np.mean(X1@beta - X0@beta))
role_values = pd.DataFrame([
    {'role_interpretation':'certified pre treatment state/design', 'symbol':'B/D/H/U', 'analysis_value':psi_total, 'calculation':'total effect over prospective pre treatment information'},
    {'role_interpretation':'mediator lock', 'symbol':'M', 'analysis_value':psi_direct, 'calculation':'direct component when post treatment mediator is held fixed'},
    {'role_interpretation':'outcome proxy adjustment', 'symbol':'Y', 'analysis_value':psi_yproxy, 'calculation':'large-population regression contrast after conditioning on outcome proxy'},
    {'role_interpretation':'observation weighted target', 'symbol':'Obs', 'analysis_value':psi_obs, 'calculation':'total effect weighted by synthetic observation mechanism'},
])
role_values.to_csv(OUT/'roleSpecificAnalysisValues.csv', index=False)

# ---- Synthetic note vignettes and typed lift validation ----
N = 120
patients=[]
features=[]
for i in range(N):
    x = rng.normal(); j = 0.55*x+rng.normal(); o = 0.45*x+0.55*j+rng.normal()
    d = rng.random() < 1/(1+np.exp(-(-0.2 + 0.45*x + 0.50*j - 0.25*o)))
    frail = j > 0.4
    high_monitor = o > 0.3
    # treatment decision time 0. create five candidate note features
    patient_id=f'SYN{i+1:03d}'
    patients.append({'patient_id':patient_id,'X':x,'J':j,'O':o,'D':int(d)})
    entries = [
      ('preClinicFrailty','B','preClinicNote','pre','pre','pre','Patient walks one block and needs help with bathing before treatment.' if frail else 'Patient independent with activities before treatment.'),
      ('orderRationale','D','orderHistory','pre','pre','pre','Treatment selected after renal function and regimen availability were reviewed.' if d else 'Standard regimen available; no special restriction recorded.'),
      ('lateHistorySummary','B','dischargeSummary','pre','post','post','Discharge summary states patient had poor mobility before treatment.' if frail else 'Discharge summary states baseline function was good before treatment.'),
      ('responseSummary','M','postProgressNote','post','post','post','After treatment, symptoms improved and clinician noted early response.'),
      ('monitoringIntensity','Obs','labOrderStream','mixed','mixed','mixed','Laboratory checks were ordered every six hours during early follow up.' if high_monitor else 'Routine daily monitoring was ordered.'),
    ]
    for name,true_role,source,clin,rec,avail,text in entries:
        features.append({'patient_id':patient_id,'feature':name,'true_role':true_role,'true_source':source,
                         'true_clinical_relation':clin,'true_recording_relation':rec,'true_availability_relation':avail,'source_text':text})
notes=pd.DataFrame(features)
notes.to_csv(OUT/'syntheticNoteVignettes.csv', index=False)

def sampled_roles(true_role, feature, audit=False):
    u = rng.random()
    # role sets generated from fixed probabilities. audit stage is sharper but remains conservative.
    if not audit:
        if true_role=='B':
            choices=[({'B'},0.72), ({'B','Obs'},0.10), ({'B','M'},0.06), ({'B','Y'},0.07), ({'Y'},0.05)]
        elif true_role=='D':
            choices=[({'D'},0.75), ({'B','D'},0.12), ({'D','Obs'},0.07), ({'B'},0.03), ({'Y'},0.03)]
        elif true_role=='M':
            choices=[({'M'},0.72), ({'B','M'},0.16), ({'M','Y'},0.08), ({'B'},0.04)]
        elif true_role=='Y':
            choices=[({'Y'},0.78), ({'B','Y'},0.16), ({'M','Y'},0.06)]
        elif true_role=='Obs':
            choices=[({'Obs'},0.68), ({'B','Obs'},0.20), ({'D','Obs'},0.07), ({'B'},0.05)]
        else:
            choices=[({'excluded'},1.0)]
    else:
        if true_role in Apre:
            choices=[({true_role},0.88), ({true_role,'Obs'},0.07), ({true_role,'Y'},0.02), ({'Y'},0.03)]
        elif true_role=='Obs':
            choices=[({'Obs'},0.86), ({'B','Obs'},0.10), ({'B'},0.04)]
        elif true_role=='M':
            choices=[({'M'},0.88), ({'B','M'},0.08), ({'B'},0.04)]
        elif true_role=='Y':
            choices=[({'Y'},0.90), ({'B','Y'},0.08), ({'B'},0.02)]
        else:
            choices=[({'excluded'},1.0)]
    cum=0
    for s,p in choices:
        cum += p
        if u <= cum: return set(s)
    return set(choices[-1][0])

def maybe_correct(true_value, categories, p_correct):
    if rng.random()<p_correct: return true_value, True
    alt=[x for x in categories if x!=true_value]
    return rng.choice(alt), False

records=[]
instability_count=0
for _,row in notes.iterrows():
    true_role=row['true_role']
    # typed lift stage
    src_correct = rng.random() < (0.96 if row['true_source']!='dischargeSummary' else 0.91)
    clin_pred, clin_ok = maybe_correct(row['true_clinical_relation'], ['pre','post','mixed'], 0.93)
    rec_pred, rec_ok = maybe_correct(row['true_recording_relation'], ['pre','post','mixed'], 0.95)
    avail_pred, avail_ok = maybe_correct(row['true_availability_relation'], ['pre','post','mixed'], 0.88)
    rset=sampled_roles(true_role,row['feature'],audit=False)
    # late history needs source separation; AI fails some.
    source_sep_required = (row['true_recording_relation']=='post' and row['true_clinical_relation']=='pre')
    source_sep_pass = (rng.random()<0.80) if source_sep_required else True
    if source_sep_required and not source_sep_pass:
        rset = rset | {'Y'}
    # audit stage
    aud_src_correct = src_correct or (rng.random()<0.85)
    aud_clin_ok = clin_ok or (rng.random()<0.80)
    aud_rec_ok = rec_ok or (rng.random()<0.85)
    aud_avail_ok = avail_ok or (rng.random()<0.78)
    aud_rset=sampled_roles(true_role,row['feature'],audit=True)
    aud_source_sep_pass = source_sep_pass or ((rng.random()<0.70) if source_sep_required else True)
    if source_sep_required and not aud_source_sep_pass:
        aud_rset = aud_rset | {'Y'}
    # Instability is generated as declared extractor variation before version and seed locking.
    # It is intentionally separate from role coverage: a feature can cover the true role but remain unstable.
    base_instability = 0.12 if true_role in {'B','D','H','U'} else 0.18
    if row['true_recording_relation']=='post' or row['true_clinical_relation']=='mixed':
        base_instability += 0.08
    unstable = rng.random() < base_instability
    records.append({**row.to_dict(),
                    'ai_source_correct':src_correct,'ai_clock_correct':clin_ok and rec_ok,'ai_availability_correct':avail_ok,
                    'ai_role_set':';'.join(sorted(rset)), 'ai_true_role_covered':true_role in rset,
                    'ai_point_admitted': len(rset)>0 and rset.issubset(Apre),
                    'ai_false_admissible': (len(rset)>0 and rset.issubset(Apre) and true_role not in Apre),
                    'source_separation_required':source_sep_required,'ai_source_separation_pass':source_sep_pass,
                    'ai_unstable':unstable,
                    'audit_source_correct':aud_src_correct,'audit_clock_correct':aud_clin_ok and aud_rec_ok,
                    'audit_availability_correct':aud_avail_ok,'audit_role_set':';'.join(sorted(aud_rset)),
                    'audit_true_role_covered':true_role in aud_rset,
                    'audit_point_admitted': len(aud_rset)>0 and aud_rset.issubset(Apre),
                    'audit_false_admissible': (len(aud_rset)>0 and aud_rset.issubset(Apre) and true_role not in Apre),
                    'audit_source_separation_pass':aud_source_sep_pass})
res=pd.DataFrame(records)
res.to_csv(OUT/'phase0FeatureLevelResults.csv', index=False)

def prop(series):
    return float(np.mean(series)) if len(series)>0 else np.nan
n_features=len(res)
ai_point=res[res['ai_point_admitted']]
audit_point=res[res['audit_point_admitted']]
sep=res[res['source_separation_required']]
metrics = []
for stage,prefix in [('AI typed lift','ai'),('AI plus audit','audit')]:
    point = res[res[f'{prefix}_point_admitted']]
    metrics.append({'stage':stage,'source_pointer_precision':prop(res[f'{prefix}_source_correct']),
                    'clock_agreement':prop(res[f'{prefix}_clock_correct']),
                    'availability_agreement':prop(res[f'{prefix}_availability_correct']),
                    'role_coverage':prop(res[f'{prefix}_true_role_covered']),
                    'false_admissible_rate':prop(point[f'{prefix}_false_admissible']) if len(point)>0 else 0.0,
                    'source_separation_failure_rate':1-prop(sep[f'{prefix}_source_separation_pass']) if len(sep)>0 else np.nan,
                    'extractor_instability':prop(res['ai_unstable']) if prefix=='ai' else 0.0,
                    'point_admitted_fraction':prop(res[f'{prefix}_point_admitted']),
                    'features':n_features})
metrics_df=pd.DataFrame(metrics)
metrics_df.to_csv(OUT/'phase0ValidationMetrics.csv', index=False)

# Prespecified analysis lists: descriptive spread, not a fiber or audit-calibrated radius.
values={row['symbol']: row['analysis_value'] for _,row in role_values.iterrows()}
# These are distinct targets and regression coefficients, not one locked estimand.
routing_values={
    'Flat table K0': [values['B/D/H/U'], values['M'], values['Y'], values['Obs']],
    'AI typed lift Ktheta': [values['B/D/H/U'], values['M'], values['Obs']],
    'AI plus Phase0 audit Ktheta∨E': [values['B/D/H/U'], values['Obs']],
}
spread_rows=[]
for st, vals in routing_values.items():
    lo=float(min(vals)); hi=float(max(vals)); rad=(hi-lo)/2; center=(hi+lo)/2
    spread_rows.append({'routing_stage':st,'lower':lo,'upper':hi,'center':center,'half_range':rad,'squared_half_range':rad**2,'retained_analysis_values':';'.join(f'{v:.3f}' for v in vals)})
spread_df=pd.DataFrame(spread_rows)
spread_df.to_csv(OUT/'phase0AnalysisSpread.csv', index=False)

print(metrics_df.round(3).to_string(index=False))
print(spread_df.round(3).to_string(index=False))


with open(OUT / 'phase0SeedSchedule.json', 'w', encoding='utf-8') as f:
    json.dump({'syntheticPhase0Seed': args.seed, 'nSyntheticPatients': N, 'nCandidateFeatures': int(len(res))}, f, indent=2)
print(f'Wrote synthetic Phase 0 validation outputs to {OUT.resolve()}')
