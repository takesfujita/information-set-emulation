"""Exact finite-support check of the EHR compression-drift identity (Table 11).
X,J,O independent uniform {-1,1}; D independent Bernoulli(1/2).
Mechanisms and outcome means use manuscript coefficients; no sampling error.
"""
from itertools import product
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def expit(x):
    return 1/(1+np.exp(-x))

def calculate():
    df=pd.DataFrame(product([-1.,1.],[0.,1.],[-1.,1.],[-1.,1.]),columns=['X','D','J','O'])
    X,D,J,O=(df[c].to_numpy() for c in ['X','D','J','O'])
    df['probability']=1/len(df)
    df['Z']=X+J
    s=expit(1.20+.35*X+.55*D+.45*O)
    pi=expit(-.25+.60*X+.85*D+1.05*J+.60*O)
    rows,cells=[],[]
    for feature,keys in [('Compressed score',['Z']),('Richer reconstruction strata',['Z','D','O'])]:
        for arm in [0,1]:
            mu=.30+.75*X+.80*D+J+.65*O+arm*(.60+.30*J)
            rho=expit(1.05+.25*arm+.45*X+.50*J+.75*O+.35*D)
            df['mu'],df['q']=mu,s*(pi if arm else 1-pi)*rho
            armcells=[]
            for key,cell in df.groupby(keys,sort=True):
                key=key if isinstance(key,tuple) else (key,)
                key='|'.join(f'{float(x):g}' for x in key)
                p=cell.probability.to_numpy()
                q,m=cell.q.to_numpy(),cell.mu.to_numpy()
                mass=p.sum(); w=p/mass
                em,eq=np.dot(w,m),np.dot(w,q)
                selected=np.dot(w,m*q)/eq
                drift=selected-em
                formula=np.dot(w,(m-em)*(q-eq))/eq
                row=dict(feature=feature,arm=arm,stratum=key,stratum_probability=mass,
                         selection_probability=eq,selected_mean=selected,population_mean=em,
                         drift=drift,covariance_formula=formula,absolute_error=abs(drift-formula))
                cells.append(row); armcells.append(row)
            err=max(x['absolute_error'] for x in armcells)
            assert err<1e-12
            rows.append(dict(feature=feature,arm=arm,
                direct_abs_drift=sum(x['stratum_probability']*abs(x['drift']) for x in armcells),
                formula_abs_drift=sum(x['stratum_probability']*abs(x['covariance_formula']) for x in armcells),
                max_identity_error=err,strata=len(armcells)))
    return pd.DataFrame(rows),pd.DataFrame(cells),df[['X','D','J','O','probability','Z']]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',default='iseSim/final')
    p.add_argument('--tabledir',default='../tables')
    args=p.parse_args()
    out,tables=Path(args.outdir),Path(args.tabledir)
    out.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    summary,cells,support=calculate()
    summary.to_csv(out/'compressionDriftSummary.csv',index=False)
    cells.to_csv(out/'compressionDriftCells.csv',index=False)
    support.to_csv(out/'compressionDriftSupport.csv',index=False)
    (out/'compressionDriftSpecification.json').write_text(json.dumps({
        'method':'exact enumeration','supportSize':16,'seed':None,
        'distribution':'independent uniform X,J,O in {-1,1}; independent D in {0,1}',
        'compressed':'X+J','richer':'(X+J,D,O)',
        'summary':'population-stratum-probability weighted absolute conditional drift',
        'maxError':'maximum absolute direct-minus-covariance error across all strata'},indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Feature & Arm & Direct abs drift & Formula abs drift & Max error \\',r'\midrule']
    for row in summary.itertuples():
        lines.append(f'{row.feature} & {row.arm} & {row.direct_abs_drift:.6f} & '
                     f'{row.formula_abs_drift:.6f} & $<10^{{-12}}$ '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (tables/'TableCompressionDrift.tex').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
