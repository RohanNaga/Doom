"""Across-arena statistics for the adaptation study: survival of the half-gap crossing, interquartile means with an arena bootstrap, performance profiles. Run from the repo root on paper/tables/tuned/adapt_summary.json."""
import json, numpy as np, csv, glob, os
rng=np.random.default_rng(0)
s=json.load(open('paper/tables/tuned/adapt_summary.json')); per=s['per_arena']; home=s['home']; steps=[0,250,500,1000,2000,4000]
A=np.array([[r['A'][str(t)] for t in steps] for r in per]); A0=A[:,0]; line=(A0+home)/2
arenas=[r['arena'] for r in per]; n=len(arenas)
# survival: fraction crossed by each step (censored at 4000)
cross=np.array([next((t for t,a in zip(steps,row) if a>=l), np.inf) for row,l in zip(A,line)])
surv={t: float(np.mean(cross<=t)) for t in steps}
print('crossed by step', surv)
# arena bootstrap of the survival fraction at 4000 and the count
B=10000; idx=rng.integers(0,n,(B,n)); frac=np.mean(cross[idx]<=4000,axis=1)
print('9/13 arena-bootstrap 95%% CI on fraction: %.2f [%.2f, %.2f]'%(np.mean(cross<=4000), *np.percentile(frac,[2.5,97.5])))
# IQM of A per step with stratified (arena) bootstrap
def iqm(x): x=np.sort(x); k=len(x)//4; return x[k:len(x)-k].mean() if len(x)>=4 else x.mean()
for j,t in enumerate(steps):
    v=A[:,j]; bs=[iqm(v[idx[b]]) for b in range(2000)]
    print(f'step {t:5d} IQM {iqm(v):.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] median {np.median(v):.2f} mean {v.mean():.2f} min {v.min():.2f} max {v.max():.2f}')
# performance profile at 4k: fraction of arenas with A4k >= tau
for tau in [2.5,3.0,3.5,4.0,4.5,home]:
    print(f'profile at 4k: P(A>= {tau:.2f}) = {np.mean(A[:,-1]>=tau):.2f}   at step 0: {np.mean(A0>=tau):.2f}')
# gain and arena-bootstrap of median gain
gain=A[:,-1]-A0; bs=np.median(gain[idx],axis=1); print('median gain %.2f [%.2f,%.2f]; IQM gain %.2f'%(np.median(gain),*np.percentile(bs,[2.5,97.5]),iqm(gain)))
# median budget with censoring: arena bootstrap of the median crossing (inf allowed)
med=[np.median(cross[idx[b]]) for b in range(B)]; med=np.array(med); print('median budget %s; bootstrap: P(median censored)=%.2f, 2.5/97.5 pct of finite %s'%(np.median(cross), np.mean(~np.isfinite(med)), np.percentile(med[np.isfinite(med)],[2.5,97.5]) if np.isfinite(med).any() else None))
# episode bootstrap within arena on crossing at 4k: reuse per-window csv? report from the summary if present
flip=[r.get('cost_half_gap_flip') for r in per]; print('per-arena crossing flip info present:', any(flip))
# rho S0 vs A4k arena bootstrap
from scipy.stats import spearmanr
S0=np.array([r['S0'] for r in per]); rhos=[spearmanr(S0[idx[b]],A[idx[b],-1])[0] for b in range(2000)]
print('rho S0~A4k %.2f [%.2f,%.2f]'%(spearmanr(S0,A[:,-1])[0],*np.nanpercentile(rhos,[2.5,97.5])))
