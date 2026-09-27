"""Correlate the fresh-set distances (D, coverage, transfer gap) with the 13-arena adaptation outcomes; run from the repo root."""
import json, glob, csv
from scipy.stats import spearmanr
HOME=4.138
rows={}
for f in glob.glob('results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl'):
    rs=[json.loads(l) for l in open(f)]; rs=[r for r in rs if r['weights']=='live' and r['decoder']=='stock']
    m=rs[0]['map_id']; by={r['step']:r for r in rs}
    A0=by[0]['heldout_A_stock']; A4=by[4000]['heldout_A_stock']; half=(A0+HOME)/2
    cost=next((s for s in sorted(by) if by[s]['heldout_A_stock']>=half), None)
    rows[m]=dict(A0=A0,A4=A4,gain=A4-A0,cost=cost if cost is not None else 8000,cens=cost is None,S0=by[0]['heldout_latent_skill'],L0=by[0]['heldout_lpips_dec'],P0=by[0]['heldout_copy_psnr_dec'])
for r in csv.DictReader(open('results/transition_distance/compare_fresh_sd1.csv')):
    if r['set']=='arenas13':
        m=int(r['map']); rows[m].update(D=float(r['D']),cov=float(r['coverage']),G=float(r['G']),state=float(r['state_only']),Ltrain=float(r['L_train']))
maps=sorted(rows)
print('map  D     cov   G      A0    A4    gain  cost  S0    copyPSNR')
for m in maps:
    r=rows[m]; print(f"{m:3d} {r['D']:.3f} {r['cov']:.3f} {r['G']:.4f} {r['A0']:5.2f} {r['A4']:5.2f} {r['gain']:5.2f} {r['cost']:5d}{'c' if r['cens'] else ' '} {r['S0']:5.2f} {r['P0']:5.2f}")
print()
print('spearman (n=13), censored cost set to 8000')
for x in ['D','cov','G','state','Ltrain','S0','P0']:
    out=[]
    for y in ['A0','A4','gain','cost','S0']:
        rho,p=spearmanr([rows[m][x] for m in maps],[rows[m][y] for m in maps]); out.append(f"{y}:{rho:+.2f}(p{p:.2f})")
    print(f"{x:6s}", ' '.join(out))
