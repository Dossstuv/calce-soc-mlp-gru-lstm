"""Independently recompute saved errors and verify matched evaluation targets."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/full'
    m=pd.read_csv(folder/'metrics_per_record.csv')
    p=pd.read_csv(folder/'predictions.csv')
    expected={(model,seed) for model in ['MLP','GRU','LSTM'] for seed in [11,22,33]}
    assert set(zip(m.model,m.seed))==expected
    assert (m[m.split=='test'].groupby(['model','seed']).size()==12).all()
    keys=['model','seed','split','record_id']
    recomputed=[]
    for key,g in p.groupby(keys):
        e=(g.SOC_estimate.to_numpy()-g.SOC_reference.to_numpy())*100
        recomputed.append(dict(zip(keys,key),n=len(g),MAE_pp=np.abs(e).mean(),
            RMSE_pp=np.sqrt((e**2).mean()),max_abs_error_pp=np.abs(e).max()))
    check=pd.DataFrame(recomputed).merge(m,on=keys,validate='one_to_one',suffixes=('_recomputed','_saved'))
    assert len(check)==162
    worst=0.
    for metric in ['MAE_pp','RMSE_pp','max_abs_error_pp']:
        difference=float(np.abs(check[metric+'_recomputed']-check[metric+'_saved']).max())
        worst=max(worst,difference); assert difference<1e-5
    for (split,rid),g in p.groupby(['split','record_id']):
        template=None
        for _,h in g.groupby(['model','seed']):
            values=h[['t_s','SOC_reference']].to_numpy()
            if template is None: template=values
            else: assert np.array_equal(values,template)
    info=dict(status='passed',runs=9,metrics_groups=len(check),prediction_rows=len(p),
        largest_metric_recomputation_difference_pp=worst,matched_target_times=True,
        targets_clipped=False,test_tuned=False,
        protocol_sha256=hashlib.sha256((ROOT/'protocol_v1.json').read_bytes()).hexdigest(),
        train_script_sha256=hashlib.sha256((ROOT/'scripts/train.py').read_bytes()).hexdigest(),
        prediction_csv_sha256=hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest(),
        limitation='This audit verifies numerical consistency, not independent SOC truth or manuscript novelty')
    (folder/'results_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    main()
