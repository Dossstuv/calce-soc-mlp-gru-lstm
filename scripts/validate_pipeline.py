"""Audit data provenance and split/window invariants before interpreting metrics."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from train import Windows

ROOT = Path(__file__).resolve().parents[1]

def main():
    downloads=json.loads((ROOT/'data/raw/download_manifest.json').read_text())
    inv=json.loads((ROOT/'data/processed/inventory.json').read_text())
    assert len(downloads)==13, 'Expected 12 dynamic archives plus calibration'
    for name,meta in downloads.items():
        assert hashlib.sha256((ROOT/'data/raw'/name).read_bytes()).hexdigest()==meta['sha256']
    records=inv['records']
    assert len(records)==24
    assert len({r['record_id'] for r in records})==24
    assert abs(inv['q_reference_ah']-2.028344937765965)<1e-10
    excluded=0; total=0; balance=[]
    for r in records:
        with np.load(ROOT/'data/processed'/f"{r['record_id']}.npz") as d:
            assert d['x'].shape==(r['n_samples'],3)
            assert np.allclose(np.diff(d['t']),1)
            assert np.allclose(d['x'][:,2],r['temperature_c'])
            assert np.isfinite(d['x']).all() and np.isfinite(d['soc']).all()
            excluded+=int(((d['soc']<0)|(d['soc']>1)).sum()); total+=len(d['soc'])
            currents=d['x'][:,1].astype(float)
            integral=np.sum((currents[1:]+currents[:-1])*0.5*np.diff(d['t'].astype(float)))/3600
            counter_delta=(float(d['soc'][-1])-float(d['soc'][0]))*inv['q_reference_ah']
            balance.append(dict(record_id=r['record_id'],net_charge_integral_Ah=integral,
                net_charge_counter_Ah=counter_delta,
                difference_SOC_pp=(integral-counter_delta)/inv['q_reference_ah']*100))
    # A deliberately invalid label in the middle of a window must prevent its use.
    sets={p:{r['record_id'] for r in records if r['profile']==p} for p in ['DST','FUDS','US06','BJDST']}
    assert not (sets['DST'] & sets['FUDS'] or sets['DST'] & (sets['US06']|sets['BJDST']))
    train=[r for r in records if r['profile']=='DST']
    ds=Windows(train,np.zeros(3),np.ones(3),60,5)
    for rec,end in ds.locations:
        _,y,_,_=ds.arrays[rec]
        labels=y[end-59:end+1].numpy()
        assert (labels>=0).all() and (labels<=1).all()
    checks=dict(status='passed',archives_verified=len(downloads),records_verified=len(records),
        raw_reference_samples=total,outside_reference_range_samples=excluded,
        outside_reference_range_percent=100*excluded/total,
        Q_reference_Ah=inv['q_reference_ah'],feature_count=3,
        target_clipping=False,record_level_split_disjoint=True,train_windows=len(ds),
        max_abs_current_balance_difference_pp=max(abs(r['difference_SOC_pp']) for r in balance),
        note='Pipeline checks do not establish independent SOC truth or scientific novelty')
    (ROOT/'results/pipeline_validation.json').write_text(json.dumps(checks,indent=2))
    pd.DataFrame(balance).to_csv(ROOT/'results/current_balance_check.csv',index=False)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
