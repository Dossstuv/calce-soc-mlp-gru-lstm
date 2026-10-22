"""Exploratory reference-definition sensitivity; no retraining or test-based tuning.

Alternative capacities are scenarios, NOT measured tolerances or corrected truth.
All scenarios use the same common evaluation timestamps for a fair comparison.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/full'
    p=pd.read_csv(folder/'predictions.csv')
    p=p[p.split=='test'].copy()
    inv=json.loads((ROOT/'data/processed/inventory.json').read_text())
    qref=inv['q_reference_ah']
    capacities=[2.0,qref,2.06]
    labels=np.column_stack([1-(1-p.SOC_reference.to_numpy())*qref/q for q in capacities])
    valid=((labels>=0)&(labels<=1)).all(axis=1)
    p=p.loc[valid].copy(); labels=labels[valid]
    rows=[]
    for j,capacity in enumerate(capacities):
        p['alternate_reference']=labels[:,j]
        for (kind,seed,rid),g in p.groupby(['model','seed','record_id']):
            e=(g.SOC_estimate-g.alternate_reference).to_numpy()*100
            rows.append(dict(model=kind,seed=seed,record_id=rid,capacity_scenario_Ah=capacity,
                n_common_targets=len(g),MAE_pp=float(np.abs(e).mean()),RMSE_pp=float(np.sqrt((e**2).mean()))))
    frame=pd.DataFrame(rows)
    frame.to_csv(folder/'reference_sensitivity_per_record.csv',index=False)
    macro=frame.groupby(['model','seed','capacity_scenario_Ah'],as_index=False)[['MAE_pp','RMSE_pp']].mean()
    macro.to_csv(folder/'reference_sensitivity_macro_per_seed.csv',index=False)
    summary=macro.groupby(['model','capacity_scenario_Ah'])[['MAE_pp','RMSE_pp']].agg(['mean','std'])
    summary.to_csv(folder/'reference_sensitivity_summary.csv')
    note=dict(stage='exploratory post hoc analysis motivated by data audit; not preregistered',
        calibration_capacity_Ah=qref,capacity_scenarios_Ah=capacities,
        interpretation='Scenarios change the reference only. They are neither independent SOC truth nor measured capacity uncertainty. No model is retrained, selected or tuned.',
        timestamp_policy='same common-domain target timestamps for all three capacities, models and seeds',
        prediction_rows_before=len(valid),prediction_rows_after=int(valid.sum()),
        distinct_common_targets=len(p[['record_id','t_s']].drop_duplicates()))
    (folder/'reference_sensitivity_notes.json').write_text(json.dumps(note,indent=2))
    print(summary.to_string())

if __name__=='__main__':
    main()
