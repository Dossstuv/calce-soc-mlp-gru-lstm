"""Create standalone scientific figures from actual recorded results."""
from pathlib import Path
import argparse, json, os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',default='pilot')
    args=parser.parse_args(); folder=ROOT/'results'/args.mode
    figures=folder/'figures'; figures.mkdir(exist_ok=True)
    m=pd.read_csv(folder/'macro_per_seed.csv')
    split='validation' if args.mode=='pilot' else 'test'
    s=m[m.split==split].groupby('model')['RMSE_pp'].agg(['mean','std']).reindex(['MLP','GRU','LSTM'])
    fig,ax=plt.subplots(figsize=(6.5,4))
    ax.bar(s.index,s['mean'],yerr=s['std'].fillna(0),color=['#8a9baa','#1a7891','#e49a38'],capsize=4)
    ax.set_ylabel('Macro RMSE (SOC percentage points)')
    ax.set_title('Pilot validation at 25 °C' if args.mode=='pilot' else 'SOC estimation on held-out load profiles')
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figures/'rmse_comparison.png',dpi=300); fig.savefig(figures/'rmse_comparison.pdf'); plt.close(fig)
    pred=pd.read_csv(folder/'predictions.csv')
    record='FUDS_25C_80SOC' if args.mode=='pilot' else 'US06_25C_80SOC'
    seed=int(pred.seed.min()); p=pred[(pred.record_id==record)&(pred.seed==seed)&(pred.split==split)]
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ref=p[p.model=='GRU']
    axes[0].plot(ref.t_s/60,ref.SOC_reference*100,'k--',label='Reconstructed reference',lw=1.5)
    for kind,color in [('MLP','#8a9baa'),('GRU','#1a7891'),('LSTM','#e49a38')]:
        q=p[p.model==kind]
        axes[0].plot(q.t_s/60,q.SOC_estimate*100,label=kind,color=color,alpha=.85,lw=.9)
        axes[1].plot(q.t_s/60,(q.SOC_estimate-q.SOC_reference)*100,color=color,lw=.7)
    axes[0].set_ylabel('SOC (%)'); axes[1].set_ylabel('Error (pp)'); axes[1].set_xlabel('Elapsed dynamic-profile time (min)')
    axes[0].set_title(f"{'FUDS' if args.mode=='pilot' else 'US06'}, 25 °C, nominal initial SOC 80%"); axes[0].legend(ncols=2,fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(figures/'soc_trace.png',dpi=300); fig.savefig(figures/'soc_trace.pdf'); plt.close(fig)
    inventory=json.loads((ROOT/'data/processed/inventory.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for temp,color in [(0,'#5371a0'),(25,'#1a7891'),(45,'#e49a38')]:
        with np.load(ROOT/'data/processed'/f'DST_{temp}C_80SOC.npz') as d:
            ax.plot(d['t']/60,d['soc']*100,label=f'{temp} °C',color=color,lw=1)
    ax.axhline(0,color='black',ls='--',lw=.8)
    ax.set(xlabel='Elapsed dynamic-profile time (min)',ylabel='Reconstructed SOC (%)',title='Reference audit: DST, nominal initial SOC 80%')
    ax.legend(title='Ambient setting'); ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(figures/'reference_audit.png',dpi=300); fig.savefig(figures/'reference_audit.pdf'); plt.close(fig)
    if args.mode!='pilot':
        metrics=pd.read_csv(folder/'metrics_per_record.csv')
        info=pd.DataFrame(inventory['records'])[['record_id','temperature_c','profile']]
        metrics=metrics[metrics.split=='test'].merge(info,on='record_id',validate='many_to_one')
        by_seed=metrics.groupby(['model','seed','temperature_c'],as_index=False).RMSE_pp.mean()
        grouped=by_seed.groupby(['model','temperature_c']).RMSE_pp.agg(['mean','std'])
        fig,ax=plt.subplots(figsize=(7,4)); temps=sorted(metrics.temperature_c.unique()); positions=np.arange(len(temps)); width=.24
        for offset,(kind,color) in enumerate([('MLP','#8a9baa'),('GRU','#1a7891'),('LSTM','#e49a38')]):
            group=grouped.loc[kind].reindex(temps)
            ax.bar(positions+(offset-1)*width,group['mean'],width,yerr=group['std'].fillna(0),label=kind,color=color,capsize=3)
        ax.set(xticks=positions,xticklabels=[str(t) for t in temps],ylabel='Macro RMSE (SOC percentage points)',xlabel='Ambient chamber setting (°C)',title='Held-out load profiles: US06 and BJDST')
        ax.legend(); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); fig.savefig(figures/'rmse_by_temperature.png',dpi=300); fig.savefig(figures/'rmse_by_temperature.pdf'); plt.close(fig)
    print(figures)

if __name__=='__main__':
    main()
