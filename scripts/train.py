"""Reproducible MLP/GRU/LSTM comparisons with record-level data separation."""
from pathlib import Path
import argparse, copy, json, random, time, platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]

class Windows(Dataset):
    def __init__(self, records, mean, std, window, stride):
        self.window, self.arrays, self.locations = window, [], []
        for record in records:
            with np.load(ROOT/'data/processed'/f"{record['record_id']}.npz") as d:
                x = ((d['x'] - mean) / std).astype('float32')
                y, valid, segment, times = d['soc'], d['valid'], d['segment'], d['t']
                valid = valid & (y >= 0) & (y <= 1)
            idx = len(self.arrays)
            self.arrays.append((torch.from_numpy(x), torch.from_numpy(y.copy()), times.copy(), record))
            for end in range(window-1, len(x), stride):
                start = end-window+1
                if valid[start:end+1].all() and segment[start] == segment[end]:
                    self.locations.append((idx, end))
        if not self.locations:
            raise ValueError('No valid windows')
    def __len__(self):
        return len(self.locations)
    def __getitem__(self, index):
        rec, end = self.locations[index]
        x,y,_,_ = self.arrays[rec]
        return x[end-self.window+1:end+1], y[end]

class Estimator(nn.Module):
    def __init__(self, kind, window, hidden):
        super().__init__()
        self.kind = kind
        if kind == 'MLP':
            self.core = nn.Sequential(nn.Flatten(), nn.Linear(window*3,hidden), nn.Tanh())
        else:
            cls = nn.GRU if kind == 'GRU' else nn.LSTM
            self.core = cls(3,hidden,batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden,1), nn.Sigmoid())
    def forward(self,x):
        z = self.core(x)
        if self.kind != 'MLP':
            z = z[0][:,-1,:]
        return self.head(z).squeeze(-1)

def predict(model, dataset, batch):
    result = []
    model.eval()
    with torch.inference_mode():
        for x,_ in DataLoader(dataset,batch_size=batch,shuffle=False):
            result.append(model(x).numpy())
    return np.concatenate(result)

def score(y,p):
    e = (p-y)*100
    return dict(MAE_pp=float(np.mean(np.abs(e))), RMSE_pp=float(np.sqrt(np.mean(e**2))),
                max_abs_error_pp=float(np.max(np.abs(e))))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['pilot','full','cross-temperature'],default='pilot')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--seeds',type=int,nargs='+')
    args=parser.parse_args()
    config=json.loads((ROOT/'protocol_v1.json').read_text())
    inventory=json.loads((ROOT/'data/processed/inventory.json').read_text())
    records=inventory['records']
    temperatures=[25] if args.mode=='pilot' else [0,25,45]
    train_temps=[0,45] if args.mode=='cross-temperature' else temperatures
    test_temps=[25] if args.mode=='cross-temperature' else temperatures
    train_r=[r for r in records if r['profile']=='DST' and r['temperature_c'] in train_temps]
    val_r=[r for r in records if r['profile']=='FUDS' and r['temperature_c'] in train_temps]
    test_r=[] if args.mode=='pilot' else [r for r in records if r['profile'] in ['US06','BJDST'] and r['temperature_c'] in test_temps]
    if len(train_r)!=2*len(train_temps) or len(val_r)!=2*len(train_temps) or (args.mode!='pilot' and len(test_r)!=4*len(test_temps)):
        raise ValueError('Required records missing; finish download and preparation first')
    train_ids={r['record_id'] for r in train_r}; val_ids={r['record_id'] for r in val_r}; test_ids={r['record_id'] for r in test_r}
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    pooled=[]
    for record in train_r:
        with np.load(ROOT/'data/processed'/f"{record['record_id']}.npz") as d:
            mask=d['valid'] & (d['soc']>=0) & (d['soc']<=1)
            pooled.append(d['x'][mask])
    pooled=np.concatenate(pooled)
    mean=pooled.mean(0); std=pooled.std(0); std[std<1e-6]=1
    window=config['window_samples']; batch=config['batch_size']
    train=Windows(train_r,mean,std,window,config['train_stride'])
    val=Windows(val_r,mean,std,window,1)
    test=Windows(test_r,mean,std,window,1) if test_r else None
    seeds=args.seeds or ([11] if args.mode=='pilot' else config['seeds'])
    epochs=args.epochs or (5 if args.mode=='pilot' else config['epochs_max'])
    out=ROOT/'results'/args.mode; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run_info=dict(mode=args.mode,epochs_max=epochs,seeds=seeds,train_records=sorted(train_ids),
        validation_records=sorted(val_ids),test_records=sorted(test_ids),normalization_mean=mean.tolist(),
        normalization_std=std.tolist(),window=window,train_windows=len(train),validation_windows=len(val),
        test_windows=len(test) if test else 0,torch=torch.__version__,python=platform.python_version(),
        platform=platform.platform(),test_evaluated=bool(test),reference='reconstructed SOC; not independently measured')
    (out/'run_info.json').write_text(json.dumps(run_info,indent=2),encoding='utf-8')
    metrics=[]; predictions=[]; histories=[]
    for seed in seeds:
        for kind in config['models']:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            generator=torch.Generator().manual_seed(seed)
            loader=DataLoader(train,batch_size=batch,shuffle=True,generator=generator)
            model=Estimator(kind,window,config['hidden_units'])
            optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
            loss_fn=nn.MSELoss(); best=float('inf'); best_state=None; stale=0; best_epoch=0
            started=time.perf_counter()
            for epoch in range(1,epochs+1):
                model.train(); losses=[]
                for x,y in loader:
                    optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item())
                p=predict(model,val,batch)
                y=np.array([float(val.arrays[r][1][e]) for r,e in val.locations])
                vloss=float(np.mean((p-y)**2))
                histories.append(dict(model=kind,seed=seed,epoch=epoch,train_MSE=float(np.mean(losses)),validation_MSE=vloss))
                if vloss<best:
                    best=vloss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
                else:
                    stale+=1
                if epoch==1 or epoch%5==0:
                    print(args.mode,kind,seed,'epoch',epoch,'validation RMSE pp',round(np.sqrt(vloss)*100,4),flush=True)
                if stale>=config['early_stopping_patience']:
                    break
            training_s=time.perf_counter()-started
            model.load_state_dict(best_state)
            torch.save(dict(state_dict=best_state,kind=kind,window=window,hidden=config['hidden_units'],
                mean=mean.tolist(),std=std.tolist(),seed=seed),out/f'{kind}_seed{seed}.pt')
            for split,dataset in [('validation',val)]+([('test',test)] if test else []):
                p=predict(model,dataset,batch)
                # Warmed, batched CPU throughput, not embedded or single-sample latency.
                started=time.perf_counter(); predict(model,dataset,batch)
                infer_ms=(time.perf_counter()-started)*1000/len(dataset)
                frame=[]
                for k,(rec,end) in enumerate(dataset.locations):
                    _,labels,times,info=dataset.arrays[rec]
                    frame.append(dict(model=kind,seed=seed,split=split,record_id=info['record_id'],
                        profile=info['profile'],temperature_c=info['temperature_c'],t_s=float(times[end]),
                        SOC_reference=float(labels[end]),SOC_estimate=float(p[k])))
                frame=pd.DataFrame(frame); predictions.append(frame)
                for rid,g in frame.groupby('record_id'):
                    row=dict(model=kind,seed=seed,split=split,record_id=rid,n=len(g),best_epoch=best_epoch,
                        training_seconds=training_s,parameters=sum(p.numel() for p in model.parameters()),
                        batch_inference_ms_per_window=infer_ms)
                    row.update(score(g.SOC_reference.to_numpy(),g.SOC_estimate.to_numpy())); metrics.append(row)
                print(kind,seed,split,'macro RMSE pp',round(np.mean([r['RMSE_pp'] for r in metrics if r['model']==kind and r['seed']==seed and r['split']==split]),4),flush=True)
            pd.DataFrame(metrics).to_csv(out/'metrics_per_record.csv',index=False)
            pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
            pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    m=pd.DataFrame(metrics)
    macro=m.groupby(['model','seed','split'],as_index=False)[['MAE_pp','RMSE_pp','max_abs_error_pp']].mean()
    macro.to_csv(out/'macro_per_seed.csv',index=False)
    macro.groupby(['model','split'])[['MAE_pp','RMSE_pp']].agg(['mean','std']).to_csv(out/'summary.csv')
    print('Saved',out,flush=True)

if __name__=='__main__':
    main()
