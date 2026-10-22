"""Causal 1 s sampling and explicitly reconstructed charge-counter SOC reference."""
from pathlib import Path
import re, zipfile, json, hashlib
import numpy as np
import pandas as pd
from inspect_data import workbook_tables

ROOT = Path(__file__).resolve().parents[1]

def load_channel(blob):
    channels = []
    for name, table in workbook_tables(blob):
        if table and 'Current(A)' in table[0] and 'Test_Time(s)' in table[0]:
            df = pd.DataFrame(table[1:], columns=table[0])
            channels.append(df)
    if len(channels) != 1:
        raise ValueError(f'Expected exactly one measurement channel, got {len(channels)}')
    return channels[0]

def main():
    raw, processed = ROOT/'data/raw', ROOT/'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    capacity_archive = raw/'SP2_Initial_capacity_10_16_2015.zip'
    with zipfile.ZipFile(capacity_archive) as archive:
        blob = archive.read(archive.namelist()[0])
    capacity_df = load_channel(blob)
    # Step 6 is the first full 1 A discharge in the separate capacity-calibration run.
    full_discharge = capacity_df[capacity_df['Step_Index'] == 6]
    before = capacity_df[capacity_df['Step_Index'] < 6]['Discharge_Capacity(Ah)'].iloc[-1]
    q_ref = float(full_discharge['Discharge_Capacity(Ah)'].iloc[-1] - before)
    if not 1.5 < q_ref < 2.5:
        raise ValueError(f'Unexpected calibration capacity: {q_ref}')
    inventory = []
    for path in sorted(raw.glob('SP2_*C_*.zip')):
        match = re.fullmatch(r'SP2_(\d+)C_(DST|FUDS|US06|BJDST)\.zip', path.name)
        if not match:
            continue
        temp, profile = int(match[1]), match[2]
        with zipfile.ZipFile(path) as archive:
            for filename in archive.namelist():
                if not filename.lower().endswith('.xls'):
                    continue
                initial_level = int(re.search(r'_(50|80)SOC', filename)[1])
                df = load_channel(archive.read(filename))
                df = df.sort_values('Test_Time(s)', kind='stable').reset_index(drop=True)
                if df['Test_Time(s)'].duplicated().any():
                    raise ValueError(f'Duplicate time in {filename}')
                # These step meanings were verified in the original 25 C files.
                # Require supporting charge/current checks for every new record.
                full = df[df['Step_Index'] == 3]
                dynamic_idx = df.index[df['Step_Index'] == 7]
                if full.empty or dynamic_idx.empty or full['Voltage(V)'].iloc[-1] < 4.15:
                    raise ValueError(f'Unrecognized conditioning steps: {filename}')
                c_full = float(full['Charge_Capacity(Ah)'].iloc[-1])
                d_full = float(full['Discharge_Capacity(Ah)'].iloc[-1])
                dynamic = df.loc[dynamic_idx[0]:dynamic_idx[-1]].copy()
                t = dynamic['Test_Time(s)'].to_numpy(float)
                v = dynamic['Voltage(V)'].to_numpy(float)
                i = dynamic['Current(A)'].to_numpy(float)  # source sign: positive charge
                c = dynamic['Charge_Capacity(Ah)'].to_numpy(float)
                d = dynamic['Discharge_Capacity(Ah)'].to_numpy(float)
                soc = 1 + ((c - c_full) - (d - d_full)) / q_ref
                if np.any(np.diff(c) < -1e-7) or np.any(np.diff(d) < -1e-7):
                    raise ValueError(f'Capacity counters reset: {filename}')
                if not all(np.isfinite(x).all() for x in [t,v,i,soc]):
                    raise ValueError(f'Non-finite data: {filename}')
                # Causal zero-order hold. No centered interpolation or future samples.
                grid = t[0] + np.arange(int(np.floor(t[-1]-t[0])) + 1)
                pos = np.searchsorted(t, grid, side='right') - 1
                x = np.column_stack([v[pos], i[pos], np.full(len(pos), temp)])
                target = soc[pos]
                record_id = f'{profile}_{temp}C_{initial_level}SOC'
                segment = np.zeros(len(grid), dtype=int)
                # Do not build windows across intervals unsupported by a recent sample.
                valid = (grid - t[pos]) <= 5.0
                segment[1:] = np.cumsum((~valid[1:]) | (~valid[:-1]))
                np.savez_compressed(processed/f'{record_id}.npz', x=x.astype('float32'),
                    soc=target.astype('float32'), t=(grid-t[0]).astype('float32'),
                    valid=valid, segment=segment)
                outlier_count = int(((target < 0) | (target > 1)).sum())
                inventory.append(dict(record_id=record_id, profile=profile, temperature_c=temp,
                    initial_soc_label=initial_level, initial_soc_reference=float(target[0]),
                    final_soc_reference=float(target[-1]), n_samples=len(grid),
                    valid_samples=int(valid.sum()), max_raw_gap_s=float(np.diff(t).max()),
                    outside_0_1_samples=outlier_count, q_reference_ah=q_ref,
                    source_archive=path.name, source_member=filename,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    soc_reference='reconstructed from tester charge/discharge counters; fixed capacity from independent initial-capacity run',
                    temperature_feature='ambient chamber setting; not measured cell temperature'))
                print(record_id, len(grid), 'samples; SOC', round(target[0]*100,3),
                      'to', round(target[-1]*100,3), '%; out of range', outlier_count, flush=True)
    report = dict(q_reference_ah=q_ref, reference_capacity_scope='single initial capacity test at reference conditions; not temperature-adaptive',
        soc_formula='1 + ((charge_Ah-charge_at_full_Ah) - (discharge_Ah-discharge_at_full_Ah))/Q_reference_Ah',
        assumptions=['end of CCCV step 3 is full charge', 'coulombic efficiency is one',
                     'capacity reference stays constant across profiles and temperatures'],
        target_clipping=False, sampling='1 s causal last-observation hold; samples older than 5 s excluded',
        records=inventory)
    (processed/'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(inventory).to_csv(ROOT/'results/data_inventory.csv', index=False)

if __name__ == '__main__':
    main()
