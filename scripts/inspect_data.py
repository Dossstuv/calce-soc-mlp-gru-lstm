"""Inspect original CALCE Arbin files without assuming a SOC column exists."""
from pathlib import Path
import sys, zipfile, json, collections, io
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
import xlrd
import openpyxl

def workbook_tables(blob):
    if blob[:2] == b'PK':
        book = openpyxl.load_workbook(io.BytesIO(blob), read_only=True, data_only=True)
        for sheet in book:
            yield sheet.title, list(sheet.values)
        book.close()
    else:
        book = xlrd.open_workbook(file_contents=blob)
        for sheet in book.sheets():
            yield sheet.name, [sheet.row_values(i) for i in range(sheet.nrows)]

def inspect(path):
    output = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith('.xls'):
                continue
            record = dict(archive=path.name, file=name, sheets=[])
            for sheet_name, table in workbook_tables(archive.read(name)):
                info = dict(name=sheet_name, rows=len(table), columns=len(table[0]) if table else 0,
                            first_rows=table[:3])
                head = table[0] if table else []
                if 'Step_Index' in head and 'Current(A)' in head:
                    index = {h:i for i,h in enumerate(head)}
                    groups = collections.defaultdict(list)
                    for row in table[1:]:
                        groups[row[index['Step_Index']]].append(row)
                    info['step_summary'] = []
                    for step,rows in groups.items():
                        entry = dict(step=step, n=len(rows))
                        for key in ['Test_Time(s)','Current(A)','Voltage(V)','Charge_Capacity(Ah)','Discharge_Capacity(Ah)']:
                            if key in index:
                                vals = [r[index[key]] for r in rows]
                                entry[key] = dict(first=vals[0], last=vals[-1], min=min(vals), max=max(vals))
                        info['step_summary'].append(entry)
                record['sheets'].append(info)
            output.append(record)
    return output

if __name__ == '__main__':
    result = []
    for path in sorted((ROOT/'data/raw').glob('*.zip')):
        print('Inspecting', path.name, flush=True)
        result.extend(inspect(path))
    out = ROOT/'results/data_inspection.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    for record in result:
        print(record['file'])
        for sheet in record['sheets']:
            print(' ', sheet['name'], sheet['rows'], 'rows;', sheet['first_rows'][:1])
            for step in sheet.get('step_summary',[]):
                print('  ', json.dumps(step))
