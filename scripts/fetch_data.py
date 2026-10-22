"""Download original CALCE archives; retain hashes and source URLs."""
from pathlib import Path
import argparse, hashlib, json, urllib.request, zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://web.calce.umd.edu/batteries/data/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperatures', nargs='+', type=int, default=[25])
    args = parser.parse_args()
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    names = ['SP2_Initial_capacity_10_16_2015.zip']
    names += [f'SP2_{t}C_{p}.zip' for t in args.temperatures for p in ['DST','FUDS','US06','BJDST']]
    manifest_path = raw / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        target = raw / name
        url = BASE + name
        if not target.exists():
            print('Downloading', name, flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Academic SOC research; Python urllib'})
            temp = target.with_suffix('.part')
            with urllib.request.urlopen(req, timeout=90) as response, temp.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not zipfile.is_zipfile(temp):
                raise ValueError(f'Not a ZIP archive: {name}')
            temp.replace(target)
        with zipfile.ZipFile(target) as archive:
            members = archive.namelist()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if name in manifest and manifest[name]['sha256'] != digest:
            raise ValueError(f'Source/cache hash changed for {name}; review provenance before replacing the manifest')
        manifest[name] = dict(url=url, bytes=target.stat().st_size,
            sha256=digest, members=members)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(name, target.stat().st_size, 'bytes;', len(members), 'members', flush=True)

if __name__ == '__main__':
    main()
