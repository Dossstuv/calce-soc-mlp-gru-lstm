"""Package the actual project code into a standalone Google Colab notebook."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def markdown(s):
    return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))

def code(s):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))

def main():
    scripts=['fetch_data.py','inspect_data.py','prepare_data.py','train.py','validate_pipeline.py','plot_results.py','audit_results.py','reference_sensitivity.py','build_manuscript.py']
    embedded={f'scripts/{name}':(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in scripts}
    embedded['protocol_v1.json']=(ROOT/'protocol_v1.json').read_text(encoding='utf-8')
    embedded['requirements.txt']=(ROOT/'requirements.txt').read_text(encoding='utf-8')
    cells=[markdown('''# Estimasi SOC CALCE: MLP, GRU, LSTM

Notebook penelitian komputasional. Satu sel SP20-2, empat profil, tiga suhu lingkungan, dua tingkat SOC awal nominal. Acuan SOC direkonstruksi dari kapasitas penguji dan kapasitas kalibrasi terpisah; bukan pengukuran SOC independen. NASA tidak dimasukkan sebelum audit terpisah.

**Penting:** sebagian acuan melewati nol pada akhir pengosongan. Nilainya dipertahankan untuk audit dan jendela di luar [0,1] dikeluarkan secara seragam, tanpa clipping target. Pembagian rekaman dilakukan sebelum pembentukan jendela. Hasil pilot adalah validasi, bukan hasil uji akhir. Tidak ada klaim arsitektur baru atau penerapan BMS.

Jalankan sel secara berurutan. Eksperimen utama dapat memerlukan beberapa menit atau lebih pada CPU. Script sengaja menggunakan CPU agar pembandingan waktu berada pada perangkat yang sama; waktu tidak mewakili ESP32.'''),
    code('''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy>=1.26,<3', 'pandas>=2.2,<4', 'openpyxl>=3.1,<4', 'xlrd==2.0.2', 'matplotlib>=3.8,<4', 'torch>=2.4,<3'])
'''),
    code("from pathlib import Path\nimport json, sys, subprocess\nROOT = Path('/content/soc_calce') if Path('/content').exists() else Path.cwd()/'soc_calce_notebook_run'\nROOT.mkdir(parents=True, exist_ok=True)\nembedded = "+repr(embedded)+"\nfor name, content in embedded.items():\n    path=ROOT/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content,encoding='utf-8')\ndef run(script,*args):\n    subprocess.check_call([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT)\nprint(ROOT)\n"),
    markdown('## 1. Unduh dan audit data asli\n\nSumber: https://calce.umd.edu/battery-data. Unduhan disertai SHA-256; berkas tidak ditimpa jika telah tersedia.'),
    code("run('fetch_data.py','--temperatures','0','25','45')\nrun('prepare_data.py')\nrun('validate_pipeline.py')\n"),
    markdown('## 2. Pilot: pelatihan dan validasi saja\n\nSatu seed, lima epoch, 25 C. Tidak mengevaluasi US06/BJDST.'),
    code("run('train.py','--mode','pilot')\nrun('plot_results.py','--mode','pilot')\n"),
    markdown('## 3. Eksperimen utama\n\nDST latih, FUDS validasi, US06/BJDST uji. Ketiga suhu ada dalam data latih. Tiga seed; maksimal 50 epoch; early stopping. Jangan memilih parameter berdasarkan hasil uji.'),
    code("run('train.py','--mode','full')\nrun('audit_results.py')\nrun('plot_results.py','--mode','full')\nrun('reference_sensitivity.py')\nrun('build_manuscript.py')\n"),
    code("import pandas as pd\ndisplay(pd.read_csv(ROOT/'results/full/summary.csv',header=[0,1],index_col=[0,1]))\nfrom IPython.display import Image, display\ndisplay(Image(filename=str(ROOT/'results/full/figures/rmse_comparison.png')))\ndisplay(Image(filename=str(ROOT/'results/full/figures/soc_trace.png')))\n"),
    markdown('## 4. Ekstensi opsional: suhu 25 C ditahan\n\nIni pelatihan baru, bukan evaluasi ulang model utama. DST/FUDS 0 dan 45 C dipakai untuk latih/validasi. US06/BJDST 25 C menjadi uji. Sel di bawah tidak perlu dijalankan untuk mereproduksi eksperimen utama.'),
    code("# Hapus komentar jika ingin menjalankan ekstensi.\n# run('train.py','--mode','cross-temperature')\n# run('plot_results.py','--mode','cross-temperature')\n"),
    markdown('## 5. Ekspor hasil\n\nSimpan konfigurasi, metrik, prediksi, dan model. Setiap angka naskah harus berasal dari keluaran aktual, bukan contoh.'),
    code("import shutil\nexport=shutil.make_archive(str(ROOT.parent/'SOC_CALCE_results'),'zip',ROOT/'results')\nprint(export)\ntry:\n    from google.colab import files\n    files.download(export)\nexcept ImportError:\n    pass\n")]
    for index,cell in enumerate(cells):
        cell['id']=f'cell-{index:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3'),language_info=dict(name='python')),cells=cells)
    (ROOT/'SOC_CALCE_Colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    for cell in cells:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook-cell>','exec')
    print('Notebook built; all code cells compile')

if __name__=='__main__':
    main()
