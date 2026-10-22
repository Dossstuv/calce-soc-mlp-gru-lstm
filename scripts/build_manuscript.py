"""Build a review draft from completed, audited experiments; never invent results."""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/full'
    m=pd.read_csv(folder/'metrics_per_record.csv')
    test=m[m.split=='test'].copy()
    expected={(kind,seed) for kind in ['MLP','GRU','LSTM'] for seed in [11,22,33]}
    if set(zip(test.model,test.seed))!=expected or not (test.groupby(['model','seed']).size()==12).all():
        raise ValueError('Main experiment incomplete; no final manuscript generated')
    run=json.loads((folder/'run_info.json').read_text())
    audit=json.loads((ROOT/'results/pipeline_validation.json').read_text())
    inv=pd.read_csv(ROOT/'results/data_inventory.csv')
    macro=test.groupby(['model','seed'],as_index=False)[['MAE_pp','RMSE_pp']].mean()
    stats=macro.groupby('model')[['MAE_pp','RMSE_pp']].agg(['mean','std'])
    best=stats['RMSE_pp']['mean'].idxmin()
    def number(x): return f'{x:.3f}'
    def pm(kind,metric):
        return f"{stats.loc[kind,(metric,'mean')]:.3f} ± {stats.loc[kind,(metric,'std')]:.3f}"
    table=['| Model | MAE makro (pp) | RMSE makro (pp) | Kesalahan absolut terbesar (pp) | Parameter |',
           '|---|---:|---:|---:|---:|']
    compute=['| Model | Waktu pelatihan termasuk validasi (s) | Throughput batch CPU (ms/jendela) |',
             '|---|---:|---:|']
    for kind in ['MLP','GRU','LSTM']:
        g=test[test.model==kind]
        table.append(f"| {kind} | {pm(kind,'MAE_pp')} | {pm(kind,'RMSE_pp')} | {g.max_abs_error_pp.max():.3f} | {int(g.parameters.iloc[0]):,} |")
        per_seed=g.drop_duplicates('seed')
        compute.append(f"| {kind} | {per_seed.training_seconds.mean():.2f} ± {per_seed.training_seconds.std():.2f} | {per_seed.batch_inference_ms_per_window.median():.4f} |")
    merged=test.merge(inv[['record_id','temperature_c','profile']],on='record_id',validate='many_to_one')
    bytemp=merged.groupby(['model','seed','temperature_c'],as_index=False).RMSE_pp.mean()
    tstats=bytemp.groupby(['model','temperature_c']).RMSE_pp.agg(['mean','std'])
    temp_table=['| Model | 0 °C | 25 °C | 45 °C |','|---|---:|---:|---:|']
    for kind in ['MLP','GRU','LSTM']:
        temp_table.append('| '+kind+' | '+' | '.join(f"{tstats.loc[(kind,t),'mean']:.3f} ± {tstats.loc[(kind,t),'std']:.3f}" for t in [0,25,45])+' |')
    span=inv.groupby('temperature_c').agg(records=('record_id','count'),samples=('n_samples','sum'),outside=('outside_0_1_samples','sum'))
    audit_table=['| Suhu lingkungan | Rekaman | Sampel 1 detik | Acuan di luar 0–100% |','|---|---:|---:|---:|']
    for t,g in span.iterrows():
        audit_table.append(f"| {t} °C | {int(g.records):,} | {int(g.samples):,} | {int(g.outside):,} |")
    results_text='; '.join(f"{kind} {pm(kind,'RMSE_pp')} pp" for kind in ['MLP','GRU','LSTM'])
    sensitivity=pd.read_csv(folder/'reference_sensitivity_macro_per_seed.csv')
    sensitivity_stats=sensitivity.groupby(['capacity_scenario_Ah','model']).RMSE_pp.agg(['mean','std'])
    sensitivity_notes=json.loads((folder/'reference_sensitivity_notes.json').read_text())
    sensitivity_table=['| Kapasitas skenario (Ah) | MLP | GRU | LSTM |','|---|---:|---:|---:|']
    for q in sorted(sensitivity.capacity_scenario_Ah.unique()):
        sensitivity_table.append(f'| {q:.5f} | '+' | '.join(f"{sensitivity_stats.loc[(q,k),'mean']:.3f} ± {sensitivity_stats.loc[(q,k),'std']:.3f}" for k in ['MLP','GRU','LSTM'])+' |')
    scenario_winners=[]
    for q in sorted(sensitivity.capacity_scenario_Ah.unique()):
        winner=sensitivity_stats.loc[q]['mean'].idxmin()
        scenario_winners.append(f'{q:.5f} Ah: {winner}')
    recurrent_difference=abs(stats.loc['GRU',('RMSE_pp','mean')]-stats.loc['LSTM',('RMSE_pp','mean')])
    manuscript=f'''# Evaluasi Estimasi Muatan Baterai Litium-Ion pada Variasi Profil Beban dan Suhu

**Evaluation of Lithium-Ion Battery State-of-Charge Estimation across Load Profiles and Ambient Temperatures**

Nama penulis, afiliasi, dan penulis korespondensi: **belum diisi**. Hapus bagian ini dari berkas review anonim sesuai template jurnal.

> Status: draf penelitian komputasional untuk peninjauan penulis. Angka berasal dari pelatihan nyata pada dataset CALCE, bukan contoh atau hasil SOH terdahulu. Belum diterapkan ke template Mestro dan belum dinyatakan siap dikirim. NASA tidak digunakan dalam hasil berikut.

## Abstrak

Estimasi muatan baterai perlu diuji pada profil beban yang berbeda dari data pelatihan dengan definisi acuan yang dapat ditelusuri. Penelitian ini membandingkan multilayer perceptron, gated recurrent unit, dan long short-term memory pada 24 rekaman satu sel litium-ion CALCE, meliputi empat profil beban, tiga suhu lingkungan, dan dua kondisi muatan awal nominal. Acuan muatan direkonstruksi dari pencatatan kapasitas penguji menggunakan kapasitas kalibrasi terpisah sebesar {audit['Q_reference_Ah']:.5f} Ah. Audit menemukan {audit['outside_reference_range_samples']:,} dari {audit['raw_reference_samples']:,} sampel acuan berada di luar domain 0–100%; label tersebut dipertahankan untuk audit dan jendela terkait dikeluarkan tanpa clipping. Profil DST dipakai untuk pelatihan, FUDS untuk validasi, serta US06 dan BJDST untuk pengujian. Semua model memakai konteks kausal 60 detik dan dijalankan pada tiga seed. RMSE makro pengujian diperoleh sebagai berikut: {results_text}. {best} memiliki rerata RMSE terendah pada konfigurasi yang diuji. Temuan dibatasi pada satu sel, suhu yang telah diwakili dalam pelatihan, dan acuan berbasis pencatatan muatan. Hasil menunjukkan perlunya menilai kesalahan estimasi bersama keterbatasan acuan dan kebutuhan komputasi.

**Kata kunci:** estimasi muatan baterai; profil beban dinamis; jaringan saraf; suhu lingkungan; audit data.

## Abstract

Battery state-of-charge estimation requires evaluation on load profiles distinct from the training data and a traceable reference definition. This study compares a multilayer perceptron, a gated recurrent unit, and a long short-term memory network using 24 records from one CALCE lithium-ion cell, covering four load profiles, three ambient temperatures, and two nominal initial charge levels. Reference state of charge is reconstructed from tester capacity counters using a separate initial-capacity calibration of {audit['Q_reference_Ah']:.5f} Ah. The audit identifies {audit['outside_reference_range_samples']:,} of {audit['raw_reference_samples']:,} samples outside the 0–100% reference domain; original labels are retained and affected windows are excluded without clipping. DST is used for training, FUDS for validation, and US06 and BJDST for testing. All models use a causal 60-second context and three random seeds. Test macro-RMSE values are {results_text}. {best} yields the lowest mean macro-RMSE for the evaluated configuration. Findings are limited to one cell, temperatures represented during training, and a charge-counter-based reference. The results emphasize interpreting estimation errors together with reference limitations and computational requirements.

**Keywords:** state-of-charge estimation; dynamic load profile; neural network; ambient temperature; data audit.

## 1. Pendahuluan

State of charge (SOC) menyatakan tingkat muatan baterai relatif terhadap kapasitas acuan yang ditetapkan. Dalam pengembangan estimator, pemilihan kapasitas acuan berhubungan langsung dengan label target dan interpretasi kesalahan. Zheng et al. menunjukkan bahwa hubungan tegangan rangkaian terbuka dan SOC serta pengaruh suhu perlu diperhatikan ketika mengembangkan estimator daring [1]. Oleh sebab itu, kesalahan numerik yang rendah terhadap suatu label harus dibaca bersama cara label tersebut diperoleh.

Pendekatan pembelajaran mesin memungkinkan pemetaan riwayat sinyal listrik ke SOC. Chemali et al. telah menerapkan long short-term memory (LSTM) untuk estimasi SOC baterai [2]. Literatur yang lebih baru juga mengembangkan kombinasi temporal convolutional network dan unscented Kalman filter pada data CALCE [3], serta model TRNN-CTA untuk mengekstraksi informasi temporal [4]. Batool et al. mengevaluasi kerangka fuzzy–LSTM dengan profil beban CALCE pada beberapa suhu [5]. Dengan demikian, penggunaan jaringan rekuren, CALCE, atau pengujian lintas profil tidak dinyatakan sebagai kebaruan tersendiri dalam penelitian ini.

Pertanyaan yang dikaji adalah bagaimana tiga konfigurasi estimator relatif sederhana berbeda ketika diuji pada profil beban yang tidak dipakai untuk pelatihan, dengan konteks masukan dan aturan pemisahan yang sama. Multilayer perceptron (MLP) dipakai sebagai pembanding tanpa keadaan rekuren. Gated recurrent unit (GRU) dan LSTM mewakili dua arsitektur rekuren yang telah dikenal [6], [7]. Pembandingan juga mempertimbangkan jumlah parameter serta waktu komputasi, sehingga kesimpulan tidak hanya didasarkan pada satu skor ketepatan.

Penelitian ini menyajikan audit acuan SOC, pemisahan rekaman sebelum pembentukan jendela, evaluasi per rekaman pada tiga suhu lingkungan, dan pelaporan variasi antarseed. Kontribusinya bersifat empiris dan reproduktif pada konfigurasi yang dinyatakan. Penelitian tidak mengusulkan arsitektur baru, tidak menggunakan hasil SOH sebagai hasil SOC, dan tidak mengklaim generalisasi lintas sel atau implementasi pada sistem manajemen baterai.

## 2. Metode Penelitian

### 2.1 Dataset dan unit analisis

Data diunduh dari repositori resmi CALCE [8]. Subset yang digunakan adalah sel INR 18650-20R dengan penamaan berkas SP20-2. Dua belas arsip dinamis memuat profil DST, FUDS, US06, dan BJDST pada suhu lingkungan 0, 25, dan 45 °C. Setiap kombinasi profil–suhu mempunyai dua rekaman dengan muatan awal nominal 50% dan 80%. Satu arsip tambahan menyediakan pengujian kapasitas awal. Unit analisis adalah rekaman pengujian berulang pada **satu sel**, bukan 24 baterai independen. URL asal, nama anggota arsip, dan SHA-256 disimpan dalam manifest.

Bagian profil dinamis dipisahkan dari pengondisian pengisian/pengosongan awal. Urutan waktu dan penghitung kapasitas diperiksa. Sinyal dibentuk pada grid satu detik dengan mengambil observasi terakhir yang sudah tersedia. Cara ini tidak memakai interpolasi terpusat atau sampel masa depan. Sampel dengan umur observasi lebih dari lima detik tidak dipakai untuk membentuk jendela.

### 2.2 Rekonstruksi acuan dan audit domain

Kapasitas acuan diperoleh dari pengosongan penuh pada berkas pengujian kapasitas awal terpisah, yaitu Qref = {audit['Q_reference_Ah']:.5f} Ah. Nilai pencatat lengkap dipertahankan dalam kode untuk replikasi, tanpa menyatakan ketelitian alat hingga seluruh digit tersebut. Titik akhir pengisian CCCV dianggap sebagai keadaan penuh. Dengan Qc dan Qd sebagai penghitung kumulatif kapasitas pengisian dan pengosongan, acuan dinyatakan oleh:

SOCref(t) = 1 + [(Qc(t) − Qc,full) − (Qd(t) − Qd,full)] / Qref. (1)

Arus positif pada sumber berarti pengisian. Efisiensi coulombic diasumsikan satu. SOCref merupakan acuan hasil rekonstruksi pencatatan muatan, bukan pengukuran SOC independen. Kapasitas awal yang tetap juga tidak menyatakan kapasitas efektif identik pada semua suhu atau profil.

Audit menemukan {audit['outside_reference_range_samples']:,} sampel ({audit['outside_reference_range_percent']:.3f}%) di luar domain SOC 0–100%, terutama pada akhir pengosongan. Nilai mentah tetap disimpan. Jendela yang memuat label di luar domain dikeluarkan secara seragam dari semua model tanpa memotong label ke nol atau satu. Pengecualian ini membatasi kesimpulan pada domain acuan yang diterima; penelitian tidak mengklaim ketepatan seluruh lintasan sampai cutoff. Ketidakpastian kapasitas acuan belum dikuantifikasi sebagai interval SOC fisik. Analisis Movassagh et al. menegaskan bahwa ketidakpastian kapasitas, pengukuran arus, integrasi, dan waktu dapat memengaruhi hasil coulomb counting [10]; audit di sini bukan klaim penemuan baru mengenai sumber ketidakpastian tersebut.

**Tabel 1. Audit sampel acuan sebelum pembentukan jendela.**

{chr(10).join(audit_table)}

![Audit acuan SOC](results/full/figures/reference_audit.png)

**Gambar 1.** Rekonstruksi acuan pada DST dengan kondisi awal nominal 80%. Nilai negatif ditampilkan untuk memperlihatkan keterbatasan kapasitas acuan tetap; nilai tersebut tidak disembunyikan melalui clipping.

### 2.3 Masukan dan pembagian data

Vektor masukan terdiri atas tegangan terminal, arus, dan setelan suhu lingkungan ruang uji. Suhu tersebut tidak diperlakukan sebagai pembacaan suhu sel per detik. Penghitung kapasitas, indeks waktu, label SOC, dan data masa depan tidak digunakan sebagai fitur. Normalisasi berupa pengurangan rerata dan pembagian simpangan baku yang dihitung hanya dari sampel latih yang memenuhi domain.

Profil DST pada ketiga suhu digunakan untuk pelatihan. FUDS pada ketiga suhu digunakan untuk validasi dan early stopping. US06 serta BJDST pada ketiga suhu digunakan untuk pengujian. Rekaman dipisahkan sebelum pembentukan jendela sehingga jendela tumpang tindih tidak berpindah antara latih dan uji. Pemisahan ini menilai profil beban yang ditahan, sementara suhu uji telah diwakili pada pelatihan.

Setiap jendela memuat 60 sampel hingga waktu t dan menghasilkan SOC pada waktu t. Keadaan rekuren diinisialisasi ulang untuk setiap jendela. Langkah pembentukan jendela latih adalah lima sampel, sedangkan validasi dan uji satu sampel. Seluruh model dievaluasi pada titik target identik. Setelah pemeriksaan jendela, terdapat {run['train_windows']:,} jendela latih, {run['validation_windows']:,} jendela validasi, dan {run['test_windows']:,} jendela uji. Jendela yang bertumpang tindih tetap berkorelasi dan tidak dihitung sebagai replikasi sel independen.

### 2.4 Model dan pelatihan

MLP menerima 180 nilai hasil perataan jendela 60 × 3, diikuti satu lapisan tersembunyi 32 unit dengan aktivasi tanh. GRU dan LSTM menggunakan satu lapisan rekuren searah dengan 32 unit tersembunyi. Keluaran keadaan terakhir diteruskan ke lapisan linear dan sigmoid untuk membatasi estimasi pada 0–1. Arsitektur mengikuti implementasi standar PyTorch; tidak ada modifikasi gerbang yang diusulkan.

Semua konfigurasi menggunakan mean squared error, Adam dengan learning rate 0,001 [9], batch 256, dan pemotongan norma gradien maksimum satu. Pelatihan dibatasi 50 epoch dengan patience delapan berdasarkan MSE validasi gabungan. Bobot pada MSE validasi minimum dipakai untuk evaluasi. Seed 11, 22, dan 33 digunakan untuk memeriksa variasi inisialisasi dan pengacakan urutan batch. Pilot lima epoch pada 25 °C memakai latih/validasi saja; skor pilot tidak dicampurkan dengan pengujian utama.

Jumlah unit tersembunyi dan konteks disamakan, sedangkan jumlah parameter tidak identik. Oleh karena itu, pembandingan merupakan evaluasi konfigurasi yang dinyatakan dan bukan pembuktian arsitektur dengan kapasitas model persis sama. Pemilihan model atau parameter tidak dilakukan berdasarkan skor uji. Semua arsitektur tetap dilaporkan.

### 2.5 Metrik dan komputasi

MAE dan RMSE dihitung dalam poin persentase SOC (pp) untuk setiap rekaman. Kesalahan adalah 100 × (SOCestimasi − SOCref). Rerata makro memberi bobot sama pada 12 rekaman uji. Rerata dan simpangan baku kemudian dihitung atas tiga seed. Kesalahan absolut terbesar diperoleh dari seluruh sampel uji pada semua seed untuk masing-masing model. Simpangan baku antarseed tidak ditafsirkan sebagai interval kepercayaan populasi baterai.

Jumlah parameter, waktu pelatihan termasuk evaluasi validasi tiap epoch, dan throughput batch CPU dilaporkan. Pengukuran inferensi dilakukan setelah pemanasan dengan batch 256 dan dua thread CPU. Satuan ms/jendela merupakan waktu total inferensi batch dibagi jumlah jendela, bukan latensi pemrosesan satu sampel di ESP32. Versi runtime eksperimen adalah PyTorch {run['torch']} dan Python {run['python']}; konfigurasi serta versi dependensi disimpan bersama kode.

## 3. Hasil dan Pembahasan

### 3.1 Ketepatan pada profil beban yang ditahan

**Tabel 2. Ketepatan pengujian utama.** Rerata ± simpangan baku dari tiga seed; MAE/RMSE merupakan rerata makro per rekaman. Kesalahan maksimum diambil dari seluruh rekaman dan seed.

{chr(10).join(table)}

Pada konfigurasi ini, {best} memiliki rerata RMSE makro terendah. Selisih rerata RMSE GRU dan LSTM adalah {recurrent_difference:.3f} pp. Perbedaan tersebut bersifat deskriptif untuk profil US06/BJDST dan definisi acuan yang dipakai. Penelitian tidak melakukan uji signifikansi berbasis replikasi sel independen. Dengan demikian, selisih skor tidak dinyatakan sebagai keunggulan universal atau bukti ketepatan SOC fisik pada seluruh baterai.

![Perbandingan RMSE](results/full/figures/rmse_comparison.png)

**Gambar 2.** RMSE makro uji; error bar menunjukkan variasi antarseed dan bukan ketidakpastian populasi sel.

![Lintasan estimasi SOC](results/full/figures/soc_trace.png)

**Gambar 3.** Contoh lintasan US06, 25 °C, kondisi awal nominal 80%, seed 11. Rekaman dan seed untuk visualisasi ditetapkan dalam kode sebelum eksperimen utama selesai dan bukan dipilih berdasarkan skor terbaik.

### 3.2 Perbedaan menurut suhu lingkungan

**Tabel 3. RMSE makro per suhu lingkungan (pp).** Pada setiap suhu, empat rekaman US06/BJDST berbobot sama; rerata ± simpangan baku dihitung antarseed.

{chr(10).join(temp_table)}

![RMSE menurut suhu](results/full/figures/rmse_by_temperature.png)

**Gambar 4.** Perbedaan skor pada suhu yang telah diwakili dalam pelatihan.

Pemisahan menurut suhu membantu menunjukkan apakah satu skor gabungan menyembunyikan kondisi yang lebih sulit. Namun, ketiga suhu masuk data latih, sehingga hasil ini tidak membuktikan generalisasi ke suhu baru. Efek suhu pada sinyal listrik dan perbedaan acuan kapasitas juga tidak dipisahkan secara kausal dalam rancangan ini. Pada suhu rendah, penghentian pengosongan dapat terjadi ketika acuan berbasis kapasitas tetap masih positif; pada sebagian rekaman suhu lebih tinggi, rekonstruksi melewati nol. Perbedaan ini memperlihatkan bahwa definisi SOC, kapasitas yang digunakan, dan batas pengosongan harus dipertimbangkan bersamaan.

### 3.3 Kebutuhan komputasi

**Tabel 4. Pengukuran komputasi pada CPU pengujian.** Waktu pelatihan adalah rerata ± simpangan baku antarseed; throughput inferensi adalah median antarseed pada seluruh jendela uji.

{chr(10).join(compute)}

Angka waktu hanya berlaku pada perangkat dan implementasi pengujian. Jumlah parameter dapat digunakan untuk membandingkan ukuran model, tetapi belum membuktikan kebutuhan RAM, latensi, konsumsi daya, atau kompatibilitas ESP32. Implementasi embedded memerlukan pengukuran tambahan. Pelatihan termasuk validasi tiap epoch, sehingga waktu total juga dipengaruhi banyaknya epoch terpilih dan biaya evaluasi.

### 3.4 Sensitivitas terhadap definisi kapasitas acuan

Pemeriksaan eksploratif tambahan mengubah kapasitas acuan label menjadi 2,00 Ah, kapasitas kalibrasi, dan 2,06 Ah tanpa melatih ulang model. Ketiga nilai merupakan skenario, bukan toleransi kapasitas terukur atau batas ketidakpastian SOC benar. Agar pengaruh pergantian label tidak tercampur dengan pergantian sampel, seluruh skenario memakai {sensitivity_notes['distinct_common_targets']:,} titik target yang sama dan memenuhi domain acuan untuk ketiganya. Angka pada tabel ini tidak boleh dibandingkan sebagai peningkatan terhadap Tabel 2 tanpa memperhatikan perbedaan cakupan titik target.

**Tabel 5. RMSE makro pada skenario definisi acuan (pp).** Rerata ± simpangan baku antarseed. Ini sensitivitas skor terhadap label, bukan akurasi terverifikasi terhadap kapasitas alternatif.

{chr(10).join(sensitivity_table)}

Model dengan rerata RMSE terendah pada setiap skenario adalah {'; '.join(scenario_winners)}. Pemeriksaan ini menunjukkan bagaimana evaluasi numerik dan urutan model dapat bergantung pada asumsi kapasitas yang digunakan dalam label. Tidak ada kapasitas skenario yang dipilih untuk meningkatkan skor. Model, bobot, dan konfigurasi tetap sama. Analisis ini bersifat post hoc setelah audit data dan tidak dinyatakan sebagai eksperimen yang dipreregistrasikan.

### 3.5 Posisi terhadap literatur dan keterbatasan

LSTM untuk SOC telah diperkenalkan dalam studi terdahulu [2], dan penelitian terbaru mengembangkan model temporal atau gabungan dengan filter [3]–[5]. Skor penelitian ini tidak dibandingkan sebagai peringkat langsung terhadap angka terbitan tersebut karena sel, rekaman, acuan SOC, preprocessing, dan protokol dapat berbeda. Pembandingan yang dapat dipertanggungjawabkan di sini adalah antarimplementasi yang dilatih dan dievaluasi ulang dalam pipeline yang sama.

Keterbatasan paling mendasar adalah penggunaan acuan rekonstruksi. Kesalahan terhadap penghitung muatan bukan pengukuran kesalahan terhadap SOC independen. Kapasitas acuan tetap menghasilkan label di luar domain pada sebagian akhir rekaman; pengecualian seragam mencegah clipping menyembunyikan persoalan, tetapi tetap membatasi cakupan evaluasi. Penelitian lanjutan perlu menilai sensitivitas terhadap kapasitas acuan, memperoleh kalibrasi per kondisi yang dapat ditelusuri, dan memperluas pengujian pada sel lain.

Keterbatasan lain meliputi satu sel, tiga seed, tidak adanya eksperimen hardware, dan tidak adanya uji suhu yang sepenuhnya ditahan dalam hasil utama. Pembandingan dengan estimator fisik seperti EKF/UKF juga belum dilakukan, sehingga hasil tidak membuktikan keunggulan terhadap keluarga metode tersebut. Data NASA tidak dimasukkan karena audit acuan dan kesetaraan protokolnya belum diselesaikan. Temuan ini tidak boleh dipakai untuk mengklaim performa lintas NASA–CALCE, prediksi umur baterai, atau kesiapan BMS.

## 4. Kesimpulan

Penelitian membandingkan MLP, GRU, dan LSTM pada 24 rekaman dinamis satu sel CALCE melalui pemisahan profil sebelum pembuatan jendela. RMSE makro pengujian adalah {results_text}. {best} memperoleh rerata RMSE terendah pada konfigurasi yang dinyatakan. Audit juga mengidentifikasi {audit['outside_reference_range_samples']:,} sampel acuan di luar domain akibat keterbatasan penggunaan kapasitas tetap; label dipertahankan dan jendela terkait dikeluarkan tanpa clipping. Hasil mendukung pembandingan empiris terhadap acuan rekonstruksi, dengan batas pada satu sel dan suhu yang telah diwakili selama pelatihan. Validasi acuan per kondisi, pengujian sel independen, dan pengukuran embedded diperlukan sebelum menyatakan penerapan pada sistem manajemen baterai.

## Ketersediaan Data dan Kode

Data asli tersedia pada repositori CALCE [8]. Kode pengunduhan, audit, preprocessing, pelatihan, konfigurasi, hash sumber, dan prediksi disertakan dalam paket penelitian. URL repositori publik kode belum ditetapkan. Tidak ada klaim bahwa paket telah diarsipkan pada repositori publik atau memiliki DOI.

## Penggunaan AI dalam Penyusunan Draf

Codex dari OpenAI digunakan untuk membantu penelusuran literatur, penyusunan kode, audit komputasional, visualisasi, dan penulisan draf. Pelatihan model dijalankan pada data sumber yang diunduh; hasil bukan angka ilustratif. Penulis manusia perlu memeriksa, merevisi, dan menyetujui naskah serta mengambil tanggung jawab ilmiah sebelum pengiriman. Pernyataan ini tidak menyatakan persetujuan tersebut sudah diberikan.

## Daftar Pustaka

[1] F. Zheng, Y. Xing, J. Jiang, B. Sun, J. Kim, dan M. Pecht, “Influence of different open circuit voltage tests on state of charge online estimation for lithium-ion batteries,” Applied Energy, vol. 183, pp. 513–525, 2016. https://doi.org/10.1016/j.apenergy.2016.09.010.

[2] E. Chemali, P. J. Kollmeyer, M. Preindl, R. Ahmed, dan A. Emadi, “Long Short-Term Memory Networks for Accurate State-of-Charge Estimation of Li-ion Batteries,” IEEE Transactions on Industrial Electronics, vol. 65, no. 8, pp. 6730–6739, 2018. https://doi.org/10.1109/TIE.2017.2787586.

[3] H. Lin, X. Chen, L. Zhang, Z. Bao, dan S. Tang, “Estimation of battery SOC using a combined approach of temporal convolutional networks and Unscented Kalman Filter,” PLOS One, vol. 21, no. 6, e0350331, 2026. https://doi.org/10.1371/journal.pone.0350331.

[4] Y. Wu, J. Zhuang, L. Liu, F. Zeng, P. Li, dan X. Cheng, “Research on Estimation of State of Charge for Lithium-Ion Batteries Based on TRNN-CTA,” Applied Sciences, vol. 15, no. 17, 9653, 2025. https://doi.org/10.3390/app15179653.

[5] D. Batool, F. Majeed, S. Oh, J. Oh, dan J. Kim, “Hybrid Fuzzy Long Short-Term Memory Framework-Based Accurate State-Of-Charge Estimation for Future-Oriented Battery Management in Vehicle-To-Grid System,” International Journal of Energy Research, 5529603, 2026. https://doi.org/10.1155/er/5529603.

[6] K. Cho et al., “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,” EMNLP, 2014. https://doi.org/10.48550/arXiv.1406.1078.

[7] S. Hochreiter dan J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997. https://doi.org/10.1162/neco.1997.9.8.1735.

[8] CALCE, University of Maryland, “Battery Data,” diakses 22 Oktober 2026. https://calce.umd.edu/battery-data.

[9] D. P. Kingma dan J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015. https://doi.org/10.48550/arXiv.1412.6980.

[10] K. Movassagh, A. Raihan, B. Balasingam, dan K. Pattipati, “A Critical Look at Coulomb Counting Approach for State of Charge Estimation in Batteries,” Energies, vol. 14, no. 14, 4074, 2021. https://doi.org/10.3390/en14144074.
'''
    def url_link(match):
        raw=match.group(0); url=raw.rstrip('.')
        return f'[Sumber]({url})'+raw[len(url):]
    manuscript=re.sub(r'(?<!\()https://\S+',url_link,manuscript)
    (ROOT/'DRAF_ARTIKEL_MESTRO_SOC.md').write_text(manuscript,encoding='utf-8')
    stats.to_csv(folder/'manuscript_summary.csv')
    tstats.to_csv(folder/'manuscript_temperature_summary.csv')
    print('Draft generated from complete experiment; best mean macro RMSE:',best)

if __name__=='__main__':
    main()
