# Estimasi SOC baterai CALCE dengan MLP, GRU, dan LSTM

Proyek ini membandingkan MLP, GRU, dan LSTM untuk **estimasi SOC saat ini** dari riwayat tegangan, arus, dan suhu lingkungan pengujian. Ini bukan peramalan SOC di masa depan, estimasi SOH, atau RUL.

## Ruang lingkup

- Data utama: satu sel CALCE INR 18650-20R, label berkas SP20-2.
- Empat profil: DST, FUDS, US06, BJDST; tiga suhu lingkungan: 0, 25, 45 derajat Celsius; dua kondisi muatan awal nominal: 50% dan 80%.
- Total 24 rekaman dinamis, ditambah satu rekaman kalibrasi kapasitas awal dalam 13 arsip asli.
- NASA belum menjadi bagian hasil. Perlu audit terpisah untuk subset, acuan SOC, dan kesetaraan fitur sebelum dimasukkan.

## Perhatian tentang acuan SOC

Berkas yang diperiksa tidak menyediakan kolom SOC. Acuan direkonstruksi dari pencatatan kapasitas penguji:

`SOC_ref(t) = 1 + [(Qc(t)-Qc_full) - (Qd(t)-Qd_full)] / Qref`

`Qref = 2.028344937765965 Ah`, diambil dari pengujian pengosongan penuh terpisah sebelum rekaman dinamis. Positif arus berarti pengisian. Titik akhir pengisian CCCV dianggap SOC 100%; efisiensi coulombic diasumsikan satu.

Ini **acuan berbasis coulomb counting**, bukan pengukuran SOC independen. Kapasitas acuan tetap menghasilkan beberapa nilai negatif pada bagian akhir pengosongan, khususnya pada suhu 45 C. Label mentah tetap disimpan, tanpa clipping. Jendela yang memuat label di luar 0–100% dikeluarkan dari pelatihan dan evaluasi semua model dengan aturan yang sama. Jumlahnya dilaporkan pada `results/pipeline_validation.json` dan `results/data_inventory.csv`. Tidak ada klaim ketepatan sepanjang seluruh pengosongan sampai cutoff, atau SOC 0–100% penuh. Efek definisi kapasitas terhadap validitas SOC perlu dibahas dalam naskah.

Suhu masukan adalah **setelan suhu ruang uji**, bukan pembacaan suhu sel pada setiap detik. Kapasitas kumulatif, indeks waktu, label SOC, dan informasi masa depan tidak dijadikan masukan model.

## Pembagian eksperimen

| Bagian | Profil | Suhu untuk eksperimen utama |
|---|---|---|
| Latih | DST | 0, 25, 45 C |
| Validasi dan early stopping | FUDS | 0, 25, 45 C |
| Uji | US06 dan BJDST | 0, 25, 45 C |

Pemisahan berdasarkan rekaman dilakukan sebelum pembuatan jendela. Normalisasi dipelajari hanya dari data latih. Pengambilan sampel 1 detik memakai observasi terakhir yang sudah tersedia. Jendela berisi 60 sampel dan menghasilkan SOC pada sampel terakhir; 59 sampel awal tidak dievaluasi.

Pilot memakai hanya DST/FUDS pada 25 C, satu seed dan lima epoch. **Metrik pilot adalah validasi, bukan hasil uji artikel.** Eksperimen utama memakai tiga seed, maksimal 50 epoch, dan early stopping berdasarkan FUDS. Semua model memakai konteks waktu, lapisan tersembunyi 32 unit, optimizer, dan aturan penghentian sama; jumlah parameter berbeda dan dilaporkan. Ini pembandingan konfigurasi tertentu, bukan bukti keunggulan universal model.

**Eksperimen tambahan yang tidak disertakan hasilnya dalam rilis ini:** `cross-temperature` merupakan pelatihan terpisah: DST 0/45 C untuk latih, FUDS 0/45 C untuk validasi, US06/BJDST 25 C untuk uji suhu yang ditahan. Hasil eksperimen utama sendiri **bukan bukti generalisasi ke suhu yang belum terlihat**, karena ketiga suhu masuk data latihnya.

## Menjalankan

Notebook `SOC_CALCE_Colab.ipynb` memuat kode sumber dan prosedur lengkap. Notebook dapat diunggah ke Google Colab; tidak memerlukan API key atau hardware eksperimen.

Gunakan Python 3.12 dan buat lingkungan terpisah. Instal `python -m pip install -r requirements.txt`, kemudian jalankan dari direktori akar repositori:

```powershell
python scripts/fetch_data.py --temperatures 0 25 45
python scripts/prepare_data.py
python scripts/validate_pipeline.py
python scripts/train.py --mode pilot
python scripts/plot_results.py --mode pilot
python scripts/train.py --mode full
python scripts/plot_results.py --mode full
python scripts/reference_sensitivity.py
python scripts/build_manuscript.py
```

Unduhan awal sekitar 46 MB; durasi pelatihan bergantung komputer.

## Membaca hasil

- `data/raw/download_manifest.json`: URL asli dan SHA-256 arsip.
- `results/data_inventory.csv`: jumlah sampel, rentang acuan, dan asal rekaman.
- `results/pipeline_validation.json`: audit integritas data dan pemisahan.
- `results/current_balance_check.csv`: selisih integral arus setelah sampling 1 detik terhadap perubahan penghitung muatan; pemeriksaan konsistensi, bukan acuan SOC independen.
- `results/<mode>/run_info.json`: daftar rekaman, normalisasi, versi runtime.
- `results/<mode>/metrics_per_record.csv`: metrik setiap rekaman dan seed.
- `results/<mode>/summary.csv`: rerata dan simpangan baku antarseed dari macro-MAE/RMSE.
- `results/<mode>/predictions.csv`: estimasi aktual untuk audit.
- `results/<mode>/figures/`: grafik PNG 300 dpi dan PDF.

MAE/RMSE dinyatakan dalam **poin persentase SOC**. Rerata makro memberi bobot sama pada setiap rekaman, sehingga rekaman panjang tidak otomatis mendominasi. Sampel dari satu rekaman berkorelasi; tiga seed bukan tiga eksperimen baterai independen. Waktu inferensi merupakan throughput batch CPU, bukan latensi ESP32 atau BMS.

Analisis `reference_sensitivity.py` adalah pemeriksaan eksploratif setelah audit data. Kapasitas 2,00 Ah, kapasitas kalibrasi, dan 2,06 Ah dipakai sebagai skenario definisi label, bukan rentang toleransi terukur atau SOC benar alternatif. Model tidak dilatih ulang dan semua skenario dinilai pada waktu target yang sama. Jangan menggunakan pemeriksaan ini untuk memilih kapasitas yang membuat skor model terlihat paling baik.

## Sumber dan posisi kontribusi

1. [CALCE Battery Data](https://calce.umd.edu/battery-data), sumber eksperimen dan arsip asli.
2. F. Zheng et al., “Influence of different open circuit voltage tests on state of charge online estimation for lithium-ion batteries,” Applied Energy, 183, 513–525, 2016. [DOI](https://doi.org/10.1016/j.apenergy.2016.09.010).
3. E. Chemali et al., “Long Short-Term Memory Networks for Accurate State-of-Charge Estimation of Li-ion Batteries,” IEEE Transactions on Industrial Electronics, 65(8), 6730–6739, 2018. [DOI](https://doi.org/10.1109/TIE.2017.2787586).
4. H. Lin et al., “Estimation of battery SOC using a combined approach of temporal convolutional networks and Unscented Kalman Filter,” PLOS One, 21(6), e0350331, 2026. [DOI](https://doi.org/10.1371/journal.pone.0350331).

GRU, LSTM, dan uji lintas profil sudah digunakan dalam literatur. Kontribusi yang dituju adalah audit acuan SOC, pembandingan yang dapat direproduksi pada profil terpisah, serta pelaporan kompromi akurasi dan komputasi. Kebaruan belum dibuktikan oleh tersedianya kode atau skor model.


## Hasil tersimpan dan versi rilis

Rilis **v1.0.0** memuat kode asli, protokol, manifest URL dan SHA-256 sumber,
sembilan checkpoint model, metrik, riwayat pelatihan, dan gambar hasil eksperimen
utama. Tidak ada pelatihan ulang saat menyiapkan rilis ini.

| Model | MAE makro uji (pp) | RMSE makro uji (pp) |
|---|---|---|
| MLP | 2.053 ± 0.217 | 2.739 ± 0.292 |
| GRU | 0.942 ± 0.082 | 1.164 ± 0.074 |
| LSTM | 0.929 ± 0.042 | 1.202 ± 0.064 |

Angka adalah rerata ± simpangan baku atas tiga seed. Setiap seed memberi bobot
sama pada 12 rekaman uji. Acuan berbasis penghitung muatan pada satu sel;
hasil bukan validasi SOC independen, lintas sel, atau hardware BMS.

Arsip `predictions-full-v1.0.0.csv.gz` pada halaman Releases memuat seluruh
1.396.827 baris prediksi tersimpan. Untuk mengaudit ulang metrik tersimpan,
ekstrak ke `results/full/predictions.csv`, lalu jalankan
`python scripts/audit_results.py`. Laporan audit juga disertakan dalam repositori.
Daftar SHA-256 aset tersedia dalam `SHA256SUMS.txt` pada Releases.

Data mentah CALCE tidak disalin ke repositori ini. Unduh langsung dari CALCE
melalui script pengunduhan; atribusi dan ketentuan sumber data tetap berlaku.
Repositori ini tidak mengklaim kepemilikan atau lisensi ulang data CALCE.
Kode dan hasil rilis ini merupakan arsip reproduksibilitas, bukan artikel yang
sudah diterima atau diterbitkan oleh jurnal. Rilis GitHub ini belum memiliki DOI.

Codex dari OpenAI membantu penyusunan kode, audit, visualisasi, dokumentasi,
dan draf. Penulis telah mengonfirmasi pemeriksaan, persetujuan, dan tanggung
jawab atas naskah. Sumber pendanaan: biaya mandiri; seluruh penulis menyatakan
tidak ada konflik kepentingan terkait penelitian ini.
