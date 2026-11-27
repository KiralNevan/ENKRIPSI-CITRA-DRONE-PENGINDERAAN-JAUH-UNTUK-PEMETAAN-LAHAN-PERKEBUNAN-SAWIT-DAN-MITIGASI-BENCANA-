# Enkripsi Citra AES dengan S-Box Hasil Algoritma Genetika

Proyek akhir mata kuliah Kriptografi: membangun S-box 16×16 dengan algoritma
genetika (pendekatan machine learning), menggantikan S-box AES-128, lalu
menguji enkripsi/dekripsi citra Lena, Mandrill, dan citra penginderaan jauh.

## Fitur

- S-box 16×16 bijective dan balanced, dibangun dengan GA (permutasi 0–255).
- Evaluasi enam kriteria: **NL, SAC, BIC-NL, BIC-SAC, LP, DAP**.
- AES-128 dari nol dengan S-box yang dapat diganti (SubBytes, SubWord,
  InvSubBytes), mode **ECB** dan **CBC**, padding PKCS#7.
- Validasi AES dengan FIPS-197 dan PyCryptodome.
- Metrik citra: **entropy, histogram, NPCR, UACI, korelasi, waktu, MSE,
  SHA-256**.
- Aplikasi **Streamlit** lima tab: S-box, metrik, enkripsi/dekripsi,
  hasil eksperimen, dan penjelasan.
- Hasil eksperimen tersimpan sebagai CSV/JSON dan figur PNG.

## Struktur

```text
.
├─ app/streamlit_app.py
├─ src/
│  ├─ aes/          # core AES, mode, konstanta
│  ├─ sbox/         # metrik, GA, IO
│  ├─ image/        # load citra, metrik citra
│  ├─ experiments/  # runner GA, benchmark, remote sensing, figur
│  └─ utils/
├─ data/raw/        # Lena, Mandrill, citra Landsat
├─ data/processed/  # citra 512×512 PNG
├─ results/
│  ├─ sbox/         # best_sbox.csv, JSON, history
│  ├─ metrics/      # CSV hasil eksperimen
│  └─ figures/      # figur laporan
├─ tests/
├─ requirements.txt
└─ README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.14 + NumPy + Numba sudah diuji. Numba dipakai untuk mempercepat
enkripsi blok AES.

## Menjalankan eksperimen

```bash
# 1. Bangun S-box dengan GA (5 seed)
python -m src.experiments.run_ga --seeds 2026,2027,2028,2029,2030 --population 80 --generations 300

# 2. Benchmark Lena & Mandrill
python -m src.experiments.run_benchmark

# 3. Citra penginderaan jauh
python -m src.experiments.run_remote --size 512

# 4. Figur laporan
python -m src.experiments.make_figures
```

## Menjalankan Streamlit

```bash
streamlit run app/streamlit_app.py
```

Buka `http://localhost:8501`.

### Deploy

1. Push repo ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io), pilih repo,
   branch, dan `app/streamlit_app.py`.
3. Salin URL publik ke Google Form sesuai instruksi dosen.

## Hasil (ringkas)

Berdasarkan `results/metrics/`:

| Metrik | AES standar | GA terbaik | Target |
|---|---:|---:|---:|
| Nonlinearity | 112 | 100 | ≥ 112 |
| SAC | 0,5049 | 0,5000 | ≈ 0,5 |
| BIC-NL | 112 | 100 | 112 |
| BIC-SAC | 0,5046 | 0,4994 | ≈ 0,5 |
| LP | 0,015625 | 0,047852 | mendekati 0 |
| DAP | 0,015625 | 0,039063 | mendekati 0 |

S-box GA **belum mencapai** target NL/BIC-NL/LP/DAP setara AES pada
anggaran 300 generasi dan 5 seed. Ini dilaporkan apa adanya dan dibahas
sebagai keterbatasan representasi permutasi + fitness multi-kriteria.

Meskipun demikian, kualitas ciphertext pada mode CBC sangat baik:

| Citra | Algoritma | Entropy | NPCR | UACI | Korelasi H/V/D |
|---|---|---:|---:|---:|---|
| Lena | AES CBC | 7,9998 | 99,60% | 33,45% | ≈ 0 |
| Lena | GA CBC | 7,9998 | 99,61% | 33,46% | ≈ 0 |
| Mandrill | AES CBC | 7,9998 | 99,61% | 33,47% | ≈ 0 |
| Mandrill | GA CBC | 7,9997 | 99,59% | 33,49% | ≈ 0 |
| Landsat | GA CBC | 7,9998 | 99,60% | 33,47% | ≈ 0 |

Semua konfigurasi **lossless** (MSE 0, hash SHA-256 identik). ECB
menunjukkan NPCR sensitivitas plaintext ≈ 0 karena tidak ada difusi
antarblok; sensitivitas kunci tetap ≈ 99,6%.

## Dataset

- **Lena** dan **Mandrill**: OpenCV sample images
  (`samples/data/lena.jpg`, `samples/data/baboon.jpg`).
- **Citra penginderaan jauh**: Landsat 7 ETM+ Kalimantan, 16 Juli 2000,
  NASA Earth Observatory (public domain), di-crop/resize ke 512×512.
  Sumber: <https://science.nasa.gov/earth/earth-observatory/kalimantan-borneo-indonesia-78461/>

## Catatan

- Kunci/IV di aplikasi adalah demo akademik, bukan untuk produksi.
- ECB tidak memiliki difusi antarblok; NPCR sensitivitas plaintext akan
  sangat kecil. Ini justru menunjukkan kelemahan ECB dibanding CBC.
- Jika target S-box (NL 112) tidak tercapai, nilai sebenarnya dilaporkan
  dan dibahas, bukan dimanipulasi.
- Hasil metrik bergantung pada implementasi dan perangkat; perbandingan
  waktu hanya berlaku pada implementasi yang sama.
