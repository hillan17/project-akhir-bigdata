# 📈 Real-Time Big Data & Machine Learning: Prediksi & Business Intelligence Pasar Kripto-Saham

Sistem analitik **Big Data & Machine Learning** *end-to-end* untuk menganalisis korelasi pasar aset kripto yang beroperasi 24/7 (**Bitcoin, Ethereum, Solana**) terhadap indikator makroekonomi dan pasar saham global (**S&P 500, Indeks Dolar DXY, Emas, IHSG**) menggunakan data historis 1 tahun (November 2025 – November 2026) serta pemantauan prediksi *real-time*.

**Disusun Oleh:**
* **Nama:** Hillan Tazidi Tsalits
* **Institusi:** Program Studi Teknologi Informasi, Fakultas Ilmu Komputer (FILKOM), Universitas Brawijaya

---

## 🏗️ Arsitektur Sistem

```text
[Binance Public Data & Yahoo Finance API]
                  │
                  ▼ (Python Ingestion)
[Hadoop HDFS 3.5.0 Data Lake (/datalake/raw & /datalake/processed)]
                  │
                  ▼ (Apache Spark / PySpark: Cleaning, Forward-Fill, Parquet, ML & BI)
[PostgreSQL Data Warehouse (market_dw: fact_market_analytics & dim_bi_factors)]
                  │
                  ▼ (Streamlit + Plotly.js Zero-Reload Stream)
[Real-Time Web Dashboard (Auto-Update 5 Detik & Sliding Window)]
```

### Fitur Utama
1. **HDFS Data Lake & Resiliensi Multi-Aset:** Menyimpan data mentah interval 1 jam (9.336 baris/aset) ke zona `/datalake/raw/` dan hasil integrasi terbersihkan dalam format kolumnar **Apache Parquet** di `/datalake/processed/`.
2. **Penanganan Data Kosong Akhir Pekan:** Menggunakan *Window Function* **Forward-Fill** dan **Backward-Fill** pada Apache Spark untuk mengisi kekosongan data indeks saham/makro saat bursa tradisional tutup di akhir pekan.
3. **Business Intelligence (Faktor Masa Lampau):** Menggunakan **Random Forest Regressor** untuk mengukur persentase pengaruh variabel historis ($H-1$ hingga $H-3$) terhadap pergerakan return harga hari ini.
4. **Predictive Analytics & Deteksi Anomali:** Model **Gradient Boosting Regressor** mencapai tingkat akurasi **97,89% (MAPE 2,11%)**, dilengkapi deteksi anomali volume transaksi berbasis **Z-Score > 2,5**.
5. **Dashboard Real-Time Zero-Reload:** Menggunakan **Streamlit** dan **Plotly.js (`Plotly.extendTraces`)** yang memperbarui grafik ticker dan sinyal keputusan ML (*Bullish / Sideways / Bearish*) setiap **5 detik** tanpa me-reload halaman.

---

## 📁 Struktur Direktori Proyek

```text
project-akhir-bigdata/
├── dashboard/
│   └── app.py                  # Aplikasi Web Dashboard (Streamlit + Plotly.js Live 5s)
├── ingestion/
│   ├── ingest_to_hdfs.py       # Akuisisi data historis 1 tahun ke Hadoop HDFS
│   └── stream_producer.py      # Producer streaming ticker real-time
├── spark_jobs/
│   └── spark_hdfs_ml_bi.py     # ETL PySpark, Parquet, Feature Engineering, ML & Load ke DW
├── docker-compose.yml          # Konfigurasi kontainer PostgreSQL & Apache Kafka
├── requirements.txt            # Daftar dependensi library Python
└── README.md                   # Dokumentasi proyek
```

---

## ⚙️ Prasyarat Sistem

* **OS:** Windows 10/11 (atau Linux)
* **Python:** Versi 3.10 – 3.12
* **Apache Hadoop:** Versi 3.5.0 (HDFS aktif pada `hdfs://localhost:9000`)
* **Apache Spark:** Versi 4.2.0 (atau 3.5+)
* **Docker Desktop:** Aktif untuk menjalankan PostgreSQL Data Warehouse & Kafka

---

## 🚀 Langkah-Langkah Menjalankan Aplikasi

### 1. Persiapan Lingkungan & Instalasi Dependensi
Buka terminal PowerShell di folder proyek dan buat *virtual environment*:

```powershell
cd D:\bigdata-crypto-macro-analytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install "psycopg[binary]"
```

### 2. Menyalakan Infrastruktur (Docker & Hadoop HDFS)
Nyalakan kontainer **PostgreSQL** dan **Kafka**, lalu jalankan klaster **Hadoop HDFS**:

```powershell
docker-compose up -d
start-dfs.cmd
```
*(Pastikan NameNode dan DataNode berjalan normal melalui `jps` atau Web UI HDFS di `http://localhost:9870`).*

### 3. Akuisisi Data Mentah ke Hadoop HDFS (Data Lake)
Jalankan skrip *ingestion* untuk mengunduh data historis 1 tahun dan menyimpannya ke HDFS (`/datalake/raw/`):

```powershell
python ingestion/ingest_to_hdfs.py
```

### 4. Pemrosesan Big Data, Pelatihan ML & Pemuatan ke Data Warehouse
Jalankan *job* Apache Spark untuk melakukan pembersihan data, konversi Parquet, pelatihan model ML/BI, dan pemuatan hasil ke PostgreSQL (`market_dw`):

```powershell
spark-submit spark_jobs/spark_hdfs_ml_bi.py
```
*(Tunggu hingga muncul pesan: `SUKSES! Seluruh hasil analitik dari HDFS berhasil dimuat ke Data Warehouse PostgreSQL!`).*

### 5. Menjalankan Dashboard Web Real-Time
Jalankan aplikasi dashboard Streamlit:

```powershell
streamlit run dashboard/app.py
```
Buka browser dan akses **`http://localhost:8501`**.

---

## 🔄 Menyalakan Ulang & Mematikan Lingkungan

* **Mematikan Aplikasi:**
  1. Tekan **`Ctrl + C`** di terminal Streamlit untuk menghentikan web dashboard.
  2. Jalankan `docker-compose stop` (atau `docker-compose down` tanpa `-v` agar data PostgreSQL tetap tersimpan).
  3. Jalankan `stop-dfs.cmd` dan `deactivate` untuk keluar dari `.venv`.
* **Menyalakan Ulang (Jika Data Sudah Ada di PostgreSQL):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  docker-compose start
  streamlit run dashboard/app.py
  ```