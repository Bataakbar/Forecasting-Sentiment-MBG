# Sentimen MBG — Makan Bergizi Gratis Forecasting Dashboard 🥗

Aplikasi dashboard interaktif berbasis web untuk memantau dan memprediksi (forecasting) sentimen masyarakat terhadap program **Makan Bergizi Gratis (MBG)**. Dashboard ini dibuat menggunakan **Flask** sebagai backend dan antarmuka web interaktif yang premium, serta memanfaatkan model deep learning **CNN-LSTM** untuk memproyeksikan tren sentimen positif dan negatif.

---

## 🌟 Fitur Utama

- 📊 **Tren & Proyeksi Sentimen Harian**: Visualisasi data aktual sentimen masyarakat (107 hari historis) dan proyeksi peramalan autoregresif 107 hari ke depan dengan pendekatan *blended seasonality* (menggabungkan model deep learning dengan data tren musiman hari-an).
- 📈 **KPI & Ringkasan Metrik**: Menampilkan total sebutan (*mention*), persentase sentimen harian terbaru (Positif, Negatif, Netral), serta metrik evaluasi model (Akurasi dan F1-Score).
- 🍩 **Distribusi Sentimen Terbaru**: Grafik lingkaran (*doughnut chart*) untuk memantau proporsi sentimen masyarakat saat ini.
- 🏷️ **Topik & Kata Kunci Dominan**: Ekstraksi kata kunci yang paling banyak dibahas masyarakat mengenai program MBG secara dinamis (seperti kualitas gizi, anggaran, sebaran wilayah, dll).
- 💬 **Feed Komentar/Postingan Terbaru**: Umpan balik real-time dari data media sosial (Twitter/X, Instagram, TikTok, Berita) yang diurutkan dari yang terbaru lengkap dengan kategori sentimennya.
- 🧪 **Panel Simulasi Prediksi Kustom**: Simulator interaktif 14 hari terakhir di mana pengguna dapat menginput persentase sentimen kustom untuk memprediksi persentase sentimen hari ke-15 secara instan menggunakan model CNN-LSTM.

---

## 🛠️ Tech Stack & Model Deep Learning

1. **Backend**: Flask (Python)
2. **Frontend**: HTML5, Vanilla CSS, Vanilla JS, Chart.js, Tabler Icons CDN
3. **Model Deep Learning**:
   - Model Sentimen Positif (`model_cnn_lstm_positif.h5`) berbasis arsitektur **CNN-LSTM** (Input Shape: `(None, 14, 1)` -> Output Shape: `(None, 1)`).
   - Model Sentimen Negatif (`model_cnn_lstm_negatif.h5`) berbasis arsitektur **CNN-LSTM** (Input Shape: `(None, 14, 1)` -> Output Shape: `(None, 1)`).
4. **Dataset**: `labeled_data.csv` (berisi postingan media sosial ter-labeli dengan atribut tanggal, teks, sentimen, dan platform).

---

## 📂 Struktur Repositori

```text
├── static/
│   ├── css/
│   │   └── styles.css          # Desain premium bertema botani
│   └── js/
│       └── app.js              # Logika frontend & integrasi API Flask
├── templates/
│   └── index.html              # Layout antarmuka dashboard
├── app.py                      # Server Flask utama & logika pemodelan / forecasting
├── labeled_data.csv            # Dataset utama
├── model_cnn_lstm_negatif.h5   # Model CNN-LSTM sentimen negatif
├── model_cnn_lstm_positif.h5   # Model CNN-LSTM sentimen positif
├── .gitignore                  # File pengecualian Git
└── README.md                   # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Project Secara Lokal

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal Python (disarankan versi 3.9 s.d. 3.11).

### 2. Instalasi Dependensi
Instal pustaka Python yang diperlukan menggunakan `pip`:
```bash
pip install flask pandas numpy tensorflow
```

### 3. Jalankan Aplikasi
Jalankan file server utama `app.py`:
```bash
python app.py
```

### 4. Buka di Browser
Setelah server berjalan, buka browser Anda dan akses alamat berikut:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📊 Detail Model Forecasting
Model memproyeksikan masa depan dengan cara **Autoregresif Multi-step**:
1. Mengambil data sentimen 14 hari sebelumnya.
2. Memprediksi hari berikutnya menggunakan model **CNN-LSTM**.
3. Hasil prediksi hari ke-15 digabungkan dengan tren rata-rata musiman hari-an (*blended seasonality*) untuk mencegah terjadinya penurunan performa (*decay*) ke garis datar jika diramal dalam jangka panjang (107 hari).
4. Hasil tersebut kemudian ditambahkan kembali ke urutan input untuk memprediksi hari berikutnya lagi, hingga terkumpul ramalan sepanjang 107 hari.
