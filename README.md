<a href="https://orcid.org/0009-0002-1084-6002" aria-label="View ORCID record - 0009-0002-1084-6002"> <img src="/ORCID-iD_icon-vector.svg" alt="ORCID iD"/>https://orcid.org/0009-0002-1084-6002</a>

# 🤳 Face ID Attendance App

Aplikasi presensi berbasis pengenalan wajah menggunakan **React Native**, **Expo**, dan model **MobileFaceNet** (TFLite). Aplikasi ini memungkinkan pendaftaran wajah "Master" dan melakukan verifikasi real-time untuk mencocokkan wajah yang tertangkap kamera dengan data yang sudah tersimpan.

---

## 🚀 Alur Kerja Aplikasi (App Flow)

Proses utama aplikasi dibagi menjadi dua fase: **Registrasi** dan **Verifikasi**.

### 1. Inisialisasi
Saat aplikasi dibuka pertama kali:
- Meminta izin akses **Kamera**.
- Memuat (loading) model **MobileFaceNet** (`.tflite`) ke dalam memory menggunakan `react-native-fast-tflite`.
- Menyiapkan penyimpanan lokal untuk file embedding wajah.

### 2. Proses Registrasi (Pendaftaran Wajah)
Tujuannya adalah mengambil "sidik jari wajah" unik dan menyimpannya sebagai referensi.
- **Capture**: Mengambil foto wajah dari kamera depan.
- **Detection**: Mencari koordinat wajah menggunakan `expo-face-detector`.
- **Extraction**:
  - Memotong (crop) gambar tepat di area wajah.
  - Mengubah ukuran (resize) menjadi **112x112 pixel**.
- **Inference**: Menjalankan model AI untuk mendapatkan **Face Embedding** (vektor numerik 128-D).
- **Storage**: Menyimpan vektor tersebut ke dalam file `face_master_data.json` di penyimpanan internal perangkat.

### 3. Proses Verifikasi (Absensi/Login)
Tujuannya adalah membandingkan wajah saat ini dengan data "Master".
- **Capture & Process**: Melakukan langkah yang sama dengan registrasi (Capture -> Detect -> Resize -> Inference).
- **Comparison**: Membandingkan vektor wajah baru dengan vektor master menggunakan algoritma **Cosine Similarity**.
- **Decision**: 
  - Jika skor similarity **> 0.75**, verifikasi dinyatakan **Berhasil** (Wajah Cocok).
  - Jika skor di bawah threshold, akses **Ditolak**.

---

## 🧠 Detail Teknis & Penggunaan Model

### 🎨 Model AI: MobileFaceNet
Aplikasi ini menggunakan **MobileFaceNet**, model *deep learning* yang sangat efisien untuk perangkat seluler.
- **Input**: Tensor [1, 112, 112, 3] (Gambar RGB 112x112).
- **Output**: 128-Dimensional Vector (Embedding) yang merepresentasikan fitur unik wajah.

### 🧪 Pre-processing
Agar model bekerja akurat, data gambar mentah harus diolah:
- **Normalisasi Pixel**: Setiap nilai RGB (0-255) diubah ke rentang **-1.0 hingga 1.0** menggunakan rumus:
  ```
  pixel_normalized = (pixel_value - 127.5) / 128.0
  ```

### 🔢 Cosine Similarity
Metode matematika untuk mengukur kemiripan antara dua vektor.
- Skor **1.0** berarti identik secara matematis.
- Skor **0.75+** digunakan sebagai ambang batas (threshold) untuk mengatasi variasi pencahayaan atau sudut wajah ringan.

---

## 🛠 Teknologi yang Digunakan

| Library | Kegunaan |
| :--- | :--- |
| **Expo Camera** | Mengakses kamera perangkat |
| **Expo Face Detector** | Mendeteksi lokasi wajah secara real-time |
| **Fast TFLite** | Menjalankan model AI (.tflite) dengan performa native |
| **Image Manipulator** | Melakukan Crop dan Resize gambar secara cepat |
| **SecureStore** | Menyimpan status pendaftaran secara aman |
| **File System** | Menyimpan file embedding wajah (.json) |

---

## 📋 Cara Menjalankan Project

1. Install dependensi:
   ```bash
   npm install
   ```
2. Jalankan di Android/iOS (membutuhkan device fisik untuk kamera):
   ```bash
   npx expo start
   ```

---

> **Note**: Untuk akurasi terbaik, pastikan wajah menghadap lurus ke kamera dan berada di area dengan pencahayaan yang cukup saat melakukan registrasi maupun verifikasi.
