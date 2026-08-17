# Optimasi Klasifikasi Penyakit Daun Jagung Menggunakan EfficientNet-B0 dengan Preprocessing CLAHE dan Explainable AI

Repositori ini memuat kode sumber eksperimen berbasis Python (Jupyter Notebook/Google Colab) untuk penelitian klasifikasi penyakit daun jagung. Sistem ini dikembangkan dengan memadukan teknik peningkatan kualitas citra, arsitektur *deep learning* yang ringan, dan transparansi keputusan model.

## Deskripsi Eksperimen
Penelitian ini bertujuan untuk mengidentifikasi penyakit daun jagung secara akurat tanpa mengorbankan interpretabilitas model. Eksperimen ini mencakup tiga komponen utama:
1. **Preprocessing (CLAHE):** Penyeragaman pencahayaan citra secara lokal menggunakan *Contrast Limited Adaptive Histogram Equalization* pada kanal L (ruang warna LAB) untuk mempertegas area lesi penyakit tanpa merusak informasi warna asli daun.
2. **Klasifikasi (EfficientNet-B0):** Penggunaan arsitektur *transfer learning* ringan untuk mengklasifikasikan citra ke dalam 4 kelas penyakit. Model dilatih dengan pendekatan dua tahap (*warm-up* dan *fine-tuning*).
3. **Interpretasi (LIME):** Implementasi *Local Interpretable Model-Agnostic Explanations* untuk memvisualisasikan area *superpixels* pada daun yang paling berkontribusi terhadap keputusan prediksi, memastikan model tidak mengambil jalan pintas dengan mempelajari latar belakang (noise).

## Dataset
Data yang digunakan bersumber dari Kaggle: **[Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)** oleh S. Ghose (2022). 
Dataset terdiri atas 3.852 citra yang terbagi ke dalam empat kelas:
*   `Blight` (Hawar Daun)
*   `Common_Rust` (Karat Daun)
*   `Gray_Leaf_Spot` (Bercak Daun)
*   `Healthy` (Sehat)

## Struktur Repositori
*   `kode_eksperimen.ipynb` : File Jupyter Notebook utama yang diekspor dari Google Colab. Berisi seluruh alur komputasi mulai dari pemuatan data, penerapan fungsi CLAHE, pelatihan model EfficientNet-B0, evaluasi metrik (*Confusion Matrix* & *Classification Report*), hingga visualisasi batas LIME.
*   `README.md` : Dokumentasi repositori.

## Persyaratan (Dependencies)
Eksperimen ini dijalankan di lingkungan Google Colab dengan pustaka utama berikut:
*   `tensorflow` / `keras` (untuk arsitektur EfficientNet-B0)
*   `opencv-python` (cv2, untuk operasi konversi warna LAB dan penerapan fungsi CLAHE)
*   `lime` (untuk Explainable AI)
*   `scikit-learn` (untuk kalkulasi metrik evaluasi)
*   `matplotlib` & `seaborn` (untuk visualisasi data dan plot)

## Cara Penggunaan
1. Unduh file `kode_eksperimen.ipynb` dari repositori ini.
2. Buka dan unggah file tersebut ke platform **Google Colab**.
3. Pastikan dataset telah diunduh dari Kaggle dan diunggah ke Google Drive Anda atau langsung ke *storage* sesi Colab.
4. Sesuaikan variabel direktori (path) dataset di dalam kode agar mengarah ke lokasi penyimpanan data Anda.
5. Jalankan (*Run All*) seluruh sel (cells) secara berurutan.

## Kinerja Model
Berdasarkan eksperimen tanpa menggunakan teknik augmentasi data, konfigurasi pada repositori ini menghasilkan tingkat **akurasi sebesar 92%**. Pengujian XAI menggunakan LIME juga memverifikasi bahwa titik berat ekstraksi fitur dari EfficientNet-B0 telah terkunci secara akurat pada area lesi penyakit (bercak pada daun).
