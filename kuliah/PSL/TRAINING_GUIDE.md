# Panduan Melatih Classifier dengan Data Rekaman Sendiri

Dokumen ini menjelaskan cara menggunakan fitur baru untuk melatih model klasifikasi dengan data audio yang Anda rekam sendiri menggunakan mikrofon laptop.

## Konsep Dasar

Tujuan dari melatih model dengan data sendiri adalah untuk mendapatkan akurasi yang lebih tinggi dan lebih relevan dengan kondisi lingkungan dan motor spesifik Anda. Data sintetik bagus untuk memulai, tetapi data real selalu lebih baik.

**Alur Proses:**

1.  **Pengumpulan Data**: Anda merekam beberapa sampel audio untuk setiap kondisi motor (`healthy`, `bearing_problem`, `propeller_problem`).
2.  **Penyimpanan Data**: Program secara otomatis menyimpan file audio Anda dalam format `.wav` ke dalam direktori yang terstruktur, seperti:
    ```
    data/
    ├── healthy/
    │   ├── healthy_20231027_100000.wav
    │   └── healthy_20231027_100005.wav
    ├── bearing_problem/
    │   └── bearing_problem_20231027_100100.wav
    └── propeller_problem/
        └── propeller_problem_20231027_100200.wav
    ```
3.  **Loading Data**: Program memuat semua file audio dari direktori `data`.
4.  **Training**: Model `RandomForestClassifier` dilatih menggunakan fitur yang diekstrak dari audio yang Anda rekam.

## Langkah-langkah Penggunaan

### **Langkah 1: Jalankan Program**

Buka terminal atau command prompt, navigasi ke direktori project, dan jalankan program:

```bash
python bldc_motor_classifier.py
```

### **Langkah 2: Pilih Opsi Training**

Program akan menampilkan menu utama. Pilih **Opsi 2**:

```
Pilih opsi:
1. Rekam audio baru dan klasifikasi
2. Latih classifier dengan data REKAMAN BARU
3. Latih classifier dengan data SINTETIK
4. Keluar

Masukkan pilihan (1-4): 2
```

### **Langkah 3: Rekam Audio untuk Setiap Kondisi**

Program akan memandu Anda untuk merekam audio untuk setiap kondisi motor.

1.  **Mulai dengan kondisi `HEALTHY`**:
    - Program akan bertanya apakah Anda ingin merekam sampel. Ketik `y` (yes).
    - Masukkan durasi rekaman (misalnya, `5` detik).
    - Dekatkan mikrofon ke motor yang dalam kondisi *sehat* dan biarkan program merekam.
    - Anda dapat merekam beberapa sampel untuk kondisi ini dengan menjawab `y` lagi.
    - Jika sudah cukup, jawab `n` (no) untuk melanjutkan ke kondisi berikutnya.

    ```
    --- Merekam untuk kondisi: HEALTHY ---
    Apakah Anda ingin merekam sampel untuk kondisi 'healthy'? (y/n): y
    Masukkan durasi rekaman (detik, cth: 5): 5
    Merekam audio selama 5.0 detik...
    Rekaman selesai!
    Audio disimpan ke: data\healthy\healthy_20231027_100005.wav
    ```

2.  **Lanjutkan dengan `BEARING_PROBLEM` dan `PROPELLER_PROBLEM`**:
    - Ulangi proses perekaman untuk motor dengan simulasi masalah bearing dan masalah propeller.

### **Langkah 4: Proses Training Otomatis**

Setelah Anda selesai merekam semua data, program akan secara otomatis:
1.  Memuat semua file `.wav` dari direktori `data`.
2.  Mengekstrak fitur dari setiap file audio.
3.  Melatih model classifier.
4.  Menampilkan akurasi training dan testing.

```
Total 5 sampel training berhasil dimuat.
Melatih classifier...
Akurasi training: 1.000
Akurasi testing: 0.850

✓ Classifier berhasil dilatih dengan data rekaman Anda!
```

### **Langkah 5: Klasifikasi Menggunakan Model Baru**

Model Anda sekarang sudah terlatih dengan data Anda sendiri. Kembali ke menu utama, Anda dapat memilih **Opsi 1** untuk merekam audio baru dan mengklasifikasikannya menggunakan model yang baru saja Anda latih. Hasil klasifikasi sekarang akan jauh lebih akurat.

## Tips untuk Pengumpulan Data yang Baik

-   **Jumlah Sampel**: Semakin banyak sampel, semakin baik. Usahakan merekam **minimal 3-5 sampel** untuk setiap kondisi.
-   **Durasi Rekaman**: Durasi **3-5 detik** per sampel sudah cukup.
-   **Konsistensi**: Jaga jarak mikrofon ke motor agar konsisten di setiap rekaman.
-   **Lingkungan**: Lakukan perekaman di lingkungan yang relatif tenang untuk mengurangi noise yang tidak diinginkan.
-   **Variasi**: Jika memungkinkan, rekam dalam sedikit variasi kecepatan motor untuk membuat model lebih robust.

## Mengelola Data Training

-   Semua data rekaman disimpan di dalam folder `data`.
-   Anda dapat **menghapus** file rekaman yang jelek dari folder ini secara manual.
-   Anda dapat **menambahkan** data dari sesi rekaman yang berbeda dengan hanya menempatkan file `.wav` di dalam subfolder yang sesuai. Program akan secara otomatis memuat semua data yang ada saat Anda menjalankan proses training. 