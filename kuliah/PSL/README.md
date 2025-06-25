# BLDC Motor A2122 Audio Classifier

Program Python untuk menangkap suara dari laptop, memfilter noise menggunakan FFT dan filter IIR Butterworth, serta mengklasifikasi kondisi BLDC motor A2122.

**Kompatibel dengan Python 3.13**

## Fitur

- **Perekaman Audio**: Menangkap suara dari mikrofon laptop
- **Filtering**: Menggunakan filter IIR Butterworth bandpass untuk menghilangkan noise
- **FFT Analysis**: Analisis frekuensi menggunakan Fast Fourier Transform
- **Klasifikasi**: Mengklasifikasi kondisi motor menjadi:
  - Healthy (Sehat)
  - Bearing Problem (Masalah bearing)
  - Propeller Problem (Masalah propeller)
- **Visualization**: Plot power spectrum, koefisien filter, dan hasil analisis

## Persyaratan Sistem

- **Python**: 3.8+ (Direkomendasikan Python 3.13)
- **OS**: Windows, Linux, macOS
- **Hardware**: Mikrofon laptop yang berfungsi

## Instalasi

### Metode 1: Instalasi Otomatis (Direkomendasikan)

**Windows:**
```bash
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### Metode 2: Instalasi Manual

1. Install dependensi Python:
```bash
pip install -r requirements.txt
```

2. Install PyAudio (mungkin memerlukan setup khusus):
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

### Metode 3: Menggunakan Conda (Alternatif)

```bash
conda create -n bldc_classifier python=3.13
conda activate bldc_classifier
conda install -c conda-forge pyaudio
pip install -r requirements.txt
```

## Penggunaan

1. Jalankan program:
```bash
python bldc_motor_classifier.py
```

2. Pilih opsi:
   - **Opsi 1**: Rekam audio baru dan klasifikasi
   - **Opsi 2**: Latih classifier dengan data sintetik
   - **Opsi 3**: Keluar

3. Untuk klasifikasi:
   - Masukkan durasi rekaman
   - Program akan merekam suara motor
   - Audio akan difilter menggunakan Butterworth filter
   - Hasil FFT dan power spectrum akan ditampilkan
   - Koefisien filter akan ditampilkan
   - Kondisi motor akan diklasifikasi

## Parameter Filter

- **Lowcut**: 100 Hz (frekuensi cutoff rendah)
- **Highcut**: 8000 Hz (frekuensi cutoff tinggi)
- **Order**: 5 (orde filter Butterworth)

## Fitur yang Diekstrak

Program mengekstrak fitur-fitur berikut untuk klasifikasi:
- RMS (Root Mean Square)
- Peak frequency dan magnitude
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero crossing rate
- MFCC (13 koefisien)
- Spectral contrast

## Output

Program akan menampilkan:
1. Plot sinyal audio asli vs filtered
2. Power spectrum (log scale)
3. FFT magnitude
4. Frequency response filter
5. Koefisien filter (numerator dan denominator)
6. Hasil klasifikasi kondisi motor

## Troubleshooting

### Masalah PyAudio di Python 3.13

PyAudio sering bermasalah dengan Python 3.13. Jika mengalami masalah:

1. **Gunakan script helper:**
   ```bash
   python pyaudio_install_helper.py
   ```

2. **Alternatif untuk Windows:**
   - Download wheel manual dari: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Pilih versi yang sesuai dengan Python 3.13 dan arsitektur sistem

3. **Gunakan Conda:**
   ```bash
   conda install -c conda-forge pyaudio
   ```

### Masalah Umum

- **Import Error**: Pastikan semua dependensi terinstall dengan benar
- **Audio Device Error**: Pastikan mikrofon laptop berfungsi
- **Memory Error**: Kurangi durasi rekaman atau chunk size

## Catatan

- Program menggunakan data sintetik untuk training jika tidak ada data real
- Untuk hasil yang lebih akurat, gunakan data training dari motor yang sebenarnya
- Pastikan lingkungan rekaman tenang untuk hasil yang optimal
- Sampling rate default: 44100 Hz

## Pengembangan

Untuk pengembangan lebih lanjut, lihat `TECHNICAL_DOCS.md` untuk dokumentasi teknis lengkap. 