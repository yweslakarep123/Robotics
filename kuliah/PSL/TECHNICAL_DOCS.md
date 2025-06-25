# Dokumentasi Teknis - BLDC Motor Audio Classifier

## Arsitektur Program

### 1. Kelas Utama: `BLDCMotorClassifier`

Program menggunakan pendekatan berorientasi objek dengan kelas utama `BLDCMotorClassifier` yang mengelola semua fungsi:

- **Audio Recording**: Menggunakan PyAudio untuk menangkap suara
- **Signal Processing**: Filter Butterworth dan FFT
- **Feature Extraction**: Ekstraksi fitur untuk klasifikasi
- **Machine Learning**: Random Forest Classifier
- **Visualization**: Plot hasil analisis

### 2. Alur Program

```
Audio Input → Preprocessing → Filtering → FFT → Feature Extraction → Classification
```

## Implementasi Filter Butterworth

### Teori Dasar

Filter Butterworth adalah filter IIR (Infinite Impulse Response) yang memiliki respons frekuensi yang maksimal flat dalam passband. Transfer function:

```
H(s) = 1 / √(1 + (s/jωc)^2n)
```

Dimana:
- `ωc` = cutoff frequency
- `n` = order filter

### Implementasi dalam Kode

```python
def butterworth_filter(self, audio_data, lowcut=100, highcut=8000, order=5):
    nyquist = self.sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Membuat filter Butterworth bandpass
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Menerapkan filter
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    return filtered_audio, b, a
```

### Parameter Filter

- **Lowcut**: 100 Hz - Menghilangkan frekuensi rendah (noise lingkungan)
- **Highcut**: 8000 Hz - Menghilangkan frekuensi tinggi (noise digital)
- **Order**: 5 - Orde filter (semakin tinggi semakin tajam)

## Analisis FFT

### Implementasi FFT

```python
def compute_fft(self, audio_data):
    # Menghitung FFT
    fft_result = fft(audio_data)
    freqs = fftfreq(len(audio_data), 1/self.sample_rate)
    
    # Menghitung magnitude dan power spectrum
    magnitude = np.abs(fft_result)
    power_spectrum = magnitude**2
    
    return positive_freqs, positive_magnitude, positive_power
```

### Power Spectrum

Power spectrum menunjukkan distribusi energi sinyal terhadap frekuensi:

```
P(f) = |X(f)|²
```

Dimana `X(f)` adalah hasil FFT dari sinyal `x(t)`.

## Ekstraksi Fitur

### Fitur yang Diekstrak

1. **RMS (Root Mean Square)**
   ```python
   rms = np.sqrt(np.mean(audio_data**2))
   ```
   - Ukuran energi sinyal
   - Indikator amplitudo rata-rata

2. **Peak Frequency**
   ```python
   peak_idx = np.argmax(magnitude)
   peak_frequency = freqs[peak_idx]
   ```
   - Frekuensi dengan magnitude tertinggi
   - Indikator frekuensi dominan motor

3. **Spectral Centroid**
   ```python
   spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
   ```
   - Pusat massa spektrum
   - Indikator "brightness" sinyal

4. **Spectral Bandwidth**
   ```python
   spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / np.sum(magnitude))
   ```
   - Lebar pita spektrum
   - Indikator variasi frekuensi

5. **Spectral Rolloff**
   ```python
   cumulative_energy = np.cumsum(magnitude)
   rolloff_threshold = 0.85 * cumulative_energy[-1]
   rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0][0]
   spectral_rolloff = freqs[rolloff_idx]
   ```
   - Frekuensi di bawah 85% energi
   - Indikator distribusi energi

6. **Zero Crossing Rate**
   ```python
   zero_crossing_rate = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
   ```
   - Jumlah perpotongan dengan sumbu nol
   - Indikator kompleksitas sinyal

7. **MFCC (Mel-frequency Cepstral Coefficients)**
   ```python
   mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
   ```
   - 13 koefisien pertama
   - Representasi spektral yang kompak

8. **Spectral Contrast**
   ```python
   contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
   ```
   - Kontras spektral
   - Indikator variasi spektral

## Klasifikasi Machine Learning

### Algoritma: Random Forest

Random Forest dipilih karena:
- Robust terhadap overfitting
- Dapat menangani data numerik dengan baik
- Memberikan feature importance
- Tidak memerlukan scaling yang ketat

### Implementasi

```python
from sklearn.ensemble import RandomForestClassifier

self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Preprocessing

1. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler
   self.scaler = StandardScaler()
   X_scaled = self.scaler.fit_transform(X)
   ```

2. **Train-Test Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

## Karakteristik BLDC Motor A2122

### Frekuensi Karakteristik

- **Healthy Motor**: 
  - Frekuensi dasar: 100-500 Hz
  - Harmonik: 2x, 3x frekuensi dasar
  - Noise rendah

- **Bearing Problem**:
  - Frekuensi tinggi: 2000-6000 Hz
  - Noise meningkat
  - Spektral bandwidth tinggi

- **Propeller Problem**:
  - Frekuensi tidak stabil
  - Modulasi amplitudo
  - Zero crossing rate tinggi

### Data Sintetik

Program menggunakan data sintetik untuk training:

```python
def generate_synthetic_data(self, num_samples=50):
    # Healthy motor
    signal_healthy = np.sin(2 * np.pi * freq_base * t) + 
                    0.3 * np.sin(2 * np.pi * 2 * freq_base * t)
    
    # Bearing problem
    signal_bearing = np.sin(2 * np.pi * freq_base * t) +
                    0.5 * np.sin(2 * np.pi * 2000 * t)
    
    # Propeller problem
    freq_var = freq_base + 50 * np.sin(2 * np.pi * 0.5 * t)
    signal_propeller = np.sin(2 * np.pi * freq_var * t)
```

## Optimasi Performa

### 1. Sampling Rate
- Default: 44100 Hz
- Cukup untuk analisis audio motor
- Balance antara akurasi dan performa

### 2. Chunk Size
- Default: 1024 samples
- Optimal untuk FFT
- Memory efficient

### 3. Filter Order
- Default: 5
- Balance antara sharpness dan computational cost

## Troubleshooting

### Masalah Umum

1. **PyAudio Installation**
   - Windows: `pip install pipwin && pipwin install pyaudio`
   - Linux: `sudo apt-get install portaudio19-dev`

2. **Memory Issues**
   - Kurangi chunk_size
   - Kurangi duration recording

3. **Accuracy Issues**
   - Gunakan data training real
   - Tune parameter filter
   - Tambah fitur baru

### Debugging

```python
# Debug audio recording
print(f"Audio shape: {audio_data.shape}")
print(f"Audio range: {audio_data.min():.3f} to {audio_data.max():.3f}")

# Debug filter
print(f"Filter coefficients b: {b}")
print(f"Filter coefficients a: {a}")

# Debug features
for key, value in features.items():
    print(f"{key}: {value:.4f}")
```

## Pengembangan Selanjutnya

1. **Real-time Processing**
   - Stream audio processing
   - Real-time classification

2. **Advanced Features**
   - Wavelet analysis
   - Time-frequency analysis
   - Deep learning models

3. **GUI Interface**
   - Tkinter/PyQt interface
   - Real-time visualization

4. **Database Integration**
   - Store historical data
   - Trend analysis 