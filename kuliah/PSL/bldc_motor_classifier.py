import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import librosa
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

class BLDCMotorClassifier:
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        """
        Inisialisasi classifier BLDC motor
        
        Parameters:
        - sample_rate: Sampling rate untuk audio (Hz)
        - chunk_size: Ukuran chunk audio
        - channels: Jumlah channel audio (1 = mono, 2 = stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def record_audio(self, duration=5, filename=None):
        """
        Merekam audio dari mikrofon laptop
        
        Parameters:
        - duration: Durasi rekaman dalam detik
        - filename: Nama file untuk menyimpan audio (opsional)
        
        Returns:
        - audio_data: Array numpy berisi data audio
        """
        print(f"Merekam audio selama {duration} detik...")
        
        # Membuka stream audio
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        num_chunks = int(self.sample_rate / self.chunk_size * duration)
        
        for i in range(num_chunks):
            data = stream.read(self.chunk_size)
            frames.append(data)
            print(f"Rekaman: {i+1}/{num_chunks}", end='\r')
        
        print("\nRekaman selesai!")
        
        # Menutup stream
        stream.stop_stream()
        stream.close()
        
        # Mengkonversi frames ke numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        
        # Menyimpan ke file jika filename diberikan
        if filename:
            self.save_audio(audio_data, filename)
        
        return audio_data
    
    def save_audio(self, audio_data, filename):
        """Menyimpan audio ke file WAV"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio disimpan ke: {filename}")
    
    def butterworth_filter(self, audio_data, lowcut=100, highcut=8000, order=5):
        """
        Menerapkan filter IIR Butterworth bandpass
        
        Parameters:
        - audio_data: Data audio input
        - lowcut: Frekuensi cutoff rendah (Hz)
        - highcut: Frekuensi cutoff tinggi (Hz)
        - order: Orde filter
        
        Returns:
        - filtered_audio: Audio yang sudah difilter
        - b, a: Koefisien filter
        """
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Membuat filter Butterworth bandpass
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Menerapkan filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio, b, a
    
    def compute_fft(self, audio_data):
        """
        Menghitung FFT dari audio data
        
        Parameters:
        - audio_data: Data audio
        
        Returns:
        - freqs: Array frekuensi
        - magnitude: Magnitude FFT
        - power_spectrum: Power spectrum
        """
        # Menghitung FFT
        fft_result = fft(audio_data)
        freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Menghitung magnitude dan power spectrum
        magnitude = np.abs(fft_result)
        power_spectrum = magnitude**2
        
        # Hanya ambil frekuensi positif
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        return positive_freqs, positive_magnitude, positive_power
    
    def extract_features(self, audio_data):
        """
        Mengekstrak fitur dari audio untuk klasifikasi
        
        Parameters:
        - audio_data: Data audio
        
        Returns:
        - features: Dictionary berisi fitur-fitur
        """
        # Menghitung FFT
        freqs, magnitude, power_spectrum = self.compute_fft(audio_data)
        
        # Fitur-fitur yang relevan untuk BLDC motor
        features = {}
        
        # RMS (Root Mean Square) - ukuran energi sinyal
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        
        # Peak frequency - frekuensi dengan magnitude tertinggi
        peak_idx = np.argmax(magnitude)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_magnitude'] = magnitude[peak_idx]
        
        # Spectral centroid - pusat massa spektrum
        features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Spectral bandwidth - lebar pita spektrum
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * magnitude) / np.sum(magnitude))
        
        # Spectral rolloff - frekuensi di bawah 85% energi
        cumulative_energy = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0][0]
        features['spectral_rolloff'] = freqs[rolloff_idx]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
        
        # MFCC (Mel-frequency cepstral coefficients) - 13 koefisien pertama
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfcc[i])
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_{i}'] = np.mean(contrast[i])
        
        return features
    
    def plot_results(self, original_audio, filtered_audio, b, a, freqs, power_spectrum, title="Analisis Audio BLDC Motor"):
        """
        Menampilkan plot hasil analisis
        
        Parameters:
        - original_audio: Audio asli
        - filtered_audio: Audio yang sudah difilter
        - b, a: Koefisien filter
        - freqs: Frekuensi FFT
        - power_spectrum: Power spectrum
        - title: Judul plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Audio asli vs filtered (time domain)
        time_original = np.linspace(0, len(original_audio)/self.sample_rate, len(original_audio))
        time_filtered = np.linspace(0, len(filtered_audio)/self.sample_rate, len(filtered_audio))
        
        axes[0, 0].plot(time_original, original_audio, label='Audio Asli', alpha=0.7)
        axes[0, 0].plot(time_filtered, filtered_audio, label='Audio Filtered', alpha=0.7)
        axes[0, 0].set_xlabel('Waktu (s)')
        axes[0, 0].set_ylabel('Amplitudo')
        axes[0, 0].set_title('Sinyal Audio (Time Domain)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Power spectrum
        axes[0, 1].semilogy(freqs, power_spectrum)
        axes[0, 1].set_xlabel('Frekuensi (Hz)')
        axes[0, 1].set_ylabel('Power Spectrum')
        axes[0, 1].set_title('Power Spectrum (Log Scale)')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, 10000)  # Batasi hingga 10kHz
        
        # Plot 3: FFT magnitude
        freqs_orig, magnitude_orig, _ = self.compute_fft(original_audio)
        freqs_filt, magnitude_filt, _ = self.compute_fft(filtered_audio)
        
        axes[1, 0].plot(freqs_orig, magnitude_orig, label='Audio Asli', alpha=0.7)
        axes[1, 0].plot(freqs_filt, magnitude_filt, label='Audio Filtered', alpha=0.7)
        axes[1, 0].set_xlabel('Frekuensi (Hz)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].set_title('FFT Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlim(0, 10000)
        
        # Plot 4: Frequency response filter
        w, h = signal.freqz(b, a, worN=8000)
        freq_response = self.sample_rate * w / (2 * np.pi)
        
        axes[1, 1].plot(freq_response, 20 * np.log10(abs(h)))
        axes[1, 1].set_xlabel('Frekuensi (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].set_title('Frequency Response Filter Butterworth')
        axes[1, 1].grid(True)
        axes[1, 1].set_xlim(0, 10000)
        
        # Plot 5: Koefisien filter
        axes[2, 0].stem(range(len(b)), b, label='Numerator (b)', markerfmt='bo')
        axes[2, 0].set_xlabel('Index')
        axes[2, 0].set_ylabel('Nilai Koefisien')
        axes[2, 0].set_title('Koefisien Filter Numerator (b)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        axes[2, 1].stem(range(len(a)), a, label='Denominator (a)', markerfmt='ro')
        axes[2, 1].set_xlabel('Index')
        axes[2, 1].set_ylabel('Nilai Koefisien')
        axes[2, 1].set_title('Koefisien Filter Denominator (a)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Menampilkan koefisien filter
        print("\n=== KOEFISIEN FILTER BUTTERWORTH ===")
        print(f"Numerator (b): {b}")
        print(f"Denominator (a): {a}")
        print(f"Orde filter: {len(b)-1}")
    
    def train_classifier(self, training_data):
        """
        Melatih classifier dengan data training
        
        Parameters:
        - training_data: List of tuples (audio_data, label)
        """
        print("Melatih classifier...")
        
        features_list = []
        labels = []
        
        for audio_data, label in training_data:
            features = self.extract_features(audio_data)
            features_list.append(list(features.values()))
            labels.append(label)
        
        # Konversi ke numpy array
        X = np.array(features_list)
        y = np.array(labels)
        
        # Split data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling fitur
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Training classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluasi
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Akurasi training: {train_score:.3f}")
        print(f"Akurasi testing: {test_score:.3f}")
        
        self.is_trained = True
    
    def collect_training_data(self, data_path='data'):
        """
        Mengumpulkan data training dengan merekam audio dari user.
        """
        print("\n=== Pengumpulan Data Training ===")
        print("Anda akan merekam audio untuk setiap kondisi motor.")
        
        # Buat direktori data jika belum ada
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        conditions = ['healthy', 'bearing_problem', 'propeller_problem']
        
        for condition in conditions:
            # Buat sub-direktori untuk setiap kondisi
            condition_path = os.path.join(data_path, condition)
            if not os.path.exists(condition_path):
                os.makedirs(condition_path)
            
            print(f"\n--- Merekam untuk kondisi: {condition.upper()} ---")
            
            while True:
                record_more = input(f"Apakah Anda ingin merekam sampel untuk kondisi '{condition}'? (y/n): ").lower()
                if record_more != 'y':
                    break
                    
                duration = float(input("Masukkan durasi rekaman (detik, cth: 5): "))
                audio_data = self.record_audio(duration)
                
                # Buat nama file unik
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(condition_path, f"{condition}_{timestamp}.wav")
                
                # Simpan audio
                self.save_audio(audio_data, filename)
                
        print("\nPengumpulan data selesai!")

    def load_training_data_from_disk(self, data_path='data'):
        """
        Memuat data training dari file WAV di disk.
        
        Returns:
        - training_data: List of tuples (audio_data, label)
        """
        print(f"\nMemuat data training dari direktori: {data_path}")
        training_data = []
        
        if not os.path.exists(data_path):
            print(f"⚠ Direktori data '{data_path}' tidak ditemukan.")
            return []
            
        for condition in os.listdir(data_path):
            condition_path = os.path.join(data_path, condition)
            if os.path.isdir(condition_path):
                for filename in os.listdir(condition_path):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(condition_path, filename)
                        try:
                            # Membaca file WAV menggunakan librosa
                            audio_data, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
                            training_data.append((audio_data, condition))
                            print(f"✓ Berhasil memuat: {filepath}")
                        except Exception as e:
                            print(f"✗ Gagal memuat {filepath}: {e}")
                            
        if not training_data:
            print("⚠ Tidak ada data training yang ditemukan.")
        else:
            print(f"\nTotal {len(training_data)} sampel training berhasil dimuat.")
            
        return training_data
    
    def classify_motor_condition(self, audio_data):
        """
        Mengklasifikasi kondisi motor berdasarkan audio
        
        Parameters:
        - audio_data: Data audio
        
        Returns:
        - prediction: Prediksi kondisi motor
        - confidence: Tingkat kepercayaan prediksi
        """
        if not self.is_trained:
            print("Classifier belum dilatih! Gunakan train_classifier() terlebih dahulu.")
            return None, None
        
        # Ekstrak fitur
        features = self.extract_features(audio_data)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scaling
        features_scaled = self.scaler.transform(features_array)
        
        # Prediksi
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = np.max(self.classifier.predict_proba(features_scaled))
        
        return prediction, confidence
    
    def generate_synthetic_data(self, num_samples=50):
        """
        Menghasilkan data sintetik untuk training (jika tidak ada data real)
        
        Parameters:
        - num_samples: Jumlah sampel per kelas
        
        Returns:
        - training_data: List of tuples (audio_data, label)
        """
        print("Menghasilkan data sintetik untuk training...")
        
        training_data = []
        
        # Healthy motor - frekuensi dominan di sekitar 100-500 Hz
        for i in range(num_samples):
            t = np.linspace(0, 5, int(5 * self.sample_rate))
            # Sinyal healthy dengan frekuensi dasar dan harmonik
            freq_base = np.random.uniform(100, 500)
            signal_healthy = (np.sin(2 * np.pi * freq_base * t) + 
                            0.3 * np.sin(2 * np.pi * 2 * freq_base * t) +
                            0.1 * np.sin(2 * np.pi * 3 * freq_base * t))
            # Tambah noise
            noise = np.random.normal(0, 0.1, len(signal_healthy))
            audio_data = signal_healthy + noise
            training_data.append((audio_data, 'healthy'))
        
        # Bearing problem - frekuensi tinggi dan noise
        for i in range(num_samples):
            t = np.linspace(0, 5, int(5 * self.sample_rate))
            freq_base = np.random.uniform(100, 500)
            # Tambah frekuensi tinggi untuk bearing problem
            signal_bearing = (np.sin(2 * np.pi * freq_base * t) +
                            0.5 * np.sin(2 * np.pi * 2000 * t) +
                            0.3 * np.sin(2 * np.pi * 4000 * t))
            # Noise lebih tinggi
            noise = np.random.normal(0, 0.3, len(signal_bearing))
            audio_data = signal_bearing + noise
            training_data.append((audio_data, 'bearing_problem'))
        
        # Propeller problem - frekuensi tidak stabil
        for i in range(num_samples):
            t = np.linspace(0, 5, int(5 * self.sample_rate))
            freq_base = np.random.uniform(100, 500)
            # Frekuensi yang berubah-ubah
            freq_var = freq_base + 50 * np.sin(2 * np.pi * 0.5 * t)
            signal_propeller = np.sin(2 * np.pi * freq_var * t)
            # Tambah noise
            noise = np.random.normal(0, 0.2, len(signal_propeller))
            audio_data = signal_propeller + noise
            training_data.append((audio_data, 'propeller_problem'))
        
        print(f"Data sintetik berhasil dibuat: {len(training_data)} sampel")
        return training_data
    
    def close(self):
        """Menutup PyAudio"""
        self.audio.terminate()

def main():
    """Fungsi utama program"""
    print("=== BLDC Motor A2122 Audio Classifier ===")
    print("Program untuk menangkap suara, memfilter noise, dan mengklasifikasi kondisi motor")
    
    # Inisialisasi classifier
    classifier = BLDCMotorClassifier()
    
    try:
        # Opsi untuk user
        while True:
            print("\nPilih opsi:")
            print("1. Rekam audio baru dan klasifikasi")
            print("2. Latih classifier dengan data REKAMAN BARU")
            print("3. Latih classifier dengan data SINTETIK")
            print("4. Keluar")
            
            choice = input("Masukkan pilihan (1-4): ")
            
            if choice == "1":
                # Rekam audio
                duration = float(input("Masukkan durasi rekaman (detik): "))
                audio_data = classifier.record_audio(duration=duration)
                
                # Filter audio
                print("\nMemfilter audio...")
                filtered_audio, b, a = classifier.butterworth_filter(audio_data)
                
                # Hitung FFT
                freqs, magnitude, power_spectrum = classifier.compute_fft(filtered_audio)
                
                # Plot hasil
                classifier.plot_results(audio_data, filtered_audio, b, a, freqs, power_spectrum)
                
                # Klasifikasi (jika classifier sudah dilatih)
                if classifier.is_trained:
                    prediction, confidence = classifier.classify_motor_condition(filtered_audio)
                    print(f"\n=== HASIL KLASIFIKASI ===")
                    print(f"Kondisi Motor: {prediction}")
                    print(f"Tingkat Kepercayaan: {confidence:.3f}")
                else:
                    print("\nClassifier belum dilatih. Gunakan opsi 2 atau 3 untuk melatih classifier.")
            
            elif choice == "2":
                # Kumpulkan data dari user
                classifier.collect_training_data()
                
                # Muat data yang baru direkam
                training_data = classifier.load_training_data_from_disk()
                
                if training_data:
                    # Latih classifier
                    classifier.train_classifier(training_data)
                    print("\n✓ Classifier berhasil dilatih dengan data rekaman Anda!")
                else:
                    print("\nPelatihan dibatalkan karena tidak ada data.")

            elif choice == "3":
                # Latih classifier dengan data sintetik
                training_data = classifier.generate_synthetic_data()
                classifier.train_classifier(training_data)
                
                # Test dengan data baru
                print("\nMenguji classifier dengan data baru...")
                test_audio = classifier.generate_synthetic_data(num_samples=1)[0][0]
                prediction, confidence = classifier.classify_motor_condition(test_audio)
                print(f"Hasil test: {prediction} (confidence: {confidence:.3f})")
            
            elif choice == "4":
                print("Program selesai.")
                break
            
            else:
                print("Pilihan tidak valid!")
    
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user.")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        classifier.close()

if __name__ == "__main__":
    main() 