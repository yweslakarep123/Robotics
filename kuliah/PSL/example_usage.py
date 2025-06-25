"""
Contoh penggunaan BLDC Motor Classifier
File ini menunjukkan cara menggunakan program secara programatik
"""

from bldc_motor_classifier import BLDCMotorClassifier
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def contoh_penggunaan():
    """Contoh penggunaan lengkap program"""
    
    print("=== Contoh Penggunaan BLDC Motor Classifier ===")
    
    # 1. Inisialisasi classifier
    classifier = BLDCMotorClassifier(sample_rate=44100, chunk_size=1024)
    
    try:
        # 2. Latih classifier dengan data sintetik
        print("\n1. Melatih classifier...")
        training_data = classifier.generate_synthetic_data(num_samples=30)
        classifier.train_classifier(training_data)
        
        # 3. Rekam audio (simulasi dengan data sintetik)
        print("\n2. Merekam audio...")
        # Untuk demo, kita gunakan data sintetik sebagai "rekaman"
        demo_audio = classifier.generate_synthetic_data(num_samples=1)[0][0]
        
        # 4. Filter audio
        print("\n3. Memfilter audio...")
        filtered_audio, b, a = classifier.butterworth_filter(demo_audio)
        
        # 5. Hitung FFT
        print("\n4. Menghitung FFT...")
        freqs, magnitude, power_spectrum = classifier.compute_fft(filtered_audio)
        
        # 6. Plot hasil
        print("\n5. Menampilkan plot...")
        classifier.plot_results(demo_audio, filtered_audio, b, a, freqs, power_spectrum, 
                              "Demo Analisis BLDC Motor A2122")
        
        # 7. Klasifikasi
        print("\n6. Mengklasifikasi kondisi motor...")
        prediction, confidence = classifier.classify_motor_condition(filtered_audio)
        
        print(f"\n=== HASIL KLASIFIKASI ===")
        print(f"Kondisi Motor: {prediction}")
        print(f"Tingkat Kepercayaan: {confidence:.3f}")
        
        # 8. Tampilkan informasi tambahan
        print(f"\n=== INFORMASI TEKNIS ===")
        print(f"Sampling Rate: {classifier.sample_rate} Hz")
        print(f"Durasi Audio: {len(demo_audio)/classifier.sample_rate:.2f} detik")
        print(f"Jumlah Sampel: {len(demo_audio)}")
        print(f"Orde Filter: {len(b)-1}")
        
        # 9. Analisis fitur
        print(f"\n=== ANALISIS FITUR ===")
        features = classifier.extract_features(filtered_audio)
        for key, value in features.items():
            print(f"{key}: {value:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        classifier.close()

def demo_filter_parameters():
    """Demo berbagai parameter filter"""
    
    print("\n=== Demo Parameter Filter ===")
    
    # Buat sinyal test
    sample_rate = 44100
    duration = 2
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Sinyal dengan multiple frekuensi
    signal_test = (np.sin(2 * np.pi * 200 * t) +  # 200 Hz
                   0.5 * np.sin(2 * np.pi * 1000 * t) +  # 1000 Hz
                   0.3 * np.sin(2 * np.pi * 5000 * t))  # 5000 Hz
    
    # Tambah noise
    noise = np.random.normal(0, 0.2, len(signal_test))
    audio_data = signal_test + noise
    
    classifier = BLDCMotorClassifier(sample_rate=sample_rate)
    
    # Test berbagai parameter filter
    filter_params = [
        (50, 1000, 3, "Low-pass filter"),
        (100, 8000, 5, "Band-pass filter"),
        (2000, 6000, 7, "High-pass filter")
    ]
    
    fig, axes = plt.subplots(len(filter_params), 2, figsize=(15, 12))
    fig.suptitle("Demo Parameter Filter Butterworth", fontsize=16)
    
    for i, (lowcut, highcut, order, title) in enumerate(filter_params):
        # Filter audio
        filtered_audio, b, a = classifier.butterworth_filter(audio_data, lowcut, highcut, order)
        
        # Plot time domain
        time = np.linspace(0, duration, len(audio_data))
        axes[i, 0].plot(time, audio_data, label='Original', alpha=0.7)
        axes[i, 0].plot(time, filtered_audio, label='Filtered', alpha=0.7)
        axes[i, 0].set_title(f"{title} (Order {order})")
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot frequency response
        w, h = signal.freqz(b, a, worN=8000)
        freq_response = sample_rate * w / (2 * np.pi)
        axes[i, 1].plot(freq_response, 20 * np.log10(abs(h)))
        axes[i, 1].set_title(f"Frequency Response - {title}")
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (dB)')
        axes[i, 1].grid(True)
        axes[i, 1].set_xlim(0, 10000)
    
    plt.tight_layout()
    plt.show()
    
    classifier.close()

if __name__ == "__main__":
    # Jalankan contoh penggunaan
    contoh_penggunaan()
    
    # Jalankan demo parameter filter
    demo_filter_parameters() 