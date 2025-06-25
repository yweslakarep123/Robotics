"""
Script untuk menginstall dependensi secara otomatis
Kompatibel dengan Python 3.13
"""

import subprocess
import sys
import os

def check_python_version():
    """Cek versi Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("⚠ Warning: Python 3.8+ recommended for optimal compatibility")
        return False
    elif version.minor >= 13:
        print("✓ Python 3.13 detected - using latest compatible versions")
        return True
    else:
        print("✓ Python version compatible")
        return True

def install_package(package):
    """Install package menggunakan pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} berhasil diinstall")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Gagal menginstall {package}")
        return False

def upgrade_pip():
    """Upgrade pip ke versi terbaru"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ Pip berhasil diupgrade")
        return True
    except subprocess.CalledProcessError:
        print("⚠ Gagal upgrade pip, melanjutkan dengan versi yang ada")
        return False

def main():
    """Install semua dependensi yang diperlukan"""
    print("=== Installing Dependencies for Python 3.13 ===")
    
    # Cek versi Python
    check_python_version()
    
    # Upgrade pip
    print("\n=== Upgrading Pip ===")
    upgrade_pip()
    
    # Daftar package yang diperlukan untuk Python 3.13
    packages = [
        "numpy>=1.26.0",
        "scipy>=1.12.0", 
        "matplotlib>=3.8.0",
        "scikit-learn>=1.4.0",
        "librosa>=0.10.1",
        "pandas>=2.2.0",
        "seaborn>=0.13.0"
    ]
    
    # Install packages
    print("\n=== Installing Core Packages ===")
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    # Install PyAudio (biasanya lebih sulit)
    print("\n=== Installing PyAudio ===")
    print("PyAudio mungkin memerlukan setup khusus...")
    
    # Coba install PyAudio dengan versi terbaru
    if install_package("pyaudio>=0.2.11"):
        print("✓ PyAudio berhasil diinstall")
        success_count += 1
    else:
        print("✗ Gagal menginstall PyAudio")
        print("\nUntuk Windows dengan Python 3.13, coba:")
        print("pip install pipwin")
        print("pipwin install pyaudio")
        print("\nAtau gunakan conda:")
        print("conda install pyaudio")
        print("\nUntuk Linux/Mac, coba:")
        print("sudo apt-get install portaudio19-dev python3-pyaudio")
        print("pip install pyaudio")
    
    # Install additional dependencies yang mungkin diperlukan
    print("\n=== Installing Additional Dependencies ===")
    additional_packages = [
        "setuptools>=68.0.0",
        "wheel>=0.41.0"
    ]
    
    for package in additional_packages:
        install_package(package)
    
    print(f"\n=== Summary ===")
    print(f"Berhasil menginstall: {success_count}/{len(packages)+1} packages")
    
    if success_count == len(packages) + 1:
        print("✓ Semua dependensi berhasil diinstall!")
        print("Anda dapat menjalankan program dengan: python bldc_motor_classifier.py")
    else:
        print("⚠ Beberapa package gagal diinstall. Silakan install manual.")
        print("\nCoba jalankan:")
        print("pip install -r requirements.txt")
    
    # Test import
    print("\n=== Testing Imports ===")
    test_imports = [
        "numpy",
        "scipy", 
        "matplotlib",
        "sklearn",
        "librosa",
        "pandas",
        "seaborn"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✓ {module} dapat diimport")
        except ImportError:
            print(f"✗ {module} tidak dapat diimport")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠ Modules yang gagal diimport: {failed_imports}")
    else:
        print("\n✓ Semua modules dapat diimport dengan sukses!")

if __name__ == "__main__":
    main() 