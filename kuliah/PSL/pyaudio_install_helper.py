"""
Script khusus untuk menginstall PyAudio di Python 3.13
PyAudio sering bermasalah dengan Python versi terbaru
"""

import subprocess
import sys
import platform
import os

def get_system_info():
    """Dapatkan informasi sistem"""
    system = platform.system()
    architecture = platform.architecture()[0]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"Sistem: {system}")
    print(f"Arsitektur: {architecture}")
    print(f"Python: {python_version}")
    
    return system, architecture, python_version

def install_pyaudio_windows():
    """Install PyAudio di Windows"""
    print("\n=== Installing PyAudio on Windows ===")
    
    methods = [
        ("pipwin", "pip install pipwin && pipwin install pyaudio"),
        ("conda", "conda install -c conda-forge pyaudio"),
        ("wheel", "pip install pyaudio --only-binary=all"),
        ("build", "pip install pyaudio")
    ]
    
    for method_name, command in methods:
        print(f"\nMencoba method: {method_name}")
        try:
            if method_name == "pipwin":
                # Install pipwin first
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
                subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
            elif method_name == "conda":
                subprocess.check_call(["conda", "install", "-c", "conda-forge", "pyaudio", "-y"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
            
            print(f"✓ PyAudio berhasil diinstall dengan method: {method_name}")
            return True
            
        except subprocess.CalledProcessError:
            print(f"✗ Method {method_name} gagal")
            continue
    
    return False

def install_pyaudio_linux():
    """Install PyAudio di Linux"""
    print("\n=== Installing PyAudio on Linux ===")
    
    # Install system dependencies
    try:
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "portaudio19-dev", "python3-pyaudio"])
        print("✓ System dependencies berhasil diinstall")
    except subprocess.CalledProcessError:
        print("⚠ Gagal install system dependencies")
    
    # Install PyAudio
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
        print("✓ PyAudio berhasil diinstall")
        return True
    except subprocess.CalledProcessError:
        print("✗ Gagal install PyAudio")
        return False

def install_pyaudio_macos():
    """Install PyAudio di macOS"""
    print("\n=== Installing PyAudio on macOS ===")
    
    # Install portaudio using brew
    try:
        subprocess.check_call(["brew", "install", "portaudio"])
        print("✓ PortAudio berhasil diinstall")
    except subprocess.CalledProcessError:
        print("⚠ Gagal install PortAudio, mencoba tanpa brew")
    
    # Install PyAudio
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
        print("✓ PyAudio berhasil diinstall")
        return True
    except subprocess.CalledProcessError:
        print("✗ Gagal install PyAudio")
        return False

def test_pyaudio():
    """Test apakah PyAudio berfungsi"""
    try:
        import pyaudio
        print("✓ PyAudio berhasil diimport")
        
        # Test basic functionality
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"✓ PyAudio berfungsi - {device_count} audio devices terdeteksi")
        p.terminate()
        return True
        
    except ImportError:
        print("✗ PyAudio tidak dapat diimport")
        return False
    except Exception as e:
        print(f"⚠ PyAudio diimport tapi ada error: {e}")
        return False

def main():
    """Main function"""
    print("=== PyAudio Installation Helper for Python 3.13 ===")
    
    # Get system info
    system, architecture, python_version = get_system_info()
    
    # Check if PyAudio is already installed
    if test_pyaudio():
        print("\n✓ PyAudio sudah terinstall dan berfungsi!")
        return
    
    # Install based on system
    success = False
    if system == "Windows":
        success = install_pyaudio_windows()
    elif system == "Linux":
        success = install_pyaudio_linux()
    elif system == "Darwin":  # macOS
        success = install_pyaudio_macos()
    else:
        print(f"⚠ Sistem {system} tidak didukung")
        return
    
    # Test installation
    if success:
        print("\n=== Testing Installation ===")
        if test_pyaudio():
            print("✓ PyAudio berhasil diinstall dan berfungsi!")
        else:
            print("⚠ PyAudio terinstall tapi tidak berfungsi dengan baik")
    else:
        print("\n✗ Gagal menginstall PyAudio")
        print("\nAlternatif:")
        print("1. Gunakan conda: conda install -c conda-forge pyaudio")
        print("2. Download wheel manual dari: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        print("3. Gunakan virtual environment dengan Python versi lama")

if __name__ == "__main__":
    main() 