"""
Test script untuk memverifikasi instalasi Python 3.13
"""

import sys
import platform

def test_python_version():
    """Test versi Python"""
    print("=== Python Version Test ===")
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.platform()}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version compatible")
        return True
    else:
        print("✗ Python version not compatible")
        return False

def test_packages():
    """Test package utama"""
    print("\n=== Packages Test ===")
    
    packages = ["numpy", "scipy", "matplotlib", "sklearn", "librosa", "pandas", "seaborn"]
    
    for package in packages:
        try:
            module = __import__(package)
            print(f"✓ {package}: OK")
        except ImportError:
            print(f"✗ {package}: FAILED")
    
    return True

def test_pyaudio():
    """Test PyAudio"""
    print("\n=== PyAudio Test ===")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"✓ PyAudio: {device_count} devices found")
        p.terminate()
        return True
    except ImportError:
        print("✗ PyAudio: NOT INSTALLED")
        return False

def main():
    """Main test"""
    print("=== Installation Test for Python 3.13 ===")
    
    test_python_version()
    test_packages()
    test_pyaudio()
    
    print("\n=== Test Complete ===")
    print("If PyAudio failed, run: python pyaudio_install_helper.py")

if __name__ == "__main__":
    main() 