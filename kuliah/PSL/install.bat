@echo off
echo === BLDC Motor Classifier - Install Dependencies for Python 3.13 ===
echo.

echo Installing Python dependencies...
python install_dependencies.py

echo.
echo === Installing PyAudio (Special handling for Python 3.13) ===
python pyaudio_install_helper.py

echo.
echo === Installation Complete ===
echo.
echo To run the program:
echo python bldc_motor_classifier.py
echo.
pause 