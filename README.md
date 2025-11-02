# Audio FFT Visualizer – Record, Analyze and Replay with Real-Time Visualization

This application is a Streamlit-based tool that allows users to record audio, extract spectral features using FFT, and visualize these features dynamically during synchronized playback.

## Features

- Record audio from the microphone with selectable duration and sample rate.
- Perform FFT-based spectral analysis to extract features such as:
  - Fundamental frequency
  - Peak frequencies and magnitudes
  - Spectral centroid and spectral rolloff
  - RMS energy and frequency band distribution
- Generate a circular real-time visualization based on extracted audio features.
- Replay the recorded audio while the visualization updates in synchronization.
- Save the recorded audio as a WAV file.
- Optional smoothing of feature transitions to improve visual continuity.
- Automatic silence detection to simplify visual output when no signal is present.

## Requirements

Install the required libraries:

```bash
pip install streamlit numpy matplotlib sounddevice scipy