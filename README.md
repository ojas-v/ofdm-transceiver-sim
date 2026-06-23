# 📡 OFDM Transceiver Simulation

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-DSP-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B.svg)

An end-to-end, software-defined physical layer simulation of an 802.11a-style OFDM transceiver. This project models the complete mathematical pipeline from raw bit generation to complex QAM mapping, IFFT framing, physical channel distortion, and receiver-side channel estimation/equalization.

It includes a live, interactive web dashboard to visualize how different channel environments (AWGN, Multipath, Rayleigh) destroy signals and how DSP equalizers (ZF/MMSE) recover them.

## 🚀 Live Interactive Dashboard
[👉 Click here to launch the Live Interactive Dashboard](https://ofdm-transceiver-sim-hsf3wc7a7orfpxkxtup3mg.streamlit.app/)

## 🧠 Core Architecture Pipeline

**Transmitter (TX):**
* **Bit Generation:** Random payload generation.
* **QAM Mapping:** Gray-coded BPSK, QPSK, 16-QAM, or 64-QAM.
* **Subcarrier Mapping:** 64-point FFT grid (48 Data, 4 Pilots, 12 Guard/DC bins).
* **Modulation:** IFFT transforms frequency-domain bins to time-domain waveforms.
* **Cyclic Prefix (CP):** 16-sample CP prepended to mitigate Inter-Symbol Interference (ISI).

**Channel Models:**
* **AWGN:** Pure thermal noise modeling.
* **Multipath FIR:** ITU-R Pedestrian A/B and Vehicular A delay profiles (frequency-selective fading).
* **Rayleigh Flat Fading:** Block fading for dense urban non-line-of-sight environments.

**Receiver (RX):**
* **Synchronization:** CP removal and framing.
* **Demodulation:** FFT transforms the time-domain signal back to the frequency domain.
* **Channel Estimation:** Least-Squares (LS) estimation using known pilot subcarriers, with linear interpolation across data bins.
* **Equalization:** One-tap Zero-Forcing (ZF) or Minimum Mean Square Error (MMSE) equalization to reverse phase/amplitude distortion.
* **QAM Demapping:** Hard-decision nearest-neighbor decoding to recover the bitstream.

## 📊 Key Results & Visuals

### 1. Multipath Channel Equalization (Vehicular A)
Shows the receiver successfully untangling severe Inter-Symbol Interference (ISI) using an MMSE equalizer.
![Multipath Equalization](assets/The%20Multipath%20(VehA).png)

### 2. Deep Fading Penalty (Rayleigh Channel)
Demonstrates the massive BER penalty in non-line-of-sight environments compared to theoretical AWGN performance.
![Rayleigh Fading](assets/The%20Rayleigh%20Fade.png)

### 3. Ideal System Baseline (64-QAM / AWGN)
Establishes the system's baseline perfection in an ideal channel, showcasing a flawless 8x8 QAM grid and 0 measured errors.
![64-QAM Baseline](assets/The%2064-QAM%20Baseline.png)

### 4. TX RF Spectrum (64-pt FFT, 20 MHz BW)
Simulated transmitter Power Spectral Density showcasing the flat OFDM plateau and sharp guard-band roll-offs.
![RF Spectrum](assets/The%20RF%20Spectrum.png)

## 🛠️ Future Roadmap
1. Implement Forward Error Correction (FEC) using LDPC or Convolutional Codes (Viterbi decoding).
2. Transition simulation to real-time SDR loopback using RTL-SDR / PlutoSDR hardware.

## 💻 Quick Start (Run Locally)

Clone the repository and run the interactive dashboard on your local machine:

```bash
# Clone the repo
git clone [https://github.com/ojas-v/ofdm-transceiver-sim.git](https://github.com/ojas-v/ofdm-transceiver-sim.git)
cd ofdm-transceiver-sim

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib streamlit

# Launch the dashboard
streamlit run app.py
