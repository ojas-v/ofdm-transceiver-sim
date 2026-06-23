import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from ofdm_lib import QAMMapper, OFDMConfig, OFDMModulator

cfg = OFDMConfig()
mapper = QAMMapper("16-QAM")
mod = OFDMModulator(cfg, mapper)

rng = np.random.default_rng(0)
bits = rng.integers(0, 2, mapper.bits_per_symbol * cfg.N_data * 200)
tx = mod.modulate(bits)

# add a touch of phase noise + frequency offset, like a real RF front end would
fs = 20e6  # 20 MHz, matches 802.11a assumption from earlier
t = np.arange(len(tx)) / fs
freq_offset = 2000  # 2 kHz CFO, typical for a cheap oscillator
tx_rf = tx * np.exp(1j * 2 * np.pi * freq_offset * t)

f, psd = scipy_signal.welch(tx_rf, fs=fs, nperseg=1024, return_onesided=False)
f = np.fft.fftshift(f) / 1e6  # MHz
psd_db = 10 * np.log10(np.fft.fftshift(psd) + 1e-14)

fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
ax.plot(f, psd_db, color="#39FF14", lw=1.0)  # spectrum-analyzer green
ax.fill_between(f, psd_db.min() - 5, psd_db, color="#39FF14", alpha=0.08)
ax.set_xlabel("Frequency offset (MHz)", color="#ccc")
ax.set_ylabel("Power (dB)", color="#ccc")
ax.set_title("OFDM TX Spectrum — 16-QAM, 64-pt FFT, 20 MHz BW", color="#eee")
ax.tick_params(colors="#888")
ax.grid(color="#1a3a1a", alpha=0.5)
for spine in ax.spines.values():
    spine.set_color("#1a3a1a")
plt.savefig("spectrum_analyzer_view.png", dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
plt.show()