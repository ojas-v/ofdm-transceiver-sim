try:
    import streamlit as st  # type: ignore
except Exception as e:
    raise ImportError("streamlit is required to run this app. Install it with: pip install streamlit") from e
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from ofdm_lib import (
    QAMMapper, OFDMConfig, OFDMModulator, OFDMReceiver,
    AWGNChannel, MultipathChannel, RayleighFlatFadingChannel,
    theoretical_ber,
)

st.set_page_config(page_title="OFDM Transceiver Sim", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#444", "axes.labelcolor": "#ccc",
    "xtick.color": "#888", "ytick.color": "#888", "text.color": "#eee",
    "grid.color": "#2a2a2a", "grid.linestyle": "--", "grid.alpha": 0.5,
})

st.title("OFDM Transceiver — Live Simulation")
st.caption("802.11a-style OFDM · 64-pt FFT · Gray-coded QAM · LS estimation + ZF/MMSE equalization")

# ── sidebar controls ──────────────────────────────────────────────
with st.sidebar:
    st.header("System Config")
    scheme = st.selectbox("Modulation", ["BPSK", "QPSK", "16-QAM", "64-QAM"], index=2)
    channel_type = st.selectbox("Channel", ["AWGN", "Multipath (PedA)", "Multipath (PedB)",
                                            "Multipath (VehA)", "Rayleigh flat fading"])
    snr_db = st.slider("SNR (Es/N0, dB)", 0.0, 30.0, 15.0, 0.5)
    eq_method = st.selectbox("Equalizer", ["ZF", "MMSE"])
    n_ofdm = st.slider("OFDM symbols to simulate", 20, 500, 100, 10)
    seed = st.number_input("Random seed", value=0, step=1)
    run_btn = st.button("Run simulation", type="primary")

# ── simulation ─────────────────────────────────────────────────────
def run_sim(scheme, channel_type, snr_db, eq_method, n_ofdm, seed):
    cfg = OFDMConfig()
    mapper = QAMMapper(scheme)
    mod = OFDMModulator(cfg, mapper)
    rx = OFDMReceiver(cfg, mapper, eq_method=eq_method)

    rng = np.random.default_rng(seed)
    bps = mapper.bits_per_symbol
    tx_bits = rng.integers(0, 2, bps * cfg.N_data * n_ofdm)
    tx_signal = mod.modulate(tx_bits)

    if channel_type == "AWGN":
        ch = AWGNChannel(seed=seed)
        rx_signal, _, _ = ch.apply(tx_signal, snr_db)
    elif channel_type.startswith("Multipath"):
        profile = channel_type.split("(")[1].rstrip(")")
        ch = MultipathChannel(profile=profile, seed=seed)
        rx_signal, h_cir, H_cfr = ch.apply(tx_signal, snr_db)
    else:
        ch = RayleighFlatFadingChannel(seed=seed)
        rx_signal, fades = ch.apply(tx_signal, snr_db, cfg.symbol_duration)

    rx_bits, eq_syms, raw_syms, H_est = rx.receive(rx_signal, snr_db)

    n_cmp = min(len(tx_bits), len(rx_bits))
    ber = np.sum(tx_bits[:n_cmp] != rx_bits[:n_cmp]) / n_cmp

    return dict(cfg=cfg, mapper=mapper, tx_signal=tx_signal, rx_signal=rx_signal,
                eq_syms=eq_syms, raw_syms=raw_syms, ber=ber, n_bits=n_cmp)


if run_btn or "last_result" not in st.session_state:
    st.session_state.last_result = run_sim(scheme, channel_type, snr_db, eq_method, n_ofdm, seed)

res = st.session_state.last_result

# ── top row: BER numbers ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Measured BER", f"{res['ber']:.2e}")
theory = theoretical_ber(scheme, np.array([snr_db]))[0]
col2.metric("Theoretical BER (AWGN)", f"{theory:.2e}")
col3.metric("Bits simulated", f"{res['n_bits']:,}")
col4.metric("Spectral efficiency", f"{QAMMapper(scheme).bits_per_symbol} bits/sym")

# ── constellation ──────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Constellation")
    ideal_pts, _ = res["mapper"].get_constellation_points()
    raw_flat = res["raw_syms"].flatten()
    eq_flat = res["eq_syms"].flatten()
    lim = max(np.abs(ideal_pts).max() * 1.6, 1.2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, syms, title, c in [(axes[0], raw_flat, "Before EQ", "#D85A30"),
                               (axes[1], eq_flat, "After EQ", "#5DCAA5")]:
        ax.scatter(syms.real, syms.imag, s=4, alpha=0.3, c=c)
        ax.scatter(ideal_pts.real, ideal_pts.imag, s=40, c="white", marker="x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Power Spectral Density")
    freqs, psd = scipy_signal.welch(res["tx_signal"], nperseg=256, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd_db = 10 * np.log10(np.fft.fftshift(psd) + 1e-14)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(freqs, psd_db, color="#7F77DD", lw=1.3)
    ax2.fill_between(freqs, psd_db.min() - 2, psd_db, alpha=0.15, color="#7F77DD")
    ax2.set_xlabel("Normalized frequency")
    ax2.set_ylabel("PSD (dB)")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

# ── BER vs SNR curve, current point marked ─────────────────────────
st.subheader("BER vs SNR — where you are right now")
snr_range = np.linspace(0, 30, 200)
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.semilogy(snr_range, theoretical_ber(scheme, snr_range), color="#5DCAA5", lw=1.5, label=f"{scheme} theory")
ax3.scatter([snr_db], [max(res["ber"], 1e-7)], color="#EF9F27", s=80, zorder=5, label="current run")
ax3.set_xlabel("SNR (dB)"); ax3.set_ylabel("BER")
ax3.set_ylim(1e-6, 1)
ax3.legend(); ax3.grid(alpha=0.2, which="both")
st.pyplot(fig3)

st.caption(f"Channel: {channel_type} · Equalizer: {eq_method} · {n_ofdm} OFDM symbols · seed={seed}")