import numpy as np
import time
from scipy.special import erfc

# ─── Cell 2: QAMMapper ────────────────────────────────────────────────────────

class QAMMapper:
    SCHEMES = ['BPSK', 'QPSK', '16-QAM', '64-QAM']
    COLORS  = {
        'BPSK':   '#5DCAA5',
        'QPSK':   '#378ADD',
        '16-QAM': '#EF9F27',
        '64-QAM': '#D85A30',
    }

    def __init__(self, scheme: str):
        if scheme not in self.SCHEMES:
            raise ValueError(f"scheme must be one of {self.SCHEMES}")
        self.scheme = scheme
        self._build_constellation()

    @staticmethod
    def _to_gray(n: int) -> int:
        return n ^ (n >> 1)

    @staticmethod
    def _from_gray(g: int) -> int:
        n, mask = g, g >> 1
        while mask:
            n    ^= mask
            mask >>= 1
        return n

    def _build_constellation(self):
        if self.scheme == 'BPSK':
            self.bits_per_symbol = 1
            self.M               = 2
            self.norm_factor     = 1.0
            self.constellation   = {0: complex(-1, 0), 1: complex(+1, 0)}
        else:
            self.M               = 4 if self.scheme == 'QPSK' else int(self.scheme.split('-')[0])
            self.bits_per_symbol = int(np.log2(self.M))
            sqrt_M               = int(np.sqrt(self.M))
            k                    = self.bits_per_symbol // 2 

            pam_levels = [-(sqrt_M - 1) + 2 * i for i in range(sqrt_M)]
            avg_power_per_dim = np.mean([lv ** 2 for lv in pam_levels])
            self.norm_factor  = np.sqrt(2 * avg_power_per_dim)

            self.constellation = {}
            for bit_int in range(self.M):
                i_gray = (bit_int >> k) & ((1 << k) - 1)
                q_gray =  bit_int       & ((1 << k) - 1)

                i_idx  = self._from_gray(i_gray)
                q_idx  = self._from_gray(q_gray)

                I = pam_levels[i_idx]
                Q = pam_levels[q_idx]

                self.constellation[bit_int] = complex(I, Q) / self.norm_factor

        sorted_keys        = sorted(self.constellation.keys())
        self._sym_array    = np.array([self.constellation[k] for k in sorted_keys])
        self._key_array    = np.array(sorted_keys, dtype=int)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bps = self.bits_per_symbol
        if len(bits) % bps != 0:
            raise ValueError(f"{self.scheme} needs bit count divisible by {bps}.")

        n_syms  = len(bits) // bps
        symbols = np.empty(n_syms, dtype=complex)

        for i in range(n_syms):
            chunk   = bits[i * bps: (i + 1) * bps]
            bit_int = 0
            for b in chunk:
                bit_int = (bit_int << 1) | int(b)
            symbols[i] = self.constellation[bit_int]

        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        bps  = self.bits_per_symbol
        distances   = np.abs(symbols[:, np.newaxis] - self._sym_array[np.newaxis, :]) ** 2
        best_idx    = np.argmin(distances, axis=1)   
        bit_ints    = self._key_array[best_idx]      

        bits = np.zeros(len(symbols) * bps, dtype=int)
        for i, bit_int in enumerate(bit_ints):
            for b in range(bps):
                bits[i * bps + b] = (int(bit_int) >> (bps - 1 - b)) & 1

        return bits

    def get_constellation_points(self):
        labels = [f"{k:0{self.bits_per_symbol}b}" for k in self._key_array]
        return self._sym_array.copy(), labels


# ─── Cell 3: OFDM System Configuration ───────────────────────────────────────

class OFDMConfig:
    def __init__(
        self,
        N_fft:        int     = 64,
        CP_len:       int     = 16,
        pilot_bins:   list    = None,
        pilot_symbol: complex = 1 + 0j,
    ):
        self.N_fft        = N_fft
        self.CP_len       = CP_len
        self.pilot_symbol = pilot_symbol

        self.pilot_bins = pilot_bins if pilot_bins is not None else [7, 21, 43, 57]

        n_side            = N_fft // 2 - (N_fft // 2 - 27)  
        pos_bins          = list(range(1, 27))                
        neg_bins          = list(range(N_fft - 26, N_fft))   
        self.active_bins  = pos_bins + neg_bins               

        pilot_set         = set(self.pilot_bins)
        self.data_bins    = [b for b in self.active_bins if b not in pilot_set]

        self.N_data         = len(self.data_bins)       
        self.N_pilots       = len(self.pilot_bins)      
        self.N_active       = len(self.active_bins)     
        self.N_guard        = N_fft - self.N_active     
        self.symbol_duration = N_fft + CP_len           


# ─── Cell 4: OFDM Modulator (TX Chain) ───────────────────────────────────────

class OFDMModulator:
    def __init__(self, config: OFDMConfig, mapper: QAMMapper):
        self.cfg    = config
        self.mapper = mapper

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bps               = self.mapper.bits_per_symbol
        bits_per_ofdm_sym = bps * self.cfg.N_data   

        remainder = len(bits) % bits_per_ofdm_sym
        if remainder != 0:
            bits = np.concatenate([bits, np.zeros(bits_per_ofdm_sym - remainder, dtype=int)])

        qam_symbols = self.mapper.modulate(bits)
        n_ofdm      = len(qam_symbols) // self.cfg.N_data
        sym_matrix  = qam_symbols.reshape(n_ofdm, self.cfg.N_data)
        tx_frames = np.zeros((n_ofdm, self.cfg.symbol_duration), dtype=complex)

        for idx, data_row in enumerate(sym_matrix):
            freq_frame = np.zeros(self.cfg.N_fft, dtype=complex)
            for i, bin_k in enumerate(self.cfg.data_bins):
                freq_frame[bin_k] = data_row[i]
            for bin_k in self.cfg.pilot_bins:
                freq_frame[bin_k] = self.cfg.pilot_symbol

            time_frame = np.fft.ifft(freq_frame)   
            cp = time_frame[-self.cfg.CP_len:]      
            tx_frames[idx] = np.concatenate([cp, time_frame])

        tx_signal = tx_frames.flatten()
        return tx_signal
    

# ─── Cell 5: OFDM Demodulator (RX Chain) ─────────────────────────────────────

class OFDMDemodulator:
    def __init__(self, config: OFDMConfig, mapper: QAMMapper):
        self.cfg    = config
        self.mapper = mapper

    def demodulate(self, rx_signal: np.ndarray):
        sym_dur = self.cfg.symbol_duration
        n_ofdm  = len(rx_signal) // sym_dur

        rx_matrix = rx_signal[:n_ofdm * sym_dur].reshape(n_ofdm, sym_dur)
        rx_data_syms  = np.zeros((n_ofdm, self.cfg.N_data),   dtype=complex)
        rx_pilot_syms = np.zeros((n_ofdm, self.cfg.N_pilots), dtype=complex)

        for idx, rx_frame in enumerate(rx_matrix):
            without_cp = rx_frame[self.cfg.CP_len:]       
            freq_frame = np.fft.fft(without_cp)           

            for i, bin_k in enumerate(self.cfg.data_bins):
                rx_data_syms[idx, i] = freq_frame[bin_k]
            for i, bin_k in enumerate(self.cfg.pilot_bins):
                rx_pilot_syms[idx, i] = freq_frame[bin_k]

        flat_syms = rx_data_syms.flatten()                
        rx_bits   = self.mapper.demodulate(flat_syms)
        return rx_bits, rx_data_syms, rx_pilot_syms
    

# ─── Cell 2: AWGN Channel ─────────────────────────────────────────────────────

class AWGNChannel:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, snr_db: float) -> tuple:
        signal_power = np.mean(np.abs(signal) ** 2)
        if signal_power == 0:
            raise ValueError("Cannot apply AWGN to a zero-power signal.")

        snr_linear  = 10 ** (snr_db / 10)       
        noise_power = signal_power / snr_linear  
        noise_std = np.sqrt(noise_power / 2)
        noise     = noise_std * (
            self.rng.standard_normal(len(signal)) +
            1j * self.rng.standard_normal(len(signal))
        )

        noisy_signal = signal + noise
        actual_noise_power = np.mean(np.abs(noise) ** 2)
        actual_snr_db      = 10 * np.log10(signal_power / actual_noise_power)

        return noisy_signal, actual_snr_db, actual_noise_power


# ─── Cell 3: Multipath FIR Channel ───────────────────────────────────────────

class MultipathChannel:
    PROFILES = {
        'PedA': {
            'delays':    [0, 2, 4, 8],
            'powers_db': [0.0, -9.7, -19.2, -22.8],
            'desc':      'ITU-R Pedestrian A',
        },
        'PedB': {
            'delays':    [0, 1, 3, 7, 11, 15],
            'powers_db': [0.0, -0.9, -4.9, -8.0, -7.8, -23.9],
            'desc':      'ITU-R Pedestrian B',
        },
        'VehA': {
            'delays':    [0, 2, 4, 6, 8, 10],
            'powers_db': [-0.8, -7.7, -13.0, -14.7, -19.2, -25.8],
            'desc':      'ITU-R Vehicular A',
        },
    }

    def __init__(self, profile: str = 'PedA', seed: int = 42):
        self.profile_name = profile
        self.profile      = self.PROFILES[profile]
        self.rng          = np.random.default_rng(seed)
        self.awgn         = AWGNChannel(seed=seed + 1)
        self._generate_taps()

    def _generate_taps(self):
        delays    = self.profile['delays']
        powers_db = self.profile['powers_db']
        amplitudes = np.array([10 ** (p / 20) for p in powers_db])
        amplitudes /= np.sqrt(np.sum(amplitudes ** 2))
        phases = self.rng.uniform(0, 2 * np.pi, len(delays))

        max_delay = max(delays)
        self.h = np.zeros(max_delay + 1, dtype=complex)
        for delay, amp, phase in zip(delays, amplitudes, phases):
            self.h[delay] = amp * np.exp(1j * phase)
        self.H_freq = np.fft.fft(self.h, n=64)

    def apply(self, tx_signal: np.ndarray, snr_db: float) -> tuple:
        rx_conv = np.convolve(tx_signal, self.h)[:len(tx_signal)]
        rx_noisy, actual_snr, _ = self.awgn.apply(rx_conv, snr_db)
        return rx_noisy, self.h.copy(), self.H_freq.copy()


# ─── Cell 4: Rayleigh Flat Fading ─────────────────────────────────────────────

class RayleighFlatFadingChannel:
    def __init__(self, seed: int = 99):
        self.rng  = np.random.default_rng(seed)
        self.awgn = AWGNChannel(seed=seed + 5)

    def apply(self, tx_signal: np.ndarray,
              snr_db: float,
              symbol_duration: int) -> tuple:
        n_syms = len(tx_signal) // symbol_duration
        rx     = np.zeros_like(tx_signal)
        fades  = np.zeros(n_syms, dtype=complex)

        for i in range(n_syms):
            h = (self.rng.standard_normal() +
                 1j * self.rng.standard_normal()) / np.sqrt(2)
            fades[i] = h
            start = i * symbol_duration
            end   = start + symbol_duration
            rx[start:end] = h * tx_signal[start:end]

        remainder = len(tx_signal) - n_syms * symbol_duration
        if remainder > 0:
            h = (self.rng.standard_normal() +
                 1j * self.rng.standard_normal()) / np.sqrt(2)
            rx[n_syms * symbol_duration:] = h * tx_signal[n_syms * symbol_duration:]

        rx_noisy, actual_snr, _ = self.awgn.apply(rx, snr_db)
        return rx_noisy, fades


# ─── Cell 5: Least-Squares Channel Estimator ──────────────────────────────────

class LSChannelEstimator:
    def __init__(self, config: OFDMConfig):
        self.cfg = config

        def to_signed(bin_idx):
            return bin_idx if bin_idx < config.N_fft // 2 else bin_idx - config.N_fft

        self.pilot_signed = np.array([to_signed(b) for b in config.pilot_bins])
        self.data_signed  = np.array([to_signed(b) for b in config.data_bins])

        sort_idx               = np.argsort(self.pilot_signed)
        self.pilot_signed_sort = self.pilot_signed[sort_idx]
        self.pilot_sort_idx    = sort_idx  

    def estimate(self,
                 rx_pilot_syms:   np.ndarray,
                 pilot_tx_symbol: complex = 1 + 0j) -> np.ndarray:
        n_ofdm, n_pilots = rx_pilot_syms.shape
        H_est_data = np.zeros((n_ofdm, self.cfg.N_data), dtype=complex)

        for sym_idx in range(n_ofdm):
            H_at_pilots = rx_pilot_syms[sym_idx] / pilot_tx_symbol
            H_sorted = H_at_pilots[self.pilot_sort_idx]

            H_data_real = np.interp(
                self.data_signed,          
                self.pilot_signed_sort,    
                H_sorted.real,             
            )
            H_data_imag = np.interp(
                self.data_signed,
                self.pilot_signed_sort,
                H_sorted.imag,             
            )
            H_est_data[sym_idx] = H_data_real + 1j * H_data_imag

        return H_est_data


# ─── Cell 6: Frequency-Domain Equalizer (Zero-Forcing) ───────────────────────

class FrequencyDomainEqualizer:
    def __init__(self, method: str = 'ZF'):
        if method not in ('ZF', 'MMSE'):
            raise ValueError("method must be 'ZF' or 'MMSE'")
        self.method = method

    def equalize(self,
                 rx_data_syms: np.ndarray,
                 H_est:        np.ndarray,
                 snr_db:       float = 20.0) -> np.ndarray:
        if self.method == 'ZF':
            eps     = 1e-10
            eq_syms = rx_data_syms / (H_est + eps)
        elif self.method == 'MMSE':
            snr_linear = 10 ** (snr_db / 10)
            sigma2     = 1.0 / snr_linear
            W       = np.conj(H_est) / (np.abs(H_est) ** 2 + sigma2)
            eq_syms = W * rx_data_syms

        return eq_syms


# ─── Cell 7: OFDMReceiver + Full System Pipeline ──────────────────────────────

class OFDMReceiver:
    def __init__(self, config: OFDMConfig, mapper: QAMMapper,
                 eq_method: str = 'ZF'):
        self.cfg      = config
        self.mapper   = mapper
        self.demod    = OFDMDemodulator(config, mapper)
        self.estimator = LSChannelEstimator(config)
        self.equalizer = FrequencyDomainEqualizer(eq_method)

    def receive(self, rx_signal: np.ndarray, snr_db: float = 20.0):
        _, raw_syms, pilot_syms = self.demod.demodulate(rx_signal)
        H_est = self.estimator.estimate(pilot_syms, self.cfg.pilot_symbol)
        eq_syms = self.equalizer.equalize(raw_syms, H_est, snr_db)
        rx_bits = self.mapper.demodulate(eq_syms.flatten())

        return rx_bits, eq_syms, raw_syms, H_est


def theoretical_ber(scheme, snr_db_arr):
    """Gray-coded square M-QAM BER over AWGN, in terms of Es/N0 (snr_db_arr)."""
    mapper = QAMMapper(scheme)
    M, k = mapper.M, mapper.bits_per_symbol
    snr_lin = 10 ** (snr_db_arr / 10)
    EbN0 = snr_lin / k

    if scheme in ('BPSK', 'QPSK'):
        return 0.5 * erfc(np.sqrt(EbN0))
    sqrt_M = np.sqrt(M)
    return (4 / k) * (1 - 1 / sqrt_M) * 0.5 * erfc(np.sqrt(3 * k * EbN0 / (2 * (M - 1))))