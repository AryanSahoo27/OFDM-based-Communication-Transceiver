import numpy as np
import adi

# ==========================================
# 1. OFDM SYSTEM PARAMETERS
# ==========================================
N_FFT = 128
SC_SPACING = 30000 
FS = int(N_FFT * SC_SPACING) # 3.84 MHz
CP_LEN = 32 
FRAME_SYMBOLS = 30

all_idx = np.arange(N_FFT)
dc_idx = [0]
guard_idx = list(range(1, 11)) + list(range(118, 128))
pilot_idx = [12, 24, 36, 48, 60, 68, 80, 92, 104]
data_idx = [i for i in all_idx if i not in dc_idx + guard_idx + pilot_idx]

def create_ofdm_symbol(data_syms, pilot_syms):
    X = np.zeros(N_FFT, dtype=complex)
    X[data_idx] = data_syms
    X[pilot_idx] = pilot_syms
    x_time = np.fft.ifft(X) * np.sqrt(N_FFT)
    x_cp = np.concatenate((x_time[-CP_LEN:], x_time))
    return x_cp

# ==========================================
# 2. FRAME GENERATION (Deterministic)
# ==========================================
tx_frame = []
qpsk_map = [1+1j, -1+1j, -1-1j, 1-1j]

# SYMBOL 0: SYNC (Hardcoded Seed so Rx knows it exactly)
np.random.seed(42) 
sync_data = np.random.choice([1+0j, -1+0j], len(data_idx))
tx_frame.extend(create_ofdm_symbol(sync_data, np.ones(len(pilot_idx))))

# SYMBOL 1 to 29: HEADER & DATA (Random Payload)
for _ in range(FRAME_SYMBOLS - 1):
    payload = np.random.choice(qpsk_map, len(data_idx))
    tx_frame.extend(create_ofdm_symbol(payload, np.ones(len(pilot_idx))))

tx_signal = np.array(tx_frame)
tx_signal = tx_signal * (2**14 * 0.4 / np.max(np.abs(tx_signal)))

# ==========================================
# 3. TRANSMISSION
# ==========================================
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.tx_lo = 900000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -10

print("Transmitting OFDM Frame...")
sdr.tx(tx_signal)
input("Press Enter to stop...")
sdr.tx_destroy_buffer()