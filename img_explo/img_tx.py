import numpy as np
import adi
import cv2

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
N_FFT = 128
SC_SPACING = 30000 
FS = int(N_FFT * SC_SPACING) # 3.84 MHz
CP_LEN = 32 

dc_idx = [0]
guard_idx = list(range(1, 11)) + list(range(118, 128))
pilot_idx = [12, 24, 36, 48, 60, 68, 80, 92, 104]
data_idx = [i for i in range(N_FFT) if i not in dc_idx + guard_idx + pilot_idx]

def create_ofdm_symbol(data_syms, pilot_syms):
    X = np.zeros(N_FFT, dtype=complex)
    X[data_idx] = data_syms
    X[pilot_idx] = pilot_syms
    x_time = np.fft.ifft(X) * np.sqrt(N_FFT)
    return np.concatenate((x_time[-CP_LEN:], x_time))

# ==========================================
# 2. IMAGE TO BITS & INTERLEAVING
# ==========================================
print("Loading and converting image...")
try:
    img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
except:
    print("test.jpg not found! Generating a dummy image.")
    img = np.kron([[0, 255] * 25, [255, 0] * 25] * 25, np.ones((1, 1), dtype=np.uint8))

bits = np.unpackbits(img.flatten()).tolist()

FRAME_CAPACITY = 28 * len(data_idx) * 2
TOTAL_FRAMES = int(np.ceil(len(bits) / FRAME_CAPACITY))
padded_bits = bits + [0] * ((TOTAL_FRAMES * FRAME_CAPACITY) - len(bits))

# --- BLOCK INTERLEAVER ---
depth = 32
padded_bits = np.reshape(padded_bits, (-1, depth)).T.flatten().tolist()
# -------------------------

qpsk_dict = {(0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j}
payload_qpsk = [qpsk_dict[(padded_bits[i], padded_bits[i+1])] for i in range(0, len(padded_bits), 2)]

# ==========================================
# 3. BUILD CONTINUOUS FRAMES (WITH SOF)
# ==========================================
tx_signal = []

np.random.seed(42); sync_sof = np.random.choice([1+0j, -1+0j], len(data_idx))  # Unique to Frame 1
np.random.seed(43); sync_data = np.random.choice([1+0j, -1+0j], len(data_idx)) # For Frames 2, 3, 4

for f in range(TOTAL_FRAMES):
    # Sync Symbol
    if f == 0:
        tx_signal.extend(create_ofdm_symbol(sync_sof, np.ones(len(pilot_idx)))) 
    else:
        tx_signal.extend(create_ofdm_symbol(sync_data, np.ones(len(pilot_idx)))) 
        
    # Header Symbol
    tx_signal.extend(create_ofdm_symbol(sync_data, np.ones(len(pilot_idx)))) 
    
    # Data Symbols
    chunk = payload_qpsk[f*(28*len(data_idx)) : (f+1)*(28*len(data_idx))]
    for i in range(28):
        sym_data = chunk[i*len(data_idx) : (i+1)*len(data_idx)]
        tx_signal.extend(create_ofdm_symbol(sym_data, np.ones(len(pilot_idx))))

tx_signal = np.array(tx_signal) * (2**14 * 0.4 / np.max(np.abs(tx_signal)))

# ==========================================
# 4. TRANSMIT
# ==========================================
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.tx_lo = 900000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -10

print(f"Transmitting {TOTAL_FRAMES} interleaved frames continuously...")
sdr.tx(tx_signal)
input("Press Enter to stop...")
sdr.tx_destroy_buffer()