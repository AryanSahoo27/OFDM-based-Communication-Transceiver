import numpy as np
import adi
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
N_FFT = 128
SC_SPACING = 30000
FS = int(N_FFT * SC_SPACING)
CP_LEN = 32
SYM_LEN = N_FFT + CP_LEN
TOTAL_FRAMES = 4 

dc_idx = [0]
guard_idx = list(range(1, 11)) + list(range(118, 128))
pilot_idx = [12, 24, 36, 48, 60, 68, 80, 92, 104]
data_idx = [i for i in range(N_FFT) if i not in dc_idx + guard_idx + pilot_idx]

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.rx_lo = 900000000
sdr.rx_buffer_size = 100000 # Large buffer for full image capture
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 40

# Local Sync Templates
np.random.seed(42); sync_sof = np.random.choice([1+0j, -1+0j], len(data_idx))
np.random.seed(43); sync_data = np.random.choice([1+0j, -1+0j], len(data_idx))

local_sof = np.zeros(N_FFT, dtype=complex)
local_sof[data_idx] = sync_sof
local_sof_time = np.fft.ifft(local_sof) * np.sqrt(N_FFT)
local_sof_cp = np.concatenate((local_sof_time[-CP_LEN:], local_sof_time))

print("Capturing Image Data...")
for _ in range(5): sdr.rx()
rx_signal = sdr.rx()

# ==========================================
# 2. CFO & SOF ALIGNMENT
# ==========================================
# CFO Correction
delayed_rx = rx_signal[N_FFT:]
original_rx = rx_signal[:-N_FFT]
f_offset = np.angle(np.mean(np.conjugate(original_rx) * delayed_rx)) * (FS / (2 * np.pi * N_FFT))
t = np.arange(len(rx_signal)) / FS
rx_signal = rx_signal * np.exp(-1j * 2 * np.pi * f_offset * t)

# Frame Alignment (Look only for Frame 1 SOF)
corr = np.abs(np.correlate(rx_signal, local_sof_cp, mode='valid'))
sof_start = np.argmax(corr)

# Calculate exact start locations for all expected frames
frame_starts = [sof_start + (f * 30 * SYM_LEN) for f in range(TOTAL_FRAMES)]
print(f"Locked to SOF. Decoding {TOTAL_FRAMES} frames...")

all_clean_qpsk = []

# ==========================================
# 3. DECODE FRAMES
# ==========================================
for f_idx, frame_start in enumerate(frame_starts):
    if frame_start + (30 * SYM_LEN) >= len(rx_signal): 
        print("Warning: Buffer cut off early.")
        break
        
    rx_sync_sym = rx_signal[frame_start + CP_LEN : frame_start + SYM_LEN]
    rx_sync_fft = np.fft.fft(rx_sync_sym) / np.sqrt(N_FFT)
    
    current_sync = sync_sof if f_idx == 0 else sync_data
    H_est_data = rx_sync_fft[data_idx] / current_sync 
    H_est_pilots = rx_sync_fft[pilot_idx] / np.ones(len(pilot_idx))

    for i in range(2, 30):
        sym_start = frame_start + (i * SYM_LEN)
        rx_fft = np.fft.fft(rx_signal[sym_start + CP_LEN : sym_start + SYM_LEN]) / np.sqrt(N_FFT)
        
        eq_data = rx_fft[data_idx] / H_est_data
        eq_pilots = rx_fft[pilot_idx] / H_est_pilots
        
        cpe_angle = np.angle(np.mean(eq_pilots))
        all_clean_qpsk.extend(eq_data * np.exp(-1j * cpe_angle))

# ==========================================
# 4. DE-INTERLEAVE & RECONSTRUCT IMAGE
# ==========================================
rx_bits = []
for s in all_clean_qpsk:
    r, i = np.real(s), np.imag(s)
    if r > 0 and i > 0: rx_bits.extend([0,0])
    elif r < 0 and i > 0: rx_bits.extend([0,1])
    elif r < 0 and i < 0: rx_bits.extend([1,1])
    elif r > 0 and i < 0: rx_bits.extend([1,0])

# --- BLOCK DE-INTERLEAVER ---
depth = 32
rx_bits = np.reshape(rx_bits, (depth, -1)).T.flatten().tolist()
# ----------------------------

img_bits = rx_bits[:20000]
img_bytes = np.packbits(img_bits)
img_matrix = img_bytes.reshape((50, 50))

plt.imshow(img_matrix, cmap='gray')
plt.title("Interleaved OFDM Image")
plt.axis('off')
plt.show()