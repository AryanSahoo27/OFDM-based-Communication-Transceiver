import numpy as np
import adi
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
N_FFT, CP_LEN = 128, 32
FS = int(N_FFT * 30000)
SYM_LEN = N_FFT + CP_LEN
TOTAL_FRAMES = 8 

dc_idx = [0]
guard_idx = list(range(1, 11)) + list(range(118, 128))
pilot_idx = [12, 24, 36, 48, 60, 68, 80, 92, 104]
data_idx = [i for i in range(N_FFT) if i not in dc_idx + guard_idx + pilot_idx]

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.rx_lo = 900000000
# MASSIVE buffer to guarantee we catch all 8 frames whole
sdr.rx_buffer_size = 400000 
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 40

np.random.seed(42); sync_sof = np.random.choice([1+0j, -1+0j], len(data_idx))
np.random.seed(43); sync_data = np.random.choice([1+0j, -1+0j], len(data_idx))

local_sof = np.zeros(N_FFT, dtype=complex)
local_sof[data_idx] = sync_sof
local_sof_time = np.fft.ifft(local_sof) * np.sqrt(N_FFT)
local_sof_cp = np.concatenate((local_sof_time[-CP_LEN:], local_sof_time))

print("Listening for 8-Frame MLP Brain...")
for _ in range(5): sdr.rx()
rx_signal = sdr.rx()

# ==========================================
# 2. CFO CORRECTION & SOF ALIGNMENT
# ==========================================
delayed_rx = rx_signal[N_FFT:]
original_rx = rx_signal[:-N_FFT]
f_offset = np.angle(np.mean(np.conjugate(original_rx) * delayed_rx)) * (FS / (2 * np.pi * N_FFT))
rx_signal = rx_signal * np.exp(-1j * 2 * np.pi * f_offset * (np.arange(len(rx_signal)) / FS))

corr = np.abs(np.correlate(rx_signal, local_sof_cp, mode='valid'))
sof_start = np.argmax(corr)
frame_starts = [sof_start + (f * 30 * SYM_LEN) for f in range(TOTAL_FRAMES)]

# ==========================================
# 3. EQUALIZATION & DEMAPPING
# ==========================================
all_clean_qpsk = []
for f_idx, frame_start in enumerate(frame_starts):
    if frame_start + (30 * SYM_LEN) >= len(rx_signal): 
        print(f"Frame {f_idx} cut off! Increase buffer.")
        break
        
    rx_sync_sym = rx_signal[frame_start + CP_LEN : frame_start + SYM_LEN]
    rx_sync_fft = np.fft.fft(rx_sync_sym) / np.sqrt(N_FFT)
    H_est_data = rx_sync_fft[data_idx] / (sync_sof if f_idx == 0 else sync_data)
    H_est_pilots = rx_sync_fft[pilot_idx] / np.ones(len(pilot_idx))

    for i in range(2, 30):
        sym_start = frame_start + (i * SYM_LEN)
        rx_fft = np.fft.fft(rx_signal[sym_start + CP_LEN : sym_start + SYM_LEN]) / np.sqrt(N_FFT)
        eq_data = rx_fft[data_idx] / H_est_data
        eq_pilots = rx_fft[pilot_idx] / H_est_pilots
        cpe_angle = np.angle(np.mean(eq_pilots))
        all_clean_qpsk.extend(eq_data * np.exp(-1j * cpe_angle))

rx_bits = []
for s in all_clean_qpsk:
    r, i = np.real(s), np.imag(s)
    if r > 0 and i > 0: rx_bits.extend([0,0])
    elif r < 0 and i > 0: rx_bits.extend([0,1])
    elif r < 0 and i < 0: rx_bits.extend([1,1])
    elif r > 0 and i < 0: rx_bits.extend([1,0])

# ==========================================
# 4. DE-INTERLEAVE & RECONSTRUCT BRAIN
# ==========================================
# De-interleave with depth 64 matching the TX
rx_bits = np.reshape(rx_bits, (64, -1)).T.flatten().tolist()

# SAFETY CHECK: Ensure we caught the full brain before reshaping
if len(rx_bits) < 38720:
    print(f"\n[ERROR] Buffer Cutoff! Only caught {len(rx_bits)} out of 38720 bits.")
    print("The radio grabbed a partial frame. Run the RX script one more time!")
    exit()

# 1210 floats * 32 bits = 38,720 bits total for the matrices
payload_bytes = np.packbits(rx_bits[:38720]).tobytes()
rx_floats = np.frombuffer(payload_bytes, dtype=np.float32)

# Slice the 1D float array exactly back into the 4 matrices
W1_rx = rx_floats[0:1024].reshape(64, 16)
b1_rx = rx_floats[1024:1040]
W2_rx = rx_floats[1040:1200].reshape(16, 10)
b2_rx = rx_floats[1200:1210]

print("2-Layer Brain Reassembled Successfully!\n")
# Slice the 1D float array exactly back into the 4 matrices
W1_rx = rx_floats[0:1024].reshape(64, 16)
b1_rx = rx_floats[1024:1040]
W2_rx = rx_floats[1040:1200].reshape(16, 10)
b2_rx = rx_floats[1200:1210]

# --- THE FIX: SANITIZE MUTATED RF WEIGHTS ---
def heal_brain(matrix):
    # Convert NaNs and Infinities to 0.0
    clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip mutated giant weights back to a normal NN range
    return np.clip(clean, -2.0, 2.0)

W1_rx = heal_brain(W1_rx)
b1_rx = heal_brain(b1_rx)
W2_rx = heal_brain(W2_rx)
b2_rx = heal_brain(b2_rx)
# --------------------------------------------

print("2-Layer Brain Reassembled and Healed Successfully!\n")
# ==========================================
# 5. LIVE PREDICTION LOOP
# ==========================================
# Reset random seed so it picks new digits every time the loop runs
np.random.seed() 
digits = load_digits()

# Test the brain on 5 different random images
for test_num in range(1, 6):
    idx = np.random.randint(0, len(digits.target))
    test_img = digits.data[idx] / 16.0
    true_label = digits.target[idx]

    # Forward Propagation Math: Z1 -> ReLU -> Z2
    Z1 = np.dot(test_img, W1_rx) + b1_rx
    A1 = np.maximum(0, Z1) # ReLU activation
    Z2 = np.dot(A1, W2_rx) + b2_rx
    prediction = np.argmax(Z2)

    print(f"Test {test_num}/5 | True: {true_label} | SDR MLP Predicts: {prediction}")

    # Plot the final test as visual proof
    if test_num == 5:
        plt.imshow(test_img.reshape(8, 8), cmap='gray')
        plt.title(f"True: {true_label} | SDR MLP Predicts: {prediction}")
        plt.axis('off')
        plt.show()