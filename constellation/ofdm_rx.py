import numpy as np
import adi
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
N_FFT = 128
SC_SPACING = 60000
FS = int(N_FFT * SC_SPACING) # 7.68 MHz
CP_LEN = 32
SYM_LEN = N_FFT + CP_LEN

all_idx = np.arange(N_FFT)
dc_idx = [0]
guard_idx = list(range(1, 11)) + list(range(118, 128))
pilot_idx = [12, 24, 36, 48, 60, 68, 80, 92, 104] 
data_idx = [i for i in all_idx if i not in dc_idx + guard_idx + pilot_idx]

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.rx_lo = 3000000000
sdr.rx_buffer_size = SYM_LEN * 30 * 5 
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 40

np.random.seed(42)
sync_data = np.random.choice([1+0j, -1+0j], len(data_idx))

local_sync = np.zeros(N_FFT, dtype=complex)
local_sync[data_idx] = sync_data
local_sync[pilot_idx] = np.ones(len(pilot_idx))
local_sync_time = np.fft.ifft(local_sync) * np.sqrt(N_FFT)
local_sync_cp = np.concatenate((local_sync_time[-CP_LEN:], local_sync_time))

# ==========================================
# 2. CAPTURE & CFO CORRECTION
# ==========================================
for _ in range(5): sdr.rx() 
rx_signal = sdr.rx()

# SAVE A RAW COPY FOR THE "BEFORE" PLOT
raw_rx_signal = rx_signal.copy()

delayed_rx = rx_signal[N_FFT:]
original_rx = rx_signal[:-N_FFT]

cp_corr = np.conjugate(original_rx) * delayed_rx
phase_drift = np.angle(np.mean(cp_corr))

f_offset = phase_drift * (FS / (2 * np.pi * N_FFT))
print(f"Correcting CFO: {f_offset:.2f} Hz")

t = np.arange(len(rx_signal)) / FS
rx_signal = rx_signal * np.exp(-1j * 2 * np.pi * f_offset * t)

# ==========================================
# 3. FRAME SYNCHRONIZATION
# ==========================================
corr = np.abs(np.correlate(rx_signal, local_sync_cp, mode='valid'))
frame_start = np.argmax(corr)
print(f"Frame locked at index: {frame_start}")

# ==========================================
# 4. DSP HEALING PIPELINE
# ==========================================
if frame_start + (3 * SYM_LEN) < len(rx_signal):
    
    # --- A. Channel Estimation ---
    rx_sync_sym = rx_signal[frame_start + CP_LEN : frame_start + SYM_LEN]
    rx_sync_fft = np.fft.fft(rx_sync_sym) / np.sqrt(N_FFT)
    H_est_data = rx_sync_fft[data_idx] / sync_data 
    H_est_pilots = rx_sync_fft[pilot_idx] / np.ones(len(pilot_idx))

    # --- B. Extract Raw vs Corrected Payloads ---
    data_start = frame_start + (2 * SYM_LEN) 
    
    # 1. The Raw Donut (For Plotting)
    raw_data_sym = raw_rx_signal[data_start + CP_LEN : data_start + SYM_LEN]
    raw_data_fft = np.fft.fft(raw_data_sym) / np.sqrt(N_FFT)
    raw_plot_data = raw_data_fft[data_idx] / np.max(np.abs(raw_data_fft[data_idx])) # Normalized for visual
    
    # 2. The CFO Corrected Data
    rx_data_sym = rx_signal[data_start + CP_LEN : data_start + SYM_LEN]
    rx_data_fft = np.fft.fft(rx_data_sym) / np.sqrt(N_FFT)
    cfo_plot_data = rx_data_fft[data_idx] / np.max(np.abs(rx_data_fft[data_idx])) # Normalized for visual
    
    # --- C. Zero-Forcing Equalization ---
    rx_equalized_data = rx_data_fft[data_idx] / H_est_data
    rx_equalized_pilots = rx_data_fft[pilot_idx] / H_est_pilots
    
    # --- D. CPE Correction ---
    cpe_angle = np.angle(np.mean(rx_equalized_pilots))
    print(f"Correcting Phase Error: {np.degrees(cpe_angle):.2f} degrees")
    rx_final_data = rx_equalized_data * np.exp(-1j * cpe_angle)
    
    # ==========================================
    # 5. VISUALIZATION (THE 4-STEP HEALING)
    # ==========================================
    plt.figure(figsize=(20, 5))
    
    # 1. Raw Radio Wave
    plt.subplot(1, 4, 1)
    plt.scatter(np.real(raw_plot_data), np.imag(raw_plot_data), alpha=0.6, color='gray')
    plt.title("1. Raw (Spinning & Echoing)")
    plt.grid(True); plt.axis('equal'); plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
    
    # 2. CFO Corrected
    plt.subplot(1, 4, 2)
    plt.scatter(np.real(cfo_plot_data), np.imag(cfo_plot_data), alpha=0.6, color='orange')
    plt.title("2. CFO Stopped (Still Echoing)")
    plt.grid(True); plt.axis('equal'); plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
    
    # 3. Equalized
    plt.subplot(1, 4, 3)
    plt.scatter(np.real(rx_equalized_data), np.imag(rx_equalized_data), alpha=0.6, color='red')
    plt.title("3. Equalized (But Rotated)")
    plt.grid(True); plt.axis('equal'); plt.xlim(-2, 2); plt.ylim(-2, 2)
    
    # 4. Final Healed QPSK
    plt.subplot(1, 4, 4)
    plt.scatter(np.real(rx_final_data), np.imag(rx_final_data), alpha=0.8, color='blue')
    plt.title("4. Final Phase-Locked QPSK")
    plt.grid(True); plt.axis('equal'); plt.xlim(-2, 2); plt.ylim(-2, 2)
    
    plt.suptitle("OFDM Physical Layer DSP Healing Pipeline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

else:
    print("Buffer too small or sync failed.")