import numpy as np
import adi

N_FFT, CP_LEN = 128, 32
FS = int(N_FFT * 30000)
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

print("Loading 2-Layer Brain...")
with open("mlp_bits.bin", "rb") as f:
    bits = np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8)).tolist()

# Math: 28 symbols * 97 carriers * 2 bits = 5432 bits per frame
# Dynamically match the physical subcarrier lanes
FRAME_CAPACITY = 28 * len(data_idx) * 2
TOTAL_FRAMES = int(np.ceil(len(bits) / FRAME_CAPACITY))
padded_bits = bits + [0] * ((TOTAL_FRAMES * FRAME_CAPACITY) - len(bits))

# Matrix Interleaver (Depth 64 for strong burst protection)
padded_bits = np.reshape(padded_bits, (-1, 64)).T.flatten().tolist()

qpsk_dict = {(0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j}
payload_qpsk = [qpsk_dict[(padded_bits[i], padded_bits[i+1])] for i in range(0, len(padded_bits), 2)]

tx_signal = []
np.random.seed(42); sync_sof = np.random.choice([1+0j, -1+0j], len(data_idx))  
np.random.seed(43); sync_data = np.random.choice([1+0j, -1+0j], len(data_idx)) 

for f in range(TOTAL_FRAMES):
    tx_signal.extend(create_ofdm_symbol(sync_sof if f == 0 else sync_data, np.ones(len(pilot_idx)))) 
    tx_signal.extend(create_ofdm_symbol(sync_data, np.ones(len(pilot_idx)))) 
    
    chunk = payload_qpsk[f*(28*len(data_idx)) : (f+1)*(28*len(data_idx))]
    for i in range(28):
        tx_signal.extend(create_ofdm_symbol(chunk[i*len(data_idx) : (i+1)*len(data_idx)], np.ones(len(pilot_idx))))

tx_signal = np.array(tx_signal) * (2**14 * 0.4 / np.max(np.abs(tx_signal)))

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = FS
sdr.tx_lo = 900000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -10

print(f"Broadcasting {TOTAL_FRAMES}-frame MLP Brain...")
sdr.tx(tx_signal)

input("Press Enter to stop transmitting...")
sdr.tx_destroy_buffer()
# --------------------------------------------