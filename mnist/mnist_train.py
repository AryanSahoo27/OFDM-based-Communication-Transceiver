import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DYNAMIC TRAINING (No Random Seeds)
# ==========================================
print("Loading digits dataset...")
digits = load_digits()
X, y = digits.data / 16.0, digits.target

# Split data dynamically every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training 2-Layer Neural Network (64 -> 16 -> 10)...")
# 1 hidden layer with 16 neurons. Default activation is ReLU.
clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000)
clf.fit(X_train, y_train)

print(f"Model Accuracy: {clf.score(X_test, y_test)*100:.2f}%")

# ==========================================
# 2. SERIALIZATION OF 4 MATRICES
# ==========================================
# W1: (64, 16), b1: (16,), W2: (16, 10), b2: (10,)
W1 = clf.coefs_[0].astype(np.float32)
b1 = clf.intercepts_[0].astype(np.float32)
W2 = clf.coefs_[1].astype(np.float32)
b2 = clf.intercepts_[1].astype(np.float32)

# Pack everything sequentially into bytes
payload_bytes = W1.tobytes() + b1.tobytes() + W2.tobytes() + b2.tobytes()
bitstream = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))

print(f"Total Floats: {len(payload_bytes)//4}")
print(f"Total Bits for SDR: {len(bitstream)}")

with open("mlp_bits.bin", "wb") as f:
    f.write(np.packbits(bitstream))
print("Brain exported to mlp_bits.bin")