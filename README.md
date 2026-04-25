# OFDM-based Communication Transceiver

This repository contains my Exploratory Project on designing a custom Python-based physical layer (PHY) for an OFDM transceiver using the ADALM-Pluto Software Defined Radio (SDR). It demonstrates baseband signal processing, burst error protection, and Over-The-Air (OTA) transmission of neural network weights.

## Project Structure

The codebase is divided into three progressive experiments:

* **`constellation/`**: The core OFDM pipeline. Demonstrates QPSK transmission and a 4-step DSP healing process on the receiver (Carrier Frequency Offset correction, Frame Synchronization, Zero-Forcing Equalization, and Common Phase Error tracking).
* **`img_explo/`**: Visual payload transmission. Converts a 50x50 image into a bitstream, applies a block interleaver to protect against RF burst errors, transmits it, and reconstructs the image on the receiver.
* **`mnist/`**: Over-The-Air Edge AI. Trains a Multi-Layer Perceptron (MLP) on the MNIST dataset, serializes the synaptic weights into bits, transmits the "brain" over the OFDM link, and reconstructs the neural network on the receiver for live digit inference.

## Hardware & OS Requirements
* ADALM-Pluto SDR (PlutoSDR)
* Appropriate Antennas
* **Operating System:** Linux (Required for the underlying hardware drivers)

## System & Software Dependencies

Because Python communicates with the SDR hardware, you must first install the Analog Devices C-libraries on your Linux machine before installing the Python packages:

1. **System Libraries:**
   * `libiio`
   * `libad9361-iio`

2. **Python Dependencies:**
   Once the system drivers are configured, install the required Python libraries:
   ```bash
   pip install numpy matplotlib pyadi-iio scikit-learn opencv-python