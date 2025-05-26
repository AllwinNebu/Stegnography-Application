
# StegApp: Steganography Pro

**StegApp** is a powerful and user-friendly steganography application that allows you to hide and extract **Text**, **Audio**, and **Image** data inside carrier images. It supports **hybrid encryption** (AES + Blowfish), **GPU acceleration** (CUDA and CuPy), and features a **modern, responsive GUI** built with **CustomTkinter**.

---

## ✨ Features

- **Hide** and **Extract**:
  - Text data
  - Audio files (.wav, .mp3 converted internally)
  - Images (.png, .jpg)
- **Hybrid Encryption**:
  - AES (Advanced Encryption Standard)
  - Blowfish Cipher
  - Key derivation using SHA-256
- **High Performance**:
  - CUDA acceleration for pixel manipulation and encryption (if GPU available)
  - Optimized multithreaded CPU fallback
- **Automatic Optimization**:
  - Image resizing if needed
  - MP3 to WAV conversion
- **Robust Error Handling**:
  - Friendly error messages
  - Validation of extracted data
- **Modern GUI**:
  - CustomTkinter dark mode UI
  - Smooth progress updates
  - Responsive and clean design
- **Large Data Support**:
  - Process large images and files efficiently
  - Memory-optimized algorithms

---

## 🛠 Installation

### Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```text
customtkinter
numpy
pillow
pycryptodome
opencv-python
soundfile
scipy
psutil
cupy-cuda12x  # Optional, only if you have a CUDA-enabled GPU
pycuda         # Optional, for full GPU support
aiofiles
```

**Note**: 
- If you don't have a CUDA-capable GPU, **CuPy** and **PyCUDA** are optional.
- StegApp automatically falls back to CPU processing if GPU is not available.

---

 🚀 Running StegApp

```bash
python StegApp.py
```

The application will launch a GUI window.

---

 📦 How to Use

### Hiding Data:
1. **Select a Carrier Image** (PNG/JPG).
2. **Choose Data Type** (Text, Audio, Image).
3. **Provide the Data**:
   - Text: Type your secret text.
   - Audio: Upload `.wav` or `.mp3` file.
   - Image: Upload another image.
4. **Enter Encryption Key**.
5. Click **"Hide Data"**.
6. A new **Stego Image** will be created (`stego_filename.png`).

Extracting Data:
1. **Select Stego Image**.
2. **Enter Decryption Key**.
3. Click **"Extract Data"**.
4. Extracted text, audio, or image will be shown/saved automatically.

---
 ⚡ Performance Notes

- On a machine with a CUDA-capable GPU, **CUDA kernels** will accelerate:
  - Pixel bit manipulation
  - Encryption/decryption
- On CPU-only systems, optimized **multi-threaded** processing ensures good performance.

---

 🧠 Technical Details

- **Encryption**:
  - Hybrid: Alternate AES and Blowfish for every data block.
  - AES ECB Mode and Blowfish ECB Mode.
- **Steganography**:
  - LSB (Least Significant Bit) embedding into RGB pixels.
  - Data is prefixed with type and size headers.
- **GPU Kernels**:
  - `process_pixels`: Embeds bits into pixels.
  - `extract_bits`: Retrieves hidden bits.
  - `crypto_kernel`: Basic XOR for accelerated encryption (optional fallback).

---
created by 
      -- Allwin Nebu
      -- Alvin Boby Mathew
      -- Ajay Dileep
      -- Mathew Somy
      
---
Further information about the software is available at the document folder 
