# 🔐 AI-Enhanced Visual Cryptography for Human-Verifiable Blockchain Banking

An end-to-end cryptographic + generative AI framework that secures blockchain banking transactions while enabling **human-verifiable authentication**. The system combines **ECDSA**, **2-of-2 Visual Cryptography**, and a **Conditional GAN (cGAN)** to transform a signed transaction QR code into **two natural-looking landscape images** that reveal the transaction only when overlaid.

---

# 📌 Overview

Traditional blockchain banking provides transparency and immutability but offers limited privacy and little that non-technical users can independently verify.

This project introduces a **human-centric security layer**:

1. **Sign the transaction** using ECDSA (P-256).
2. **Generate a QR code** containing the signed transaction.
3. **Use a Conditional GAN** to hide the QR inside two realistic landscape images.
4. **Overlay the two shares** to visually recover the QR code.
5. **Verify the digital signature** to confirm authenticity.

> Each share appears to be an ordinary landscape image. Individually they reveal nothing; together they reveal the hidden transaction QR.

---

# 🏗️ System Architecture

```text
                 Transaction Data
                        │
                        ▼
              ECDSA Signature (P-256)
                        │
                        ▼
                  SHA-256 Hash
                        │
                        ▼
                 Transaction QR Code
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
Landscape Cover A                Landscape Cover B
        └───────────────┬────────────────┘
                        ▼
          Conditional GAN (U-Net Generator)
                 ┌──────────┴──────────┐
                 ▼                     ▼
            Share A               Share B
                 └──────────┬──────────┘
                            ▼
                 Physical Overlay (A × B)
                            ▼
                  Hidden QR Recovered
                            ▼
                 Verify ECDSA Signature
```

---

# ✨ Features

- Human-verifiable blockchain transaction validation
- Privacy-preserving visual cryptography
- ECDSA digital signatures (P-256)
- SHA-256 transaction hashing
- QR-based transaction encoding
- Conditional GAN for realistic visual shares
- U-Net Generator
- PatchGAN Discriminator
- Adversarial secrecy network against reconstruction attacks
- TensorFlow/Keras implementation
- End-to-end transaction verification

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| AI Framework | TensorFlow, Keras |
| Computer Vision | OpenCV, Pillow |
| Deep Learning | Conditional GAN, U-Net, PatchGAN |
| Cryptography | PyCryptodome, ECDSA, SHA-256 |
| QR Processing | qrcode |
| Visualization | Matplotlib |
| Numerical Computing | NumPy |

---

# 📦 Installation

```bash
pip install tensorflow numpy pillow qrcode pycryptodome matplotlib opencv-python
```

---

# 🚀 Usage

## Step 1 — Verify Cryptography Pipeline

```bash
python cryptography.py
```

Performs:

- Transaction signing
- SHA-256 hashing
- QR generation
- Classic Visual Cryptography share generation

---

## Step 2 — Train the AI Model

```bash
python vcgan.py
```

or train on Google Colab using the provided notebook.

---

## Step 3 — Generate Secure Shares

```bash
python infer.py cover1.jpg cover2.jpg
```

Outputs:

```
shareA.png
shareB.png
overlay.png
recovered_qr.png
```

The recovered QR is decoded and the ECDSA signature is verified.

---

# 📁 Project Structure

```text
.
├── cryptography.py
├── vcgan.py
├── infer.py
├── COLAB.md
├── docs/
│   └── superpowers/
│       └── specs/
├── outputs/
├── models/
└── README.md
```

---

# ⚙️ Configuration

Inside `vcgan.py`:

- **LAMBDA_RECON** → Increase if QR reconstruction quality is poor.
- **LAMBDA_CONTENT** → Increase if generated shares lose visual quality.

Balancing these parameters controls the trade-off between image realism and QR recoverability.

---

# 📚 References

1. Naor, M., & Shamir, A. (1994). *Visual Cryptography*. EUROCRYPT '94.
2. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)*.

---

# 📄 License

This project is released under the **MIT License**.
