# 🔐 AI-Enhanced Visual Cryptography for Human-Verifiable Blockchain Banking

An end-to-end cryptographic + generative-AI system that secures financial
transactions and lets a human **visually verify** them. It combines Elliptic
Curve Digital Signatures (ECDSA), 2-of-2 Visual Cryptography (VC), and a
Conditional GAN that turns a signed transaction QR into **two natural-looking
landscape "shares"** — innocent on their own, but revealing the QR when stacked.

---

## 📌 Overview

Blockchain banking is transparent but leaks privacy and offers little that a
non-technical user can independently verify. This project adds a human-verifiable
layer: security you can check with your own eyes by overlaying two pictures.

1. **Digital signature & QR encoding** — the transaction is signed with ECDSA
   (P-256) and its hash + signature are encoded into a QR code.
2. **AI-generated visual shares** — a Conditional GAN, conditioned on the QR and
   two real landscape cover images, outputs **Share A** and **Share B**. Each
   looks like an ordinary landscape and reveals nothing about the transaction.
3. **Human-verifiable recovery** — physically stacking the two shares
   (`overlay = A × B`, transparency multiply) reveals the QR in the combined
   image's brightness. No decoder network is needed — just the overlay.

> The generative direction is **QR → shares** (the secret is hidden *into* the
> shares), and recovery is a real physical stacking operation — true visual
> cryptography, not a neural decode.

---

## 🏗 System Architecture

```
[ Transaction Data ]
         │
         ▼
  [ ECDSA Sign (P-256) ] ──► [ SHA-256 Digest ]
         │
         ▼
  [ QR Code (secret) ]        [ Real Landscape Covers  C_a , C_b ]
         └──────────────┬─────────────────┘
                        ▼
              [ cGAN Generator (U-Net) ]
                 ┌──────┴───────┐
                 ▼              ▼
            [ Share A ]     [ Share B ]     ← look like landscapes
                 └──────┬───────┘
                        ▼  physical stack (A × B)
                  [ Overlay ] ──► QR revealed ──► verify signature
```

---

## 🛠 Features

- **Authenticity & non-repudiation** — PyCryptodome, P-256 ECC, FIPS-186-3 DSS.
- **Human-verifiable security** — the transaction is confirmed by stacking two
  images, not by trusting a black box.
- **Privacy / information hiding** — each share alone is a plain landscape and
  reveals nothing about the payload.
- **Robust to AI reconstruction attacks** — an adversarial *secrecy* network is
  trained to recover the QR from a single share; the generator is optimized to
  defeat it.
- **Deep-learning core** — TensorFlow/Keras: U-Net generator (skip connections),
  PatchGAN discriminator, plus reconstruction, content, adversarial, and secrecy
  losses.

---

## 📦 Prerequisites & Installation

Python 3.9+:

```bash
pip install tensorflow numpy pillow qrcode pycryptodome matplotlib opencv-python
```

---

## 💻 Usage

### 1. Verify the pipeline locally (no GPU, seconds)

```bash
python cryptography.py   # sign → QR → classic VC shares (Steps 2–4 self-check)
python vcgan.py          # GAN smoke test: one train step, asserts shapes/losses
```

### 2. Train the GAN (needs a GPU — use Google Colab)

Local Windows has no usable GPU for TensorFlow, so training runs on a free Colab
T4. Follow **`COLAB.md`**: download the Kaggle `landscape-pictures` dataset,
train, and download `generator.weights.h5`.

### 3. Generate + verify shares for a real transaction

```bash
python infer.py cover1.jpg cover2.jpg
```

Outputs `shareA.png`, `shareB.png`, `overlay.png`, `recovered_qr.png`, then
decodes the overlay and confirms the ECDSA signature — the human-verifiable proof.

---

## 📂 Project Files

| File | Role |
|------|------|
| `cryptography.py` | ECDSA signing, QR encoding, classic 2-of-2 VC baseline |
| `vcgan.py` | The AI model: U-Net generator, discriminator, secrecy attacker, losses, training |
| `infer.py` | Generate + verify shares for a real signed transaction |
| `COLAB.md` | Step-by-step GPU training on Google Colab |
| `docs/superpowers/specs/` | Design spec |

---

## ⚙️ Tuning

The one knob that matters lives at the top of `vcgan.py`:

- Raise `LAMBDA_RECON` if the recovered QR won't scan.
- Raise `LAMBDA_CONTENT` if the shares look too dark/muted.

This beauty-vs-recovery tension is inherent to true visual cryptography.

---

## 📑 References

- **Visual Cryptography** — Naor, M. & Shamir, A. (1994). *Visual Cryptography.*
  Advances in Cryptology – EUROCRYPT '94.
- **Conditional GANs (pix2pix)** — Isola, P., Zhu, J.-Y., Zhou, T. & Efros, A. A.
  (2017). *Image-to-Image Translation with Conditional Adversarial Networks.*

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
