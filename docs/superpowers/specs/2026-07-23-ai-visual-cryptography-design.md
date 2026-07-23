# AI-Enhanced Visual Cryptography for Blockchain Banking — Design

**Date:** 2026-07-23
**Status:** Approved

## Goal

Replace the broken `GAN.py` (shares → landscape, trained on noise, no recovery) with a
faithful generative visual-cryptography system: a GAN produces **two natural-looking
landscape shares** from a signed-transaction QR such that

1. **each share alone reveals nothing** (privacy; robust to AI reconstruction attacks), and
2. **physically stacking the two shares reveals the QR** (human-verifiable, no network needed).

This realizes the abstract: *AI-enhanced visual cryptography for human-verifiable,
privacy-preserving blockchain banking transactions.*

## Decisions (locked)

- **Recovery mechanism:** physical stacking (true VC). Overlay `O = A ⊙ B` (pixel-wise
  multiply, transparency stacking). QR read from `luminance(O)` by thresholding. No decoder
  network — recovery is a human physical operation.
- **Compute:** trained on Google Colab (free T4 GPU). Code is plain TensorFlow.
- **Shares:** full RGB color (multiply still recovers QR via luminance).
- **QRs:** generated on-the-fly during training from `cryptography.py` for generalization.
- **Dataset:** Kaggle `arnaud58/landscape-pictures` (~4k images), resized to 128×128.

## Core mechanism

```
sample two real landscapes  C_a, C_b   ┐
signed-transaction QR (1-ch)  ─────────┤─► Generator G ─┬─► Share A ≈ C_a
noise z                       ─────────┘                └─► Share B ≈ C_b

RECOVER:  O = A ⊙ B   →   QR visible in luminance(O)  →  threshold to decode
```

Where a QR module is **black**, G darkens A *or* B (not both) so the product is dark.
Where **white**, both stay bright so the product is bright. Randomizing which share carries
the darkening is what gives per-share secrecy.

## Losses

| Loss | Purpose | Form |
|---|---|---|
| `L_recon` | overlay reveals QR | `BCE(soft_threshold(lum(A⊙B)), QR)` |
| `L_content` | shares look like their cover landscape | `L1(A, C_a) + L1(B, C_b)` |
| `L_adv` | shares pass as real landscapes | PatchGAN discriminator |
| `L_secrecy` | one share alone leaks nothing | attacker net predicts QR from a single share; G maximizes its error |

Total generator loss: `λ_recon·L_recon + λ_content·L_content + λ_adv·L_adv + λ_sec·L_secrecy`.
**Loss weights are the calibration knobs**; `λ_recon` vs `λ_content` is the beauty-vs-recovery
tension and must be tuned on real hardware.

## Components / files

- **`cryptography.py`** — unchanged. Reused for `sign` + `make_qr`.
- **`vcgan.py`** (new) — models (U-Net generator w/ 2 heads, PatchGAN discriminator, secrecy
  attacker), the 4 losses, `overlay()` op, `train_step`, `build_dataset`, and a `demo()`
  `__main__` self-check that runs a tiny CPU smoke test (2 images, 1 step, asserts shapes +
  loss finite) — verifiable without a GPU.
- **`infer.py`** (new) — load trained generator weights, take a real transaction, emit
  `shareA.png`, `shareB.png`, `overlay.png`; verify the recovered QR decodes and the signature
  checks. The demo figure + human-verifiable proof.
- **`COLAB.md`** (new) — runnable cells: Kaggle download (`kaggle.json`), config, train, save
  `generator.weights.h5`, download.
- **`GAN.py`** — deleted (wrong direction, unrecoverable).

## Data flow

Colab: Kaggle landscapes → resize 128×128 → per step sample `(C_a, C_b)` pairs + generate a
batch of transaction QRs on the fly → train → save `generator.weights.h5`.
Local: `infer.py` loads those weights for real transactions.

## Honest ceilings

- Recovered QR is contrast-based and soft; `infer.py` thresholds + relies on QR error
  correction. If a QR won't scan: shorten payload (truncate hash/sig) or raise `λ_recon`.
  Inherent to true VC.
- Shares look slightly **muted** — the darkening is the cost of real stacking recovery.
- No local GPU (TF ≥ 2.11 has none on native Windows); training happens on Colab. Local runs
  are inference + smoke tests only.

## Test / verification

- `vcgan.py` `demo()`: 1 train step on 2 synthetic images; assert output shapes and finite
  losses. Runs on CPU in seconds.
- `infer.py`: after training, assert `overlay.png` recovered-QR decodes to the payload and the
  signature verifies — the end-to-end human-verifiable claim, checked in code.
