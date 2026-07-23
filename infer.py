"""Generate + verify visual-crypto shares for a real signed transaction.

Loads the trained generator, takes two real landscape covers, and emits:
  shareA.png, shareB.png   - each an innocent-looking landscape
  overlay.png              - the physical stack (A * B), QR readable by eye
  recovered_qr.png         - thresholded overlay luminance

Then proves the human-verifiable claim in code: the recovered QR decodes to the
transaction payload and the ECC signature verifies.

Usage:  python infer.py cover1.jpg cover2.jpg
(covers default to the first two images found under ./landscapes if omitted)
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

import cryptography as vc          # reuse sign / make_qr / verify from Step 2-3
import vcgan

_HERE = Path(__file__).parent
IMG = vcgan.IMG


def _load_cover(path):
    img = Image.open(path).convert("RGB").resize((IMG, IMG))
    return np.asarray(img, np.float32) / 255.0


def _signed_qr_target():
    """Build the real signed-transaction QR and its 128x128 target + payload."""
    class _Tx:
        txid, amount = "TXDEMO1", 10000.0
        def model_dump_json(self):
            return '{"sender":"Alice","receiver":"Bob","amount":10000}'

    tx = _Tx()
    message = tx.model_dump_json().encode()
    sig = vc.sign(message)
    payload = json.dumps({
        "txid": tx.txid,
        "amount": tx.amount,
        "hash": vc.SHA256.new(message).hexdigest(),
        "signature": sig.hex(),
    })
    return vcgan.qr_target(payload), payload, message, sig


def _default_covers():
    for d in ("landscapes", "."):
        imgs = sorted(p for p in Path(d).rglob("*")
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                      and "share" not in p.name and "overlay" not in p.name
                      and "recovered" not in p.name and "reconstructed" not in p.name)
        if len(imgs) >= 2:
            return str(imgs[0]), str(imgs[1])
    raise FileNotFoundError("need two cover images; pass them as args or add a ./landscapes folder")


def _decode_qr(gray_uint8):
    """Best-effort QR decode; returns payload string or None."""
    try:
        import cv2
    except ImportError:
        return None
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(gray_uint8)
    return data or None


def main(cover_a=None, cover_b=None):
    if not vcgan.WEIGHTS_FILE.exists():
        sys.exit(f"no trained weights at {vcgan.WEIGHTS_FILE} - train on Colab first (see COLAB.md)")
    if cover_a is None:
        cover_a, cover_b = _default_covers()

    G = vcgan.build_generator()
    G.load_weights(vcgan.WEIGHTS_FILE)

    qr, payload, message, sig = _signed_qr_target()
    ca = _load_cover(cover_a)[None]
    cb = _load_cover(cover_b)[None]
    z = tf.random.normal((1, IMG, IMG, 1))

    out = G([ca, cb, qr[None], z], training=False)
    A, B = vcgan.split_ab(out)
    ov = vcgan.overlay(A, B)
    lum = vcgan.luminance(ov)[0, ..., 0].numpy()

    def save(arr01, path):
        Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)

    save(A[0].numpy(), _HERE / "shareA.png")
    save(B[0].numpy(), _HERE / "shareB.png")
    save(ov[0].numpy(), _HERE / "overlay.png")

    # threshold overlay luminance -> binary recovered QR (Otsu-ish midpoint)
    thresh = (lum > (lum.min() + lum.max()) / 2).astype(np.uint8) * 255
    Image.fromarray(thresh).save(_HERE / "recovered_qr.png")

    print("wrote shareA.png, shareB.png, overlay.png, recovered_qr.png")

    decoded = _decode_qr(thresh)
    if decoded is None:
        print("QR decode skipped (install opencv-python to auto-verify the overlay)")
    elif decoded == payload:
        assert vc.verify(message, sig), "signature failed"
        print("VERIFIED - overlay decoded to the transaction AND signature checks")
    else:
        print("overlay did not decode cleanly - raise LAMBDA_RECON or shorten the payload "
              "(see design doc 'Honest ceilings')")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args[:2]) if args else main()
