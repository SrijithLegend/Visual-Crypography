"""Steps 2-3 — Digital Signature (ECC / P-256) + QR encoding.

sign(message)              -> signature bytes, bound to the message
verify(message, signature) -> True if untampered
make_qr(tx, signature)     -> saves transaction_qr.png, returns the QR module
                              matrix (bool, True = black) for Step 4

The keypair is generated once and saved to signing_key.pem so signatures
stay verifiable across runs. Run this file directly for a self-check plus a
Step 4 (visual-crypto) demo.
"""
import json
from pathlib import Path

import numpy as np
import qrcode
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256

_HERE = Path(__file__).parent
KEY_FILE = _HERE / "signing_key.pem"
QR_FILE = _HERE / "transaction_qr.png"
SHARE_A = _HERE / "shareA.png"
SHARE_B = _HERE / "shareB.png"
RECON_FILE = _HERE / "reconstructed_qr.png"


def _load_key():
    if KEY_FILE.exists():
        return ECC.import_key(KEY_FILE.read_text())
    key = ECC.generate(curve="p256")
    KEY_FILE.write_text(key.export_key(format="PEM"))
    return key


_key = _load_key()


def sign(message: bytes) -> bytes:
    """Sign the transaction bytes with the private key."""
    digest = SHA256.new(message)
    return DSS.new(_key, "fips-186-3").sign(digest)


def verify(message: bytes, signature: bytes) -> bool:
    """Check the signature against the message using the public key."""
    digest = SHA256.new(message)
    try:
        DSS.new(_key.public_key(), "fips-186-3").verify(digest, signature)
        return True
    except ValueError:
        return False


def make_qr(tx, signature: bytes, path: Path = QR_FILE) -> np.ndarray:
    """Step 3 — encode txid+amount+hash+signature into a QR.

    Saves a scannable PNG at `path` and returns the QR module matrix
    (bool array, True = black module) for the visual-crypto step.
    """
    message = tx.model_dump_json().encode()
    payload = json.dumps(
        {
            "txid": tx.txid,
            "amount": tx.amount,
            "hash": SHA256.new(message).hexdigest(),
            "signature": signature.hex(),
        }
    )
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(path)
    return np.array(qr.get_matrix(), dtype=bool)  # True = black module


def split_shares(qr_matrix: np.ndarray) -> None:
    """Step 4 — 2-of-2 visual cryptography.

    Splits the QR into shareA/shareB (each reveals nothing alone) and saves
    their overlay to reconstructed_qr.png as proof they recombine.
    """
    from PIL import Image

    height, width = qr_matrix.shape
    share1 = np.zeros((height * 2, width * 2), dtype=bool)  # True = black subpixel
    share2 = np.zeros((height * 2, width * 2), dtype=bool)

    block_a = np.array([[True, False], [False, True]])
    block_b = np.array([[False, True], [True, False]])

    for r in range(height):
        for c in range(width):
            is_black = qr_matrix[r, c]
            sub = block_a if np.random.rand() > 0.5 else block_b
            share1[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = sub
            share2[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = ~sub if is_black else sub

    overlay = np.bitwise_or(share1, share2)  # stacking the shares
    Image.fromarray(~share1).save(SHARE_A)   # ~ : True(black) -> 0(black pixel)
    Image.fromarray(~share2).save(SHARE_B)
    Image.fromarray(~overlay).save(RECON_FILE)


if __name__ == "__main__":
    # --- Step 2 self-check: valid signature verifies, tampered one does not ---
    msg = b'{"sender":"Alice","receiver":"Bob","amount":10000}'
    sig = sign(msg)
    assert verify(msg, sig), "valid signature must verify"
    assert not verify(msg + b"x", sig), "tampered message must fail"
    print("Step 2 OK - signature verifies, tamper rejected")

    # --- Step 3 self-check: QR PNG is written ---
    class _Tx:
        txid, amount = "TXDEMO1", 10000.0
        def model_dump_json(self):
            return '{"sender":"Alice","receiver":"Bob","amount":10000}'

    matrix = make_qr(_Tx(), sig)
    assert QR_FILE.exists() and matrix.dtype == bool, "QR PNG + matrix expected"
    print(f"Step 3 OK - wrote {QR_FILE.name} ({matrix.shape[0]}x{matrix.shape[1]} modules)")

    # --- Step 4 self-check: shares reveal nothing alone, overlay restores QR ---
    split_shares(matrix)
    assert SHARE_A.exists() and SHARE_B.exists() and RECON_FILE.exists()
    print(f"Step 4 OK - wrote {SHARE_A.name}, {SHARE_B.name}, {RECON_FILE.name}")
