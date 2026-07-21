"""Step 2 — Digital Signature (ECC / P-256).

sign(message)  -> signature bytes, bound to the message
verify(message, signature) -> True if untampered

The keypair is generated once and saved to signing_key.pem so signatures
stay verifiable across runs. Run this file directly for a self-check plus a
Step 3/4 (QR + visual-crypto) demo.
"""
from pathlib import Path

from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256

KEY_FILE = Path(__file__).parent / "signing_key.pem"


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


if __name__ == "__main__":
    # --- self-check: a valid signature verifies, a tampered message does not ---
    msg = b'{"sender":"Alice","receiver":"Bob","amount":10000}'
    sig = sign(msg)
    assert verify(msg, sig), "valid signature must verify"
    assert not verify(msg + b"x", sig), "tampered message must fail"
    print("Step 2 OK — signature verifies, tamper rejected")

    # --- Step 3/4 demo: QR of the signature, split into visual-crypto shares ---
    import numpy as np
    from PIL import Image
    import qrcode

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    qr.add_data(sig)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert("1")
    qr_array = np.array(qr_img)
    height, width = qr_array.shape

    share1 = np.zeros((height * 2, width * 2), dtype=bool)
    share2 = np.zeros((height * 2, width * 2), dtype=bool)

    block_white_0 = np.array([[True, False], [False, True]])
    block_white_1 = np.array([[False, True], [True, False]])

    for r in range(height):
        for c in range(width):
            is_black = not qr_array[r, c]
            rand_choice = np.random.rand() > 0.5
            if is_black:
                s1_sub = block_white_0 if rand_choice else block_white_1
                s2_sub = ~s1_sub
            else:
                s1_sub = block_white_0 if rand_choice else block_white_1
                s2_sub = s1_sub.copy()
            share1[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = s1_sub
            share2[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = s2_sub

    Image.fromarray(~share1).save("share1.png")
    Image.fromarray(~share2).save("share2.png")
    overlay = np.bitwise_or(share1, share2)
    Image.fromarray(~overlay).save("reconstructed_qr.png")
    print("Step 3/4 demo — wrote share1.png, share2.png, reconstructed_qr.png")
