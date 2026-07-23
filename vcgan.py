"""AI-enhanced visual cryptography GAN — true physical-stacking VC.

Generator turns two real landscape covers + a transaction QR into two color
shares A, B such that:
  - each share looks like its cover landscape (privacy / hides the secret),
  - the physical overlay  O = A * B  reveals the QR in luminance (human-verifiable),
  - a single share alone leaks nothing (secrecy attacker enforces it).

Recovery is a physical multiply — no network needed to read the QR. That is the
"human-verifiable overlay" claim from the paper.

Train on Colab GPU (see COLAB.md). Locally, run `python vcgan.py` for a CPU
smoke test of the whole pipeline.
"""
import json
import random
from pathlib import Path

import numpy as np
import qrcode
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG = 128
_HERE = Path(__file__).parent
WEIGHTS_FILE = _HERE / "generator.weights.h5"

# --- loss weights: THE calibration knobs (recon vs beauty is the core tension) ---
LAMBDA_RECON = 12.0
LAMBDA_CONTENT = 8.0
LAMBDA_ADV = 1.0
LAMBDA_SECRECY = 2.0


# ======================= QR target =======================
def qr_target(payload: str) -> np.ndarray:
    """128x128x1 float target: 0.0 = black module, 1.0 = white."""
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=1, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    m = np.array(qr.get_matrix(), dtype=np.float32)      # True(1.0) = black module
    white = 1.0 - m                                      # 0 = black, 1 = white
    t = tf.image.resize(white[..., None], (IMG, IMG), method="nearest")
    return t.numpy().astype(np.float32)


def random_payload() -> str:
    """A transaction-shaped payload (same fields/size as the real signed QR)."""
    return json.dumps({
        "txid": f"TX{random.randint(0, 10 ** 8):08d}",
        "amount": round(random.uniform(1, 1e6), 2),
        "hash": "%064x" % random.getrandbits(256),
        "signature": "%0128x" % random.getrandbits(512),
    })


def qr_batch(n: int) -> np.ndarray:
    return np.stack([qr_target(random_payload()) for _ in range(n)])


# ======================= overlay physics =======================
_LUM = tf.constant([0.299, 0.587, 0.114])


def luminance(x01):
    return tf.reduce_sum(x01 * _LUM, axis=-1, keepdims=True)


def overlay(a01, b01):
    """Physical transparency stacking: pixel-wise multiply."""
    return a01 * b01


def split_ab(gen_out):
    """6-ch tanh output in [-1,1] -> two RGB shares in [0,1]."""
    a = (gen_out[..., :3] + 1.0) / 2.0
    b = (gen_out[..., 3:] + 1.0) / 2.0
    return a, b


# ======================= models =======================
def _down(x, f, bn=True):
    x = layers.Conv2D(f, 4, strides=2, padding="same")(x)
    if bn:
        x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)


def _up(x, f):
    x = layers.Conv2DTranspose(f, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_generator():
    ca = layers.Input((IMG, IMG, 3), name="cover_a")
    cb = layers.Input((IMG, IMG, 3), name="cover_b")
    qr = layers.Input((IMG, IMG, 1), name="qr")
    z = layers.Input((IMG, IMG, 1), name="noise")
    x = layers.Concatenate()([ca, cb, qr, z])            # 8 channels

    e1 = _down(x, 64, bn=False)                          # 64
    e2 = _down(e1, 128)                                  # 32
    e3 = _down(e2, 256)                                  # 16
    e4 = _down(e3, 512)                                  # 8
    b = _down(e4, 512)                                   # 4

    d1 = layers.Concatenate()([_up(b, 512), e4])         # 8
    d2 = layers.Concatenate()([_up(d1, 256), e3])        # 16
    d3 = layers.Concatenate()([_up(d2, 128), e2])        # 32
    d4 = layers.Concatenate()([_up(d3, 64), e1])         # 64
    out = layers.Conv2DTranspose(6, 4, strides=2, padding="same",
                                 activation="tanh")(d4)   # 128, 6ch (A|B)
    return Model([ca, cb, qr, z], out, name="vc_generator")


def build_discriminator():
    """PatchGAN: is this single share a real landscape? (logits)"""
    inp = layers.Input((IMG, IMG, 3))
    x = _down(inp, 64, bn=False)
    x = _down(x, 128)
    x = _down(x, 256)
    out = layers.Conv2D(1, 4, padding="same")(x)         # patch logits
    return Model(inp, out, name="vc_discriminator")


def build_attacker():
    """Adversary: predict the QR from ONE share. Generator must defeat it."""
    inp = layers.Input((IMG, IMG, 3))
    e1 = _down(inp, 64, bn=False)
    e2 = _down(e1, 128)
    e3 = _down(e2, 256)
    d1 = layers.Concatenate()([_up(e3, 128), e2])
    d2 = layers.Concatenate()([_up(d1, 64), e1])
    out = layers.Conv2DTranspose(1, 4, strides=2, padding="same",
                                 activation="sigmoid")(d2)
    return Model(inp, out, name="vc_attacker")


# ======================= losses =======================
_bce = tf.keras.losses.BinaryCrossentropy()                 # probs in [0,1]
_bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def recon_loss(a01, b01, qr):
    # push luminance(A*B) -> 1 on white modules (both bright), -> 0 on black (>=1 dark)
    return _bce(qr, luminance(overlay(a01, b01)))


def content_loss(a01, b01, ca01, cb01):
    return tf.reduce_mean(tf.abs(a01 - ca01)) + tf.reduce_mean(tf.abs(b01 - cb01))


# ======================= train step =======================
def make_optimizers():
    return (tf.keras.optimizers.Adam(2e-4, beta_1=0.5),   # G
            tf.keras.optimizers.Adam(2e-4, beta_1=0.5),   # D
            tf.keras.optimizers.Adam(2e-4, beta_1=0.5))   # ATT


@tf.function
def train_step(models, opts, ca, cb, qr, z, real_imgs):
    G, D, ATT = models
    g_opt, d_opt, a_opt = opts

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as a_tape:
        out = G([ca, cb, qr, z], training=True)
        A, B = split_ab(out)

        # discriminator: real covers vs generated shares
        d_real = D(real_imgs, training=True)
        d_fake_a = D(A, training=True)
        d_fake_b = D(B, training=True)
        d_loss = (_bce_logits(tf.ones_like(d_real), d_real)
                  + 0.5 * _bce_logits(tf.zeros_like(d_fake_a), d_fake_a)
                  + 0.5 * _bce_logits(tf.zeros_like(d_fake_b), d_fake_b))

        # attacker: recover QR from a single (detached) share
        att_a = ATT(tf.stop_gradient(A), training=True)
        att_b = ATT(tf.stop_gradient(B), training=True)
        att_loss = _bce(qr, att_a) + _bce(qr, att_b)

        # generator
        adv = (_bce_logits(tf.ones_like(d_fake_a), d_fake_a)
               + _bce_logits(tf.ones_like(d_fake_b), d_fake_b))
        rec = recon_loss(A, B, qr)
        con = content_loss(A, B, ca, cb)
        # secrecy: make attacker unable to do better than 0.5 (no info) on each share
        half = 0.5 * tf.ones_like(qr)
        sec = _bce(half, ATT(A, training=False)) + _bce(half, ATT(B, training=False))
        g_loss = (LAMBDA_ADV * adv + LAMBDA_RECON * rec
                  + LAMBDA_CONTENT * con + LAMBDA_SECRECY * sec)

    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, G.trainable_variables),
                              G.trainable_variables))
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, D.trainable_variables),
                              D.trainable_variables))
    a_opt.apply_gradients(zip(a_tape.gradient(att_loss, ATT.trainable_variables),
                              ATT.trainable_variables))
    return g_loss, d_loss, att_loss, rec, con


# ======================= data =======================
def build_dataset(data_dir, batch=16):
    """tf.data of real landscape images in [0,1], 128x128x3."""
    files = [str(p) for p in Path(data_dir).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not files:
        raise FileNotFoundError(f"no images under {data_dir}")

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMG, IMG))
        return tf.cast(img, tf.float32) / 255.0

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(len(files)).map(load, tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE), len(files)


def train(data_dir, epochs=200, batch=16, log_every=50):
    ds, n = build_dataset(data_dir, batch)
    print(f"training on {n} images, {epochs} epochs, batch {batch}")
    G, D, ATT = build_generator(), build_discriminator(), build_attacker()
    opts = make_optimizers()
    step = 0
    for epoch in range(epochs):
        for imgs in ds:
            bs = tf.shape(imgs)[0]
            # ponytail: pair covers within the batch (reverse) and draw "real" by shuffle,
            #           avoids maintaining three datasets. Upgrade to 3 iterators if it matters.
            ca = imgs
            cb = tf.reverse(imgs, axis=[0])
            real = tf.random.shuffle(imgs)
            qr = tf.constant(qr_batch(int(bs)), tf.float32)
            z = tf.random.normal((bs, IMG, IMG, 1))
            g, d, a, rec, con = train_step((G, D, ATT), opts, ca, cb, qr, z, real)
            if step % log_every == 0:
                print(f"e{epoch} s{step}  g={float(g):.3f} d={float(d):.3f} "
                      f"att={float(a):.3f} rec={float(rec):.3f} con={float(con):.3f}")
            step += 1
    G.save_weights(WEIGHTS_FILE)
    print(f"saved {WEIGHTS_FILE}")
    return G


# ======================= smoke test =======================
def demo():
    """One train step on 2 synthetic images — pipeline check, runs on CPU."""
    G, D, ATT = build_generator(), build_discriminator(), build_attacker()
    opts = make_optimizers()
    ca = tf.random.uniform((2, IMG, IMG, 3))
    cb = tf.random.uniform((2, IMG, IMG, 3))
    real = tf.random.uniform((2, IMG, IMG, 3))
    qr = tf.constant(qr_batch(2), tf.float32)
    z = tf.random.normal((2, IMG, IMG, 1))

    out = G([ca, cb, qr, z], training=False)
    A, B = split_ab(out)
    assert out.shape == (2, IMG, IMG, 6), out.shape
    assert A.shape == (2, IMG, IMG, 3)
    assert luminance(overlay(A, B)).shape == (2, IMG, IMG, 1)

    g, d, a, rec, con = train_step((G, D, ATT), opts, ca, cb, qr, z, real)
    for name, v in [("g", g), ("d", d), ("att", a), ("rec", rec), ("con", con)]:
        assert np.isfinite(float(v)), f"{name} not finite"
    print(f"demo OK - shapes valid, losses finite "
          f"(g={float(g):.3f} d={float(d):.3f} att={float(a):.3f})")


if __name__ == "__main__":
    demo()
