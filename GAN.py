import json
from pathlib import Path
import numpy as np
import qrcode
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256

_HERE = Path(__file__).parent if "__file__" in locals() else Path.cwd()
KEY_FILE = _HERE / "signing_key.pem"
QR_FILE = _HERE / "transaction_qr.png"
SHARE_A = _HERE / "shareA.png"
SHARE_B = _HERE / "shareB.png"
RECON_FILE = _HERE / "reconstructed_qr.png"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
LATENT_DIM = 100

def _load_key():
    if KEY_FILE.exists():
        return ECC.import_key(KEY_FILE.read_text())
    key = ECC.generate(curve="p256")
    KEY_FILE.write_text(key.export_key(format="PEM"))
    return key

_key = _load_key()

def sign(message: bytes) -> bytes:
    digest = SHA256.new(message)
    return DSS.new(_key, "fips-186-3").sign(digest)

def verify(message: bytes, signature: bytes) -> bool:
    digest = SHA256.new(message)
    try:
        DSS.new(_key.public_key(), "fips-186-3").verify(digest, signature)
        return True
    except ValueError:
        return False

class Transaction:
    def __init__(self, txid="TXDEMO1", amount=10000.0):
        self.txid = txid
        self.amount = amount

    def model_dump_json(self):
        return json.dumps({"sender": "Alice", "receiver": "Bob", "amount": self.amount})

def make_qr(tx, signature: bytes, path: Path = QR_FILE) -> np.ndarray:
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
    return np.array(qr.get_matrix(), dtype=bool)

def split_shares(qr_matrix: np.ndarray) -> None:
    height, width = qr_matrix.shape
    share1 = np.zeros((height * 2, width * 2), dtype=bool)
    share2 = np.zeros((height * 2, width * 2), dtype=bool)

    block_a = np.array([[True, False], [False, True]])
    block_b = np.array([[False, True], [True, False]])

    for r in range(height):
        for c in range(width):
            is_black = qr_matrix[r, c]
            sub = block_a if np.random.rand() > 0.5 else block_b
            share1[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = sub
            share2[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = ~sub if is_black else sub

    overlay = np.bitwise_or(share1, share2)
    Image.fromarray(~share1).resize(IMAGE_SIZE).save(SHARE_A)
    Image.fromarray(~share2).resize(IMAGE_SIZE).save(SHARE_B)
    Image.fromarray(~overlay).resize(IMAGE_SIZE).save(RECON_FILE)

def load_and_prep_shares(share_a_path, share_b_path, target_size=IMAGE_SIZE):
    img_a = Image.open(share_a_path).convert('L').resize(target_size)
    img_b = Image.open(share_b_path).convert('L').resize(target_size)

    arr_a = (np.array(img_a, dtype=np.float32) - 127.5) / 127.5
    arr_b = (np.array(img_b, dtype=np.float32) - 127.5) / 127.5

    shares_stacked = np.stack([arr_a, arr_b], axis=-1)
    return shares_stacked

def generate_dummy_dataset(num_samples=500):
    shares_list = []
    for _ in range(num_samples):
        dummy_matrix = np.random.choice([True, False], size=(32, 32))
        split_shares(dummy_matrix)
        stacked = load_and_prep_shares(SHARE_A, SHARE_B)
        shares_list.append(stacked)
    
    landscapes = np.random.uniform(-1.0, 1.0, size=(num_samples, 128, 128, 3)).astype(np.float32)
    return np.array(shares_list, dtype=np.float32), landscapes

def build_generator():
    shares_input = layers.Input(shape=(128, 128, 2), name="shares_input")
    noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")

    x_noise = layers.Dense(128 * 128 * 1)(noise_input)
    x_noise = layers.Reshape((128, 128, 1))(x_noise)

    x = layers.Concatenate()([shares_input, x_noise])

    e1 = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    e1 = layers.LeakyReLU(0.2)(e1)

    e2 = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2)

    b = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(e2)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(0.2)(b)

    d1 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(b)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, e2])

    d2 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, e1])

    output_image = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh")(d2)

    return Model(inputs=[shares_input, noise_input], outputs=output_image, name="cGAN_Generator")

def build_discriminator():
    target_image = layers.Input(shape=(128, 128, 3), name="landscape_input")
    shares_condition = layers.Input(shape=(128, 128, 2), name="shares_condition_input")

    combined = layers.Concatenate()([target_image, shares_condition])

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(combined)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    output_score = layers.Conv2D(1, kernel_size=4, strides=1, padding="same", activation="sigmoid")(x)

    return Model(inputs=[target_image, shares_condition], outputs=output_score, name="cGAN_Discriminator")

msg = b'{"sender":"Alice","receiver":"Bob","amount":10000}'
sig = sign(msg)
tx = Transaction()
matrix = make_qr(tx, sig)
split_shares(matrix)

shares_data, landscape_data = generate_dummy_dataset(100)
dataset = tf.data.Dataset.from_tensor_slices((shares_data, landscape_data))
dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

generator = build_generator()
discriminator = build_discriminator()

g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(shares_batch, real_landscapes):
    batch_size = tf.shape(real_landscapes)[0]
    noise = tf.random.normal([batch_size, LATENT_DIM])

    with tf.GradientTape() as d_tape:
        fake_landscapes = generator([shares_batch, noise], training=True)

        real_output = discriminator([real_landscapes, shares_batch], training=True)
        fake_output = discriminator([fake_landscapes, shares_batch], training=True)

        d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
        d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    noise = tf.random.normal([batch_size, LATENT_DIM])

    with tf.GradientTape() as g_tape:
        fake_landscapes = generator([shares_batch, noise], training=True)
        fake_output = discriminator([fake_landscapes, shares_batch], training=True)

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss

EPOCHS = 2

for epoch in range(EPOCHS):
    for shares_batch, landscape_batch in dataset:
        d_loss, g_loss = train_step(shares_batch, landscape_batch)

real_shares_input = load_and_prep_shares(SHARE_A, SHARE_B)
real_shares_tensor = np.expand_dims(real_shares_input, axis=0)

sample_noise = tf.random.normal([1, LATENT_DIM])
generated_landscape = generator([real_shares_tensor, sample_noise], training=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(Image.open(SHARE_A), cmap="gray")
axes[0].set_title("Share A")
axes[0].axis("off")

axes[1].imshow(Image.open(SHARE_B), cmap="gray")
axes[1].set_title("Share B")
axes[1].axis("off")

output_img = (generated_landscape[0].numpy() + 1.0) / 2.0
axes[2].imshow(output_img)
axes[2].set_title("Generated Landscape")
axes[2].axis("off")

plt.show()