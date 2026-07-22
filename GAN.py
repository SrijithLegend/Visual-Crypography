import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub


dataset_dir = kagglehub.dataset_download("arnaud58/landscape-pictures")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

raw_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels=None,
    label_mode=None,
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

def normalize_image(image):
    image = tf.cast(image, tf.float32)
    return (image - 127.5) / 127.5

dataset = raw_dataset.map(
    normalize_image, 
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(buffer_size=tf.data.AUTOTUNE)

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(100,)),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(128 * 128 * 3, activation="tanh"),
    tf.keras.layers.Reshape((128, 128, 3))
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 3)),

    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)

loss_fn = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
        d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)

        d_loss = d_loss_real + d_loss_fake

    gradients = d_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    d_optimizer.apply_gradients(
        zip(gradients, discriminator.trainable_variables)
    )

    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

    gradients = g_tape.gradient(
        g_loss,
        generator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(gradients, generator.trainable_variables)
    )

    return d_loss, g_loss

EPOCHS = 30

for epoch in range(EPOCHS):
    for images in dataset:
        d_loss, g_loss = train_step(images)

    print(
        f"Epoch {epoch+1} | "
        f"D Loss: {d_loss:.4f} | "
        f"G Loss: {g_loss:.4f}"
    )

noise = tf.random.normal([1, 100])
generated = generator(noise, training=False)

img = (generated[0] + 1) / 2.0

plt.imshow(img)
plt.axis("off")
plt.show()